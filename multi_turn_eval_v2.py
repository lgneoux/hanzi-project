# ==========================================================================
# multi_turn_eval_v2.py
# ==========================================================================
# 多轮对话版汉字结构测评 v2
#
# 最终输出：
#       eval_multiturn_results_v2/sample_100.json
#       eval_multiturn_results_v2/multiturn_{model}.jsonl
#       eval_multiturn_results_v2/multiturn_summary.json
#       eval_multiturn_results_v2/multiturn_summary.csv
#       eval_multiturn_results_v2/error_summary.json
#       eval_multiturn_results_v2/failed_final_{model}.jsonl
#
# ==========================================================================

import csv
import json
import os
import random
import re
import time
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


# ==========================================================================
# 1. 基础配置区
# ==========================================================================

PRIMARY_API_KEY = "" 
BACKUP_API_KEY = ""     
BASE_URL = "https://aiberm.com/v1"

MODEL_LIST = [
    "claude-opus-4-6",
    "gemini-3-flash-preview",
    "glm-5.1",
    "gemini-3-pro-preview",
    "gpt-5.4",
    "deepseek-v3.2",
    "kimi-k2.5"
]

INPUT_FILE = "total_result.json"
CHAR_DATA_FILE = "char_data1.json"   # 可选；如果存在，会额外统计第一轮部首识别准确率
OUTPUT_DIR = "eval_multiturn_results_v2"

SAMPLE_SIZE = 100
RANDOM_SEED = 20260611

MAX_WORKERS = 8

# 大轮次重跑次数：不是单条内部无限 retry
MAX_TRIES = 2

# 单次 API 内部仅用于处理偶发网络/限额错误，次数不要太多
API_MAX_RETRIES = 4

REQUEST_TIMEOUT = 200

# 是否断点续跑：如果结果文件已有某个 id 的 success 记录，就不再跑
RESUME = True

# 可疑长输出 / 短输出判定，用于 error 统计和是否重跑
SUSPICIOUS_LONG_MULTIPLIER = 3
SUSPICIOUS_LONG_EXTRA = 30

# 是否在最终答案比期望长时截断。仅用于清洗，不使用标准答案。
TRUNCATE_OVERLONG_TO_EXPECTED_LEN = True

# 如果 3 轮都失败，是否把“最后一次尝试”写入最终文件
WRITE_LAST_ATTEMPT_IF_FAILED = True


# ==========================================================================
# 2. 客户端初始化
# ==========================================================================

primary_client = OpenAI(
    api_key=PRIMARY_API_KEY,
    base_url=BASE_URL if BASE_URL != "YOUR_BASE_URL_HERE" else None,
)

backup_client = OpenAI(
    api_key=BACKUP_API_KEY,
    base_url=BASE_URL if BASE_URL != "YOUR_BASE_URL_HERE" else None,
) if BACKUP_API_KEY else None


# ==========================================================================
# 3. CJK 汉字清洗：保留扩展区
# ==========================================================================

HANZI_CLEAN_PATTERN = (
    r"[^\u3007"
    r"\u3400-\u4DBF"
    r"\u4E00-\u9FFF"
    r"\uF900-\uFAFF"
    r"\U00020000-\U0002A6DF"
    r"\U0002A700-\U0002B73F"
    r"\U0002B740-\U0002B81F"
    r"\U0002B820-\U0002CEAF"
    r"\U0002CEB0-\U0002EBEF"
    r"\U0002F800-\U0002FA1F"
    r"\U00030000-\U0003134F"
    r"]"
)


def clean_hanzi_text(text: str) -> str:
    return re.sub(HANZI_CLEAN_PATTERN, "", text or "")


def strip_common_answer_prefix(text: str) -> str:
    prefixes = [
        "最终答案",
        "答案",
        "结果",
        "输出",
        "预测结果",
        "模型输出",
        "完整词语",
        "处理结果",
        "变换结果",
        "变换后",
        "直接输出",
    ]

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
                changed = True

    return text


def sort_key_by_id(item: dict):
    item_id = item.get("id")
    try:
        return (0, int(item_id))
    except Exception:
        return (1, str(item_id))


# ==========================================================================
# 4. 文件读写
# ==========================================================================

def load_json(filepath: str, default=None) -> Any:
    if not os.path.exists(filepath):
        if default is not None:
            return default
        raise FileNotFoundError(f"找不到文件：{filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            return json.load(f)


def read_jsonl(filepath: str) -> List[dict]:
    items = []
    if not os.path.exists(filepath):
        return items

    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️ 跳过坏 JSON 行：{filepath} 第 {line_no} 行")

    return items


def rewrite_jsonl(filepath: str, items: List[dict]) -> None:
    items = sorted(items, key=sort_key_by_id)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(filepath: str, item: dict) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()


def write_json(filepath: str, data: Any) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ==========================================================================
# 5. 抽样：覆盖短中长
# ==========================================================================

def length_bucket(n: int) -> str:
    if n <= 3:
        return "short_1_3"
    if n <= 6:
        return "medium_4_6"
    if n <= 10:
        return "long_7_10"
    return "very_long_11_plus"


def stratified_sample(data: List[dict], sample_size: int, seed: int) -> List[dict]:
    rng = random.Random(seed)

    buckets = {
        "short_1_3": [],
        "medium_4_6": [],
        "long_7_10": [],
        "very_long_11_plus": [],
    }

    for item in data:
        phrase = item.get("phrase", "")
        replace = item.get("replace", "")
        change = item.get("change", "")

        if not phrase or not replace or not change:
            continue

        if len(phrase) != len(replace):
            continue

        buckets[length_bucket(len(phrase))].append(item)

    for items in buckets.values():
        rng.shuffle(items)

    target_alloc = {
        "short_1_3": 20,
        "medium_4_6": 30,
        "long_7_10": 25,
        "very_long_11_plus": 25,
    }

    if sample_size != 100:
        scale = sample_size / 100
        target_alloc = {k: max(1, round(v * scale)) for k, v in target_alloc.items()}

    selected = []
    used_ids = set()

    for bucket_name, target_n in target_alloc.items():
        chosen = buckets[bucket_name][:target_n]
        for item in chosen:
            selected.append(item)
            used_ids.add(item.get("id"))

    if len(selected) < sample_size:
        rest = [
            item for items in buckets.values()
            for item in items
            if item.get("id") not in used_ids
        ]
        rng.shuffle(rest)
        selected.extend(rest[:sample_size - len(selected)])

    return sorted(selected[:sample_size], key=sort_key_by_id)


# ==========================================================================
# 6. Prompt
# ==========================================================================

SYSTEM_PROMPT = (
    "你是一个精通汉字结构、部首和汉字变换规则的语言学专家。"
    "你必须严格按照用户要求输出，不要输出解释。"
)


def build_radical_prompt(phrase: str) -> str:
    n = len(phrase)
    return f"""下面 {n} 个字的部首分别是什么：{phrase}

请直接输出这 {n} 个字对应的部首，顺序必须和原词一致。
禁止解释，禁止标点，禁止空格，禁止编号，禁止 Markdown。
只输出部首串本体。"""


def build_transform_prompt_v2(phrase: str, change: str, radical_answer: str) -> str:
    """
    v2 明确第二轮指代，避免模型把第一轮 assistant 的部首串当成待变换对象。
    """
    n = len(phrase)
    return f"""刚才讨论的原词是：{phrase}
你刚才给出的部首识别结果是：{radical_answer}

现在请把原词“{phrase}”中的每一个汉字的部首变成“{change}”。

请严格按下面规则处理原词中的每一个汉字：
1. 如果该字的当前部首刚好是目标部首：删去该部首。删去后必须是完整字，否则保持原字不变。
2. 如果该字加上目标部首能组成新字：加上该部首。
3. 如果该字的当前部首替换为目标部首能组成新字：替换该部首。
4. 如果以上都不满足：保持原字不变。

注意：
- 如果既能加又能换，优先加。
- 必须处理的是原词“{phrase}”，不是部首串“{radical_answer}”。
- 输出字数必须和原词完全一致：{n} 个汉字。
- 只输出最终变换后的完整词语。
- 禁止解释、分析、标点、空格、英文、Markdown。"""


# ==========================================================================
# 7. API 调用
# ==========================================================================

def should_switch_to_backup(err: Exception, attempt: int) -> bool:
    if not backup_client:
        return False

    err_str = str(err).lower()
    fatal_keywords = [
        "quota",
        "401",
        "403",
        "unauthorized",
        "invalid",
        "balance",
        "insufficient",
    ]

    is_fatal = any(k in err_str for k in fatal_keywords)
    is_fatigue = attempt >= (API_MAX_RETRIES // 2)

    return is_fatal or is_fatigue


def chat_completion(model_name: str, messages: List[dict], max_tokens: int) -> str:
    active_client = primary_client
    last_error = None

    for attempt in range(API_MAX_RETRIES):
        try:
            response = active_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=REQUEST_TIMEOUT,
            )

            content = (response.choices[0].message.content or "").strip()

            if not content:
                raise ValueError("模型返回了空字符串")

            return content

        except Exception as e:
            last_error = e

            if should_switch_to_backup(e, attempt):
                active_client = backup_client

            if attempt < API_MAX_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))

    return f"ERROR: 请求失败。最后报错: {str(last_error)}"


# ==========================================================================
# 8. 清洗和评分
# ==========================================================================

def clean_to_expected_len(raw_text: str, expected_len: int, phrase: str = "") -> Tuple[str, str]:
    """
    把模型输出清洗为 expected_len 个汉字。
    不使用 replace，只使用长度约束。
    """
    if not raw_text:
        return "ERROR: 空预测值", "empty"

    if raw_text.startswith("ERROR:"):
        return raw_text, "api_error"

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    exact_candidates = []
    phrase_like_candidate = None

    # 多个等长候选时，优先取最后一个非 phrase 候选，适配“初稿 + redo + 最终答案”
    for idx, line in enumerate(lines):
        candidate = strip_common_answer_prefix(clean_hanzi_text(line))

        if len(candidate) == expected_len:
            if candidate != phrase:
                exact_candidates.append((idx, candidate))
            elif phrase_like_candidate is None:
                phrase_like_candidate = candidate

    if exact_candidates:
        idx, candidate = exact_candidates[-1]
        if len(exact_candidates) == 1:
            return candidate, f"line_exact_{idx}"
        return candidate, f"line_exact_last_of_{len(exact_candidates)}_{idx}"

    all_hanzi = strip_common_answer_prefix(clean_hanzi_text(raw_text))
    if len(all_hanzi) == expected_len:
        return all_hanzi, "all_exact"

    # 只对明显像答案行的长输出做截断
    for idx, line in enumerate(lines):
        candidate = strip_common_answer_prefix(clean_hanzi_text(line))

        if len(candidate) > expected_len:
            looks_like_answer_line = (
                idx == 0
                or line.startswith("答案")
                or line.startswith("最终答案")
                or line.startswith("结果")
                or line.startswith("输出")
                or line.startswith("预测结果")
                or line.startswith("模型输出")
                or line.startswith("变换结果")
            )

            if looks_like_answer_line and TRUNCATE_OVERLONG_TO_EXPECTED_LEN:
                truncated = candidate[:expected_len]
                if truncated != phrase:
                    return truncated, f"line_truncate_{idx}"
                if phrase_like_candidate is None:
                    phrase_like_candidate = truncated

    if phrase_like_candidate is not None:
        return phrase_like_candidate, "phrase_like_only"

    return "ERROR: 未找到等长汉字答案", "no_candidate"


def calculate_phrase_score(prediction: str, truth: str) -> float:
    if not prediction or not truth or prediction.startswith("ERROR:"):
        return 0.0

    if len(truth) == 0:
        return 0.0

    score = 0.0
    char_weight = 1.0 / len(truth)

    for p_char, t_char in zip(prediction, truth):
        if p_char == t_char:
            score += char_weight

    return score


def char_level_correct(prediction: str, truth: str) -> Tuple[int, int]:
    total = len(truth)

    if not prediction or prediction.startswith("ERROR:"):
        return 0, total

    correct = 0
    for p_char, t_char in zip(prediction, truth):
        if p_char == t_char:
            correct += 1

    return correct, total


def get_truth_radicals(phrase: str, char_data: dict) -> Optional[str]:
    if not char_data:
        return None

    radicals = []
    for ch in phrase:
        info = char_data.get(ch)
        if not info or "radical" not in info:
            return None
        radicals.append(info.get("radical", ""))

    return "".join(radicals)


# ==========================================================================
# 9. Error 评测标准
# ==========================================================================

def is_suspicious_long(raw_text: str, expected_len: int) -> bool:
    if not raw_text or raw_text.startswith("ERROR:"):
        return False
    return (
        len(raw_text) > expected_len * SUSPICIOUS_LONG_MULTIPLIER
        or len(raw_text) > expected_len + SUSPICIOUS_LONG_EXTRA
    )


def is_suspicious_short_hanzi(cleaned_text: str, expected_len: int) -> bool:
    if not cleaned_text or cleaned_text.startswith("ERROR:"):
        return False
    return len(cleaned_text) < expected_len


def classify_attempt_error(attempt: dict) -> Tuple[bool, List[str]]:
    """
    返回：
        should_retry: 是否需要下一轮重跑
        error_types: 错误标签列表

    规则：
    只要出现任何 error 类型，都进入下一轮重测。
    包括：
        - api_error
        - radical_clean_error
        - transform_clean_error
        - suspicious_short_output
        - suspicious_long_output
    """
    error_types = []

    radical_raw = safe_text(attempt.get("radical_prediction_raw", ""))
    radical_clean = safe_text(attempt.get("radical_prediction_cleaned", ""))
    transform_raw = safe_text(attempt.get("model_prediction_raw", ""))
    transform_clean = safe_text(attempt.get("cleaned_prediction", ""))

    try:
        expected_len = int(attempt.get("phrase_len", 0))
    except Exception:
        expected_len = 0

    if radical_raw.startswith("ERROR:") or transform_raw.startswith("ERROR:"):
        error_types.append("api_error")

    if radical_clean.startswith("ERROR:"):
        error_types.append("radical_clean_error")

    if transform_clean.startswith("ERROR:"):
        error_types.append("transform_clean_error")

    if is_suspicious_long(transform_raw, expected_len):
        error_types.append("suspicious_long_output")

    if is_suspicious_short_hanzi(transform_clean, expected_len):
        error_types.append("suspicious_short_output")

    # 关键修改：只要有任何 error，就重跑
    should_retry = (
    "api_error" in error_types
    or "radical_clean_error" in error_types
    or "transform_clean_error" in error_types
    or "suspicious_short_output" in error_types
)

    return should_retry, error_types


# ==========================================================================
# 10. 单条多轮评测
# ==========================================================================

def run_one_attempt(model_name: str, item: dict, char_data: dict, attempt_no: int) -> dict:
    phrase = item["phrase"]
    change = item["change"]
    truth = item["replace"]
    expected_len = len(phrase)

    user_1 = build_radical_prompt(phrase)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_1},
    ]

    radical_raw = chat_completion(
        model_name=model_name,
        messages=messages,
        max_tokens=3000
    )

    radical_clean, radical_clean_note = clean_to_expected_len(
        radical_raw,
        expected_len=expected_len,
        phrase="",
    )

    # 真实多轮：把第一轮 assistant 的原始回答放回对话历史
    messages.append({"role": "assistant", "content": radical_raw})

    user_2 = build_transform_prompt_v2(phrase, change, radical_raw)
    messages.append({"role": "user", "content": user_2})

    transform_raw = chat_completion(
        model_name=model_name,
        messages=messages,
        max_tokens=6000,
    )

    transform_clean, transform_clean_note = clean_to_expected_len(
        transform_raw,
        expected_len=expected_len,
        phrase=phrase,
    )

    score = calculate_phrase_score(transform_clean, truth)
    correct_chars, total_chars = char_level_correct(transform_clean, truth)

    truth_radicals = get_truth_radicals(phrase, char_data)
    radical_score = None
    radical_correct_chars = None
    radical_total_chars = None

    if truth_radicals is not None:
        radical_score = calculate_phrase_score(radical_clean, truth_radicals)
        radical_correct_chars, radical_total_chars = char_level_correct(radical_clean, truth_radicals)

    attempt = {
        "attempt_no": attempt_no,
        "id": item.get("id"),
        "model": model_name,
        "phrase": phrase,
        "change": change,
        "replace": truth,
        "phrase_len": expected_len,
        "length_bucket": length_bucket(expected_len),

        "turns": [
            {
                "role": "user",
                "content": user_1,
            },
            {
                "role": "assistant",
                "content": radical_raw,
                "cleaned": radical_clean,
                "cleaning_note": radical_clean_note,
            },
            {
                "role": "user",
                "content": user_2,
            },
            {
                "role": "assistant",
                "content": transform_raw,
                "cleaned": transform_clean,
                "cleaning_note": transform_clean_note,
            },
        ],

        "radical_prediction_raw": radical_raw,
        "radical_prediction_cleaned": radical_clean,
        "radical_cleaning_note": radical_clean_note,
        "truth_radicals": truth_radicals,
        "radical_score": radical_score,
        "radical_correct_chars": radical_correct_chars,
        "radical_total_chars": radical_total_chars,

        "model_prediction_raw": transform_raw,
        "cleaned_prediction": transform_clean,
        "cleaning_note": transform_clean_note,
        "score": score,
        "correct_chars": correct_chars,
        "total_chars": total_chars,
    }

    should_retry, error_types = classify_attempt_error(attempt)
    attempt["should_retry"] = should_retry
    attempt["error_types"] = error_types
    attempt["status"] = "error" if should_retry else "success"

    return attempt


def build_final_result_from_attempts(item: dict, model_name: str, attempts: List[dict]) -> dict:
    """
    从多次 attempt 中选择最终结果：
    - 优先选择最后一次 success
    - 如果没有 success，选择最后一次 attempt
    - 保留 attempts 全历史
    """
    success_attempts = [a for a in attempts if a.get("status") == "success"]

    if success_attempts:
        final = success_attempts[-1].copy()
        final["final_status"] = "success"
    elif attempts:
        final = attempts[-1].copy()
        final["final_status"] = "final_error_after_retries"
        final["error_types"] = sorted(set(final.get("error_types", []) + ["final_error_after_retries"]))
    else:
        phrase = item.get("phrase", "")
        truth = item.get("replace", "")
        final = {
            "id": item.get("id"),
            "model": model_name,
            "phrase": phrase,
            "change": item.get("change"),
            "replace": truth,
            "phrase_len": len(phrase),
            "length_bucket": length_bucket(len(phrase)),
            "turns": [],
            "attempt_no": 0,
            "attempts": [],
            "model_prediction_raw": "ERROR: 没有任何 attempt",
            "cleaned_prediction": "ERROR: 没有任何 attempt",
            "score": 0.0,
            "correct_chars": 0,
            "total_chars": len(truth),
            "status": "error",
            "final_status": "final_error_after_retries",
            "error_types": ["final_error_after_retries"],
        }

    final["attempts"] = attempts
    final["num_attempts"] = len(attempts)

    # final 结果不再需要 should_retry
    final.pop("should_retry", None)

    return final


# ==========================================================================
# 11. 单模型运行：大轮次重跑 error 样本
# ==========================================================================

def summarize_results(model_name: str, results: List[dict]) -> dict:
    total_items = len(results)
    total_score = sum(item.get("score", 0.0) for item in results)
    avg_score = total_score / total_items if total_items else 0.0

    total_correct_chars = sum(item.get("correct_chars", 0) for item in results)
    total_chars = sum(item.get("total_chars", 0) for item in results)
    char_accuracy = total_correct_chars / total_chars if total_chars else 0.0

    radical_items = [item for item in results if item.get("radical_score") is not None]
    radical_avg_score = None
    radical_char_accuracy = None

    if radical_items:
        radical_avg_score = sum(item["radical_score"] for item in radical_items) / len(radical_items)
        radical_correct = sum(item.get("radical_correct_chars", 0) for item in radical_items)
        radical_total = sum(item.get("radical_total_chars", 0) for item in radical_items)
        radical_char_accuracy = radical_correct / radical_total if radical_total else 0.0

    error_counts = {}
    for item in results:
        for error_type in item.get("error_types", []):
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        if item.get("final_status") == "success":
            error_counts["success"] = error_counts.get("success", 0) + 1

    bucket_summary = {}
    for bucket in ["short_1_3", "medium_4_6", "long_7_10", "very_long_11_plus"]:
        bucket_items = [item for item in results if item.get("length_bucket") == bucket]
        if not bucket_items:
            continue
        bucket_summary[bucket] = {
            "n": len(bucket_items),
            "avg_score": sum(item.get("score", 0.0) for item in bucket_items) / len(bucket_items),
            "char_accuracy": (
                sum(item.get("correct_chars", 0) for item in bucket_items)
                / max(1, sum(item.get("total_chars", 0) for item in bucket_items))
            ),
        }

    return {
        "model": model_name,
        "total_items": total_items,
        "avg_score": avg_score,
        "char_accuracy": char_accuracy,
        "total_correct_chars": total_correct_chars,
        "total_chars": total_chars,
        "radical_avg_score": radical_avg_score,
        "radical_char_accuracy": radical_char_accuracy,
        "error_counts": error_counts,
        "bucket_summary": bucket_summary,
    }


def run_model(model_name: str, sample_items: List[dict], char_data: dict) -> Tuple[List[dict], dict]:
    print(f"\n🚀 多轮评测模型：{model_name}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"multiturn_{model_name}.jsonl")
    failed_file = os.path.join(OUTPUT_DIR, f"failed_final_{model_name}.jsonl")

    done_results = []
    done_ids = set()

    if RESUME and os.path.exists(output_file):
        existing = read_jsonl(output_file)

        # 只有 final_status=success 且没有任何 error_types 的样本才视为完成。
        # 如果旧结果虽然 success，但带有 radical_clean_error / suspicious_long_output 等，
        # 也会重新进入本轮测试。
        for item in existing:
            error_types = item.get("error_types", [])
            if item.get("final_status") == "success" and not error_types:
                done_results.append(item)
                done_ids.add(item.get("id"))

        print(f"🔁 检测到已有无 error 的 success 结果：{len(done_ids)} 条，将跳过这些 id。")

    pending = [item.copy() for item in sample_items if item.get("id") not in done_ids]
    attempts_by_id = {item.get("id"): [] for item in pending}

    accepted_by_id = {item.get("id"): item for item in done_results}

    for round_idx in range(1, MAX_TRIES + 1):
        if not pending:
            break

        print(f"\n🔁 第 {round_idx}/{MAX_TRIES} 轮：待处理 {len(pending)} 条")

        next_pending = []

        with tqdm(total=len(pending), desc=f"multiturn {model_name} round {round_idx}", unit="条") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_item = {
                    executor.submit(run_one_attempt, model_name, item.copy(), char_data, round_idx): item
                    for item in pending
                }

                for future in concurrent.futures.as_completed(future_to_item):
                    original_item = future_to_item[future]

                    try:
                        attempt = future.result()
                    except Exception as e:
                        phrase = original_item.get("phrase", "")
                        truth = original_item.get("replace", "")
                        attempt = {
                            "attempt_no": round_idx,
                            "id": original_item.get("id"),
                            "model": model_name,
                            "phrase": phrase,
                            "change": original_item.get("change"),
                            "replace": truth,
                            "phrase_len": len(phrase),
                            "length_bucket": length_bucket(len(phrase)),
                            "turns": [],
                            "radical_prediction_raw": f"ERROR: Future 异常: {str(e)}",
                            "radical_prediction_cleaned": f"ERROR: Future 异常: {str(e)}",
                            "model_prediction_raw": f"ERROR: Future 异常: {str(e)}",
                            "cleaned_prediction": f"ERROR: Future 异常: {str(e)}",
                            "score": 0.0,
                            "correct_chars": 0,
                            "total_chars": len(truth),
                            "status": "error",
                            "should_retry": True,
                            "error_types": ["api_error"],
                        }

                    item_id = attempt.get("id")
                    attempts_by_id.setdefault(item_id, []).append(attempt)

                    if attempt.get("status") == "success":
                        final_result = build_final_result_from_attempts(original_item, model_name, attempts_by_id[item_id])
                        accepted_by_id[item_id] = final_result
                    else:
                        next_pending.append(original_item.copy())

                    pbar.update(1)

        print(f"✅ 第 {round_idx} 轮结束：已成功 {len(accepted_by_id)} / {len(sample_items)}，下一轮待重跑 {len(next_pending)}")
        pending = next_pending

    # 仍失败的样本写入最终结果
    if pending and WRITE_LAST_ATTEMPT_IF_FAILED:
        print(f"⚠️ {len(pending)} 条样本在 {MAX_TRIES} 轮后仍失败，将写入 final_error。")
        for item in pending:
            item_id = item.get("id")
            final_result = build_final_result_from_attempts(item, model_name, attempts_by_id.get(item_id, []))
            accepted_by_id[item_id] = final_result

    results = [accepted_by_id[item.get("id")] for item in sample_items if item.get("id") in accepted_by_id]
    results = sorted(results, key=sort_key_by_id)

    rewrite_jsonl(output_file, results)

    failed_results = [item for item in results if item.get("final_status") != "success"]
    rewrite_jsonl(failed_file, failed_results)

    summary = summarize_results(model_name, results)

    print(f"✅ {model_name} 完成：平均逐题得分 {summary['avg_score']:.2%}，按字准确率 {summary['char_accuracy']:.2%}")
    print(f"🧪 Error counts: {summary['error_counts']}")

    return results, summary


# ==========================================================================
# 12. 汇总输出
# ==========================================================================

def write_summary_files(summaries: List[dict]) -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    summary_json = os.path.join(OUTPUT_DIR, "multiturn_summary.json")
    summary_csv = os.path.join(OUTPUT_DIR, "multiturn_summary.csv")
    error_json = os.path.join(OUTPUT_DIR, "error_summary.json")

    write_json(summary_json, summaries)

    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "total_items",
            "avg_score",
            "char_accuracy",
            "total_correct_chars",
            "total_chars",
            "radical_avg_score",
            "radical_char_accuracy",
            "success",
            "api_error",
            "radical_clean_error",
            "transform_clean_error",
            "suspicious_short_output",
            "suspicious_long_output",
            "final_error_after_retries",
        ])

        for s in summaries:
            ec = s.get("error_counts", {})
            writer.writerow([
                s.get("model"),
                s.get("total_items"),
                s.get("avg_score"),
                s.get("char_accuracy"),
                s.get("total_correct_chars"),
                s.get("total_chars"),
                s.get("radical_avg_score"),
                s.get("radical_char_accuracy"),
                ec.get("success", 0),
                ec.get("api_error", 0),
                ec.get("radical_clean_error", 0),
                ec.get("transform_clean_error", 0),
                ec.get("suspicious_short_output", 0),
                ec.get("suspicious_long_output", 0),
                ec.get("final_error_after_retries", 0),
            ])

    error_payload = {
        s["model"]: s.get("error_counts", {})
        for s in summaries
    }
    write_json(error_json, error_payload)

    print(f"\n📊 汇总 JSON 已写出：{summary_json}")
    print(f"📊 汇总 CSV 已写出：{summary_csv}")
    print(f"📊 Error 汇总已写出：{error_json}")


# ==========================================================================
# 13. 主程序
# ==========================================================================

def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    all_data = load_json(INPUT_FILE)
    char_data = load_json(CHAR_DATA_FILE, default={})

    sample_items = stratified_sample(all_data, SAMPLE_SIZE, RANDOM_SEED)

    sample_file = os.path.join(OUTPUT_DIR, "sample_100.json")
    write_json(sample_file, sample_items)

    print(f"✅ 已抽样 {len(sample_items)} 道，样本文件：{sample_file}")

    dist = {}
    for item in sample_items:
        bucket = length_bucket(len(item.get("phrase", "")))
        dist[bucket] = dist.get(bucket, 0) + 1
    print(f"📏 样本长度分布：{dist}")

    summaries = []

    for model_name in MODEL_LIST:
        _, summary = run_model(model_name, sample_items, char_data)
        summaries.append(summary)

    write_summary_files(summaries)

    print("\n🎉 多轮对话测评 v2 全部完成。")


if __name__ == "__main__":
    main()

# ==========================================================================
# multi_turn_eval_without_correct.py
# ==========================================================================
# 多轮对话版汉字结构测评
#
# 最终输出：
#       eval_multiturn_results_without_correct/sample_500.json
#       eval_multiturn_results_without_correct/multiturn_{model}.jsonl
#       eval_multiturn_results_without_correct/multiturn_summary.json
#       eval_multiturn_results_without_correct/multiturn_summary.csv
#       eval_multiturn_results_without_correct/error_summary.json
#       eval_multiturn_results_without_correct/failed_final_{model}.jsonl
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

PRIMARY_API_KEY = os.getenv("PRIMARY_API_KEY")
BACKUP_API_KEY = os.getenv("BACKUP_API_KEY")
BASE_URL = os.getenv("BASE_URL")

MODEL_LIST = [
    "anthropic/claude-opus-4.6",
    "google/gemini-3.1-pro-preview",
    "openai/gpt-5.4",
    "deepseek/deepseek-v3.2",
    "moonshotai/kimi-k2.5",
    "z-ai/glm-5.1"
]

INPUT_FILE = "total_result.json"
CHAR_DATA_FILE = "char_data1_patched.json"  
OUTPUT_DIR = "eval_multiturn_results_without_correct"

SAMPLE_SIZE = 500
RANDOM_SEED = 20260611

MAX_WORKERS = 8

# 每完成多少条任务保存一次中间结果
SAVE_EVERY = 10

# 大轮次重跑次数
MAX_TRIES = 2

# 单次 API 内部仅用于处理偶发网络/限额错误
API_MAX_RETRIES = 4

REQUEST_TIMEOUT = 200

# 清洗模型：仅用于从混杂输出中提取答案，不参与预测
CHECK_MODEL = "openai/gpt-5.4"
USE_CLEAN_MODEL = True
FORCE_CLEAN_MODEL_ON_RISKY_LONG_OUTPUT = True

# 第二轮对话历史中是否使用清洗后的部首回答。
# True：降低第一轮长解释/思考过程对第二轮的污染。
# False：保留严格真实多轮，把第一轮原始回答完整放回历史。
USE_CLEANED_RADICAL_IN_HISTORY = True

# 是否断点续跑
RESUME = True

# 可疑长输出 / 短输出判定，用于 error 统计和是否重跑
SUSPICIOUS_LONG_MULTIPLIER = 3
SUSPICIOUS_LONG_EXTRA = 30

# 是否在最终答案比期望长时截断。仅用于清洗，不使用标准答案。
TRUNCATE_OVERLONG_TO_EXPECTED_LEN = True

# 如果 MAX_TRIES 轮都失败，是否把“最后一次尝试”写入最终文件
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
    r"[^\u2E80-\u2EFF\u2F00-\u2FDF\u3007"
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


def safe_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "".join(safe_text(v) for v in x)
    if isinstance(x, tuple):
        return "".join(safe_text(v) for v in x)
    return str(x)


def safe_filename(name: str) -> str:
    """
    把模型名转换为安全文件名。
    例如 anthropic/claude-opus-4.6 -> anthropic_claude-opus-4.6
    避免模型名里的 / 在 Windows 或 Linux 中被当成目录分隔符。
    """
    name = safe_text(name)
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", name)
    name = name.strip("_")
    return name or "unknown_model"


def clean_hanzi_text(text: str) -> str:
    return re.sub(HANZI_CLEAN_PATTERN, "", safe_text(text))


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


def build_transform_prompt(phrase: str, change: str, radical_answer: str) -> str:
    """
    明确第二轮指代，避免模型把第一轮 assistant 的部首串当成待变换对象。
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

def build_clean_prompt(raw_text: str, phrase: str, expected_len: int, task_type: str = "transform") -> str:
    return f"""请从下面模型输出中提取最终答案。

原词：{phrase}
任务类型：{task_type}
要求：
1. 输出必须与原词长度一致：{expected_len} 个字符。
2. 只输出答案本体。
3. 不要输出解释、分析、标点、英文、Markdown。
4. 如果模型输出了思考过程，忽略思考过程，只保留最终答案。

模型输出：
{raw_text}
"""


def clean_with_check_model(raw_text: str, expected_len: int, phrase: str, task_type: str = "transform") -> Tuple[str, str]:
    result = chat_completion(
        model_name=CHECK_MODEL,
        messages=[
            {"role": "system", "content": "你是答案提取助手，只输出最终答案。"},
            {"role": "user", "content": build_clean_prompt(raw_text, phrase, expected_len, task_type)},
        ],
        max_tokens=1000,
    )
    if not result or result.startswith("ERROR:"):
        return "ERROR: 清洗模型失败", "clean_model_error"
    result = strip_common_answer_prefix(clean_hanzi_text(result))
    if len(result) == expected_len:
        return result, "clean_model_exact"
    if len(result) > expected_len:
        return result[:expected_len], "clean_model_truncate"
    return "ERROR: 清洗模型长度错误", "clean_model_bad_len"


def clean_to_expected_len_local(raw_text: str, expected_len: int, phrase: str = "") -> Tuple[str, str]:
    if not raw_text:
        return "ERROR: 空预测值", "empty"
    if raw_text.startswith("ERROR:"):
        return raw_text, "api_error"

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    candidates = []
    phrase_like = None

    for idx, line in enumerate(lines):
        candidate = strip_common_answer_prefix(clean_hanzi_text(line))
        if len(candidate) == expected_len:
            if candidate != phrase:
                candidates.append((idx, candidate))
            else:
                phrase_like = candidate

    if candidates:
        idx, ans = candidates[-1]
        return ans, f"line_exact_{idx}"

    all_hanzi = strip_common_answer_prefix(clean_hanzi_text(raw_text))
    if len(all_hanzi) == expected_len:
        return all_hanzi, "all_exact"

    for idx, line in enumerate(lines):
        candidate = strip_common_answer_prefix(clean_hanzi_text(line))
        if len(candidate) > expected_len and idx == 0:
            return candidate[:expected_len], f"line_truncate_{idx}"

    if phrase_like:
        return phrase_like, "phrase_like_only"

    return "ERROR: 未找到等长汉字答案", "no_candidate"


def clean_to_expected_len(raw_text: str, expected_len: int, phrase: str = "", task_type: str = "transform") -> Tuple[str, str]:
    local, note = clean_to_expected_len_local(raw_text, expected_len, phrase)
    raw_text = safe_text(raw_text)

    risky = (
        len(raw_text) > expected_len * SUSPICIOUS_LONG_MULTIPLIER
        or len(raw_text) > expected_len + SUSPICIOUS_LONG_EXTRA
    ) and ("truncate" in note or "last" in note or "phrase_like" in note)

    if local.startswith("ERROR:") or (USE_CLEAN_MODEL and FORCE_CLEAN_MODEL_ON_RISKY_LONG_OUTPUT and risky):
        model_ans, model_note = clean_with_check_model(raw_text, expected_len, phrase, task_type)
        if not model_ans.startswith("ERROR:"):
            return model_ans, model_note

    return local, note


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

# 统一定义：这些错误无论是在当前连续运行中，还是断点续跑时，
# 都必须被视为“尚未完成”，进入重跑。
RETRYABLE_ERROR_TYPES = {
    "api_error",
    "radical_clean_error",
    "transform_clean_error",
    "suspicious_short_output",
    "final_error_after_retries",
}


def has_retryable_error(error_types) -> bool:
    """
    当前运行与断点续跑共用同一套错误判断标准，
    避免出现“本轮被当作 success，但下次启动又被重跑”的不一致。
    """
    return bool(set(error_types or []) & RETRYABLE_ERROR_TYPES)


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


    if is_suspicious_short_hanzi(transform_clean, expected_len):
        error_types.append("suspicious_short_output")

    # 当前连续运行与断点续跑使用同一套标准。
    # 只要存在任何 RETRYABLE_ERROR_TYPES，就不能标记为 success。
    should_retry = has_retryable_error(error_types)

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
        phrase=phrase,
        task_type="radical",
    )

    # 第二轮对话历史：默认放入清洗后的部首串，避免第一轮长解释/思考过程污染第二轮。
    # 原始回答 radical_raw 仍会完整保存在结果字段里，便于后续排查。
    radical_history_content = radical_clean if USE_CLEANED_RADICAL_IN_HISTORY else radical_raw
    messages.append({"role": "assistant", "content": radical_history_content})

    user_2 = build_transform_prompt(phrase, change, radical_clean)
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
        task_type="transform",
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

    warning_types = []

    if is_suspicious_long(radical_raw, expected_len):
        warning_types.append("suspicious_long_radical_output")

    if is_suspicious_long(transform_raw, expected_len):
        warning_types.append("suspicious_long_transform_output")

    attempt["should_retry"] = should_retry
    attempt["error_types"] = error_types
    attempt["warning_types"] = warning_types
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

    warning_counts = {}

    for item in results:
        for warning_type in item.get("warning_types", []):
            warning_counts[warning_type] = (
                warning_counts.get(warning_type, 0) + 1
            )

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
        "warning_counts": warning_counts,
        "bucket_summary": bucket_summary,
    }


def run_model(
    model_name: str,
    sample_items: List[dict],
    char_data: dict,
) -> Tuple[List[dict], dict]:

    print(f"\n🚀 多轮评测模型：{model_name}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    safe_model_name = safe_filename(model_name)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"multiturn_{safe_model_name}.jsonl",
    )
    failed_file = os.path.join(
        OUTPUT_DIR,
        f"failed_final_{safe_model_name}.jsonl",
    )

    done_results = []
    done_ids = set()

    # ==============================================================
    # 1. 读取已有结果，用于断点续跑
    # ==============================================================

    if RESUME and os.path.exists(output_file):
        existing = read_jsonl(output_file)

        # 只有没有任何错误的成功结果才跳过
        for item in existing:
            error_types = item.get("error_types", [])

            if (
                item.get("final_status") == "success"
                and not error_types
            ):
                done_results.append(item)
                done_ids.add(item.get("id"))

        print(
            f"🔁 检测到已有可复用的无错误 success 结果："
            f"{len(done_ids)} 条，将跳过这些 id。"
        )

    pending = [
        item.copy()
        for item in sample_items
        if item.get("id") not in done_ids
    ]

    attempts_by_id = {
        item.get("id"): []
        for item in pending
    }

    accepted_by_id = {
        item.get("id"): item
        for item in done_results
    }

    # 从上一次保存之后，又完成了多少次任务
    completed_since_save = 0

    # ==============================================================
    # 2. 检查点保存函数
    # ==============================================================

    def save_checkpoint() -> None:
        """
        保存当前已经成功完成的结果。

        使用 rewrite_jsonl 覆盖写入，而不是 append_jsonl，
        避免同一个 id 因重试或断点续跑而重复出现。
        """
        checkpoint_results = sorted(
            accepted_by_id.values(),
            key=sort_key_by_id,
        )

        rewrite_jsonl(
            output_file,
            checkpoint_results,
        )

        tqdm.write(
            f"💾 已保存检查点："
            f"{len(checkpoint_results)} 条成功结果"
        )

    # ==============================================================
    # 3. 大轮次运行
    # ==============================================================

    for round_idx in range(1, MAX_TRIES + 1):
        if not pending:
            break

        print(
            f"\n🔁 第 {round_idx}/{MAX_TRIES} 轮："
            f"待处理 {len(pending)} 条"
        )

        next_pending = []

        with tqdm(
            total=len(pending),
            desc=f"multiturn {model_name} round {round_idx}",
            unit="条",
        ) as pbar:

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:

                future_to_item = {
                    executor.submit(
                        run_one_attempt,
                        model_name,
                        item.copy(),
                        char_data,
                        round_idx,
                    ): item
                    for item in pending
                }

                for future in concurrent.futures.as_completed(
                    future_to_item
                ):
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
                            "length_bucket": length_bucket(
                                len(phrase)
                            ),
                            "turns": [],

                            "radical_prediction_raw": (
                                f"ERROR: Future 异常: {str(e)}"
                            ),
                            "radical_prediction_cleaned": (
                                f"ERROR: Future 异常: {str(e)}"
                            ),
                            "model_prediction_raw": (
                                f"ERROR: Future 异常: {str(e)}"
                            ),
                            "cleaned_prediction": (
                                f"ERROR: Future 异常: {str(e)}"
                            ),

                            "score": 0.0,
                            "correct_chars": 0,
                            "total_chars": len(truth),

                            "status": "error",
                            "should_retry": True,
                            "error_types": ["api_error"],
                            "warning_types": [],
                        }

                    item_id = attempt.get("id")

                    attempts_by_id.setdefault(
                        item_id,
                        [],
                    ).append(attempt)

                    # 成功：写入 accepted_by_id
                    if attempt.get("status") == "success":
                        final_result = build_final_result_from_attempts(
                            original_item,
                            model_name,
                            attempts_by_id[item_id],
                        )

                        accepted_by_id[item_id] = final_result

                    # 失败：进入下一大轮重跑
                    else:
                        next_pending.append(
                            original_item.copy()
                        )

                    # 每完成一条 future，都增加计数
                    completed_since_save += 1

                    # 每完成 SAVE_EVERY 条，保存一次
                    if completed_since_save >= SAVE_EVERY:
                        save_checkpoint()
                        completed_since_save = 0

                    pbar.update(1)

        # 本轮结束时强制保存一次
        # 防止最后不足 SAVE_EVERY 条的结果没有落盘
        save_checkpoint()
        completed_since_save = 0

        print(
            f"✅ 第 {round_idx} 轮结束："
            f"已成功 {len(accepted_by_id)} / "
            f"{len(sample_items)}，"
            f"下一轮待重跑 {len(next_pending)}"
        )

        pending = next_pending

    # ==============================================================
    # 4. 达到最大轮次后，仍然失败的样本写入最终结果
    # ==============================================================

    if pending and WRITE_LAST_ATTEMPT_IF_FAILED:
        print(
            f"⚠️ {len(pending)} 条样本在 "
            f"{MAX_TRIES} 轮后仍失败，"
            f"将写入 final_error。"
        )

        for item in pending:
            item_id = item.get("id")

            final_result = build_final_result_from_attempts(
                item,
                model_name,
                attempts_by_id.get(item_id, []),
            )

            accepted_by_id[item_id] = final_result

        # 最终失败项加入后立即保存
        save_checkpoint()

    # ==============================================================
    # 5. 整理完整结果
    # ==============================================================

    results = [
        accepted_by_id[item.get("id")]
        for item in sample_items
        if item.get("id") in accepted_by_id
    ]

    results = sorted(
        results,
        key=sort_key_by_id,
    )

    # 最终再完整覆盖保存一次
    rewrite_jsonl(
        output_file,
        results,
    )

    # ==============================================================
    # 6. 单独保存最终失败项
    # ==============================================================

    failed_results = [
        item
        for item in results
        if item.get("final_status") != "success"
    ]

    rewrite_jsonl(
        failed_file,
        failed_results,
    )

    # ==============================================================
    # 7. 汇总统计
    # ==============================================================

    summary = summarize_results(
        model_name,
        results,
    )

    print(
        f"✅ {model_name} 完成："
        f"平均逐题得分 {summary['avg_score']:.2%}，"
        f"按字准确率 {summary['char_accuracy']:.2%}"
    )
    print(
        f"🧪 Error counts: "
        f"{summary['error_counts']}"
    )
    print(
        f"⚠️ Warning counts: "
        f"{summary['warning_counts']}"
    )

    return results, summary

# ==========================================================================
# 12. 汇总输出
# ==========================================================================

def write_summary_files(summaries: List[dict]) -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    summary_json = os.path.join(
        OUTPUT_DIR,
        "multiturn_summary.json",
    )
    summary_csv = os.path.join(
        OUTPUT_DIR,
        "multiturn_summary.csv",
    )
    error_json = os.path.join(
        OUTPUT_DIR,
        "error_summary.json",
    )
    warning_json = os.path.join(
        OUTPUT_DIR,
        "warning_summary.json",
    )

    write_json(summary_json, summaries)

    with open(
        summary_csv,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
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

            # 成功和真正的流程错误
            "success",
            "api_error",
            "radical_clean_error",
            "transform_clean_error",
            "suspicious_short_output",
            "final_error_after_retries",

            # 只记录、不触发重跑的警告
            "suspicious_long_radical_output",
            "suspicious_long_transform_output",
        ])

        for s in summaries:
            ec = s.get("error_counts", {})
            wc = s.get("warning_counts", {})

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
                ec.get("final_error_after_retries", 0),

                wc.get("suspicious_long_radical_output", 0),
                wc.get("suspicious_long_transform_output", 0),
            ])

    error_payload = {
        s["model"]: s.get("error_counts", {})
        for s in summaries
    }

    warning_payload = {
        s["model"]: s.get("warning_counts", {})
        for s in summaries
    }

    write_json(error_json, error_payload)
    write_json(warning_json, warning_payload)

    print(f"\n📊 汇总 JSON 已写出：{summary_json}")
    print(f"📊 汇总 CSV 已写出：{summary_csv}")
    print(f"📊 Error 汇总已写出：{error_json}")
    print(f"⚠️ Warning 汇总已写出：{warning_json}")


# ==========================================================================
# 13. 主程序
# ==========================================================================

def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    all_data = load_json(INPUT_FILE)
    char_data = load_json(CHAR_DATA_FILE, default={})

    sample_items = stratified_sample(all_data, SAMPLE_SIZE, RANDOM_SEED)

    sample_file = os.path.join(OUTPUT_DIR, "sample_500.json")
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

    print("\n🎉 多轮对话测评全部完成。")


if __name__ == "__main__":
    main()

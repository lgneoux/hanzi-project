# ==========================================================================
# Phase 2: 汉字结构测评系统 —— 文本清洗阶段
# ==========================================================================
# 输入：eval_results/raw_{model}.jsonl
# 输出：eval_results/clean_{model}.jsonl

# ==========================================================================

import json
import os
import time
import re
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

if not PRIMARY_API_KEY:
    raise RuntimeError("请先设置环境变量 PRIMARY_API_KEY")

MODEL_LIST = [
    "claude-opus-4-6",
    "gemini-3-flash-preview",
    "glm-4.7",
    "gemini-3-pro-preview",
    "gpt-5.4",
    "deepseek-v3.2",
    "kimi-k2.5"
]

CHECK_MODEL = "gpt-5.4"

OUTPUT_DIR = "eval_results"

MAX_WORKERS = 10
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

# 是否物理清理已有 clean 文件里的无效/过期记录
CLEAN_BAD_HISTORY = True

# 如果本地或清洗模型得到的答案比 phrase 长，是否截断到 phrase 长度
TRUNCATE_OVERLONG_TO_EXPECTED_LEN = True

# 保留完整 CJK 汉字范围：
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
    r"\U00031350-\U000323AF"
    r"]"
)


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
# 3. 通用工具函数
# ==========================================================================

def sort_key_by_id(item: dict):
    item_id = item.get("id")
    try:
        return (0, int(item_id))
    except Exception:
        return (1, str(item_id))


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


def is_valid_prediction_text(text: str) -> bool:
    return bool(text) and not text.startswith("ERROR:")


def dedupe_items_by_id_prefer_valid(items: List[dict], pred_key: str) -> List[dict]:
    """
    同一个 id 如果重复出现：
    - 优先保留 pred_key 不是 ERROR 的记录
    - 如果都是有效，保留最后出现的
    - 如果都是无效，也保留最后出现的
    """
    by_id = {}

    for item in items:
        item_id = item.get("id")
        if item_id is None:
            continue

        old = by_id.get(item_id)
        if old is None:
            by_id[item_id] = item
            continue

        old_valid = is_valid_prediction_text(old.get(pred_key, ""))
        new_valid = is_valid_prediction_text(item.get(pred_key, ""))

        if new_valid or not old_valid:
            by_id[item_id] = item

    return sorted(by_id.values(), key=sort_key_by_id)


def rewrite_jsonl(filepath: str, items: List[dict]) -> None:
    items = sorted(items, key=sort_key_by_id)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl_line(file_handle, item: dict) -> None:
    file_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    file_handle.flush()


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
    is_fatigue = attempt >= (MAX_RETRIES // 2)

    return is_fatal or is_fatigue


def clean_hanzi_text(text: str) -> str:
    """
    清除非汉字字符，但保留 CJK 扩展区汉字。
    例如：clean_hanzi_text("招拎四㧍") == "招拎四㧍"
    """
    return re.sub(HANZI_CLEAN_PATTERN, "", text or "")


def strip_common_answer_prefix(text: str) -> str:
    """
    只从候选串开头去掉常见提示词，避免“答案洪洞汫涉”被当作答案。
    注意：不会去掉中间的汉字，避免破坏真实答案。
    """
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



def is_hanzi_char(ch: str) -> bool:
    """
    判断单个字符是否属于当前允许保留的汉字范围。
    支持 CJK 扩展区汉字，例如：㧍。
    """
    return bool(ch) and clean_hanzi_text(ch) == ch


def leading_hanzi_run(text: str) -> str:
    """
    提取原始文本开头连续出现的汉字串。
    只要开头第一个有效字符不是汉字，就返回空串。
    """
    text = (text or "").strip()
    chars = []

    for ch in text:
        if is_hanzi_char(ch):
            chars.append(ch)
        else:
            break

    return "".join(chars)


def trailing_hanzi_run(text: str) -> str:
    """
    提取原始文本结尾连续出现的汉字串。
    只要结尾最后一个有效字符不是汉字，就返回空串。
    """
    text = (text or "").strip()
    chars = []

    for ch in reversed(text):
        if is_hanzi_char(ch):
            chars.append(ch)
        else:
            break

    return "".join(reversed(chars))


def edge_candidate_from_raw_text(raw_pred: str, expected_len: int, phrase: str = "") -> Tuple[Optional[str], str]:
    """
    全文兜底时使用的“首尾优先”策略。

    经验观察：
    - 很多模型会把最终答案放在最开头，然后后面接解释。
    - 也有一些模型会先解释，最后一行给最终答案。
    - 还有一些模型会先复述原 phrase，再分析，最后给真正答案。

    所以这里不再无脑取全文前 expected_len 个汉字，而是：
    1. 先看原始文本开头是否是连续汉字。
    2. 再看原始文本结尾是否是连续汉字。
    3. 如果开头候选刚好等于原 phrase，但结尾候选不是原 phrase，
       优先选择结尾候选，避免把“复述输入”当成答案。
    4. 如果没有更好的结尾候选，才保留开头候选。

    注意：这里看的是“原始文本边界”，不是清洗后的全文汉字串边界。
    这样可以避免中文解释混在中间时，被简单拼接后误截。
    """
    if not raw_pred or expected_len <= 0:
        return None, "edge_no_raw_or_bad_expected_len"

    raw_stripped = raw_pred.strip()

    head = strip_common_answer_prefix(leading_hanzi_run(raw_stripped))
    head_candidate = head[:expected_len] if len(head) >= expected_len else None

    tail = trailing_hanzi_run(raw_stripped)
    tail_candidate = tail[-expected_len:] if len(tail) >= expected_len else None

    # 如果开头只是复述原 phrase，而结尾有另一个候选，优先取结尾。
    if head_candidate == phrase and tail_candidate and tail_candidate != phrase:
        return tail_candidate, "local_edge_tail_over_repeated_phrase_head"

    if head_candidate:
        return head_candidate, "local_edge_head"

    if tail_candidate:
        return tail_candidate, "local_edge_tail"

    return None, "edge_no_candidate"


def same_raw_source(clean_item: dict, raw_item: dict) -> bool:
    """
    防止 raw 文件重新预测后，旧 clean 结果被误认为可复用。
    """
    for key in ["phrase", "change", "replace", "model_prediction"]:
        if clean_item.get(key) != raw_item.get(key):
            return False
    return True


# ==========================================================================
# 4. 本地候选答案提取
# ==========================================================================

def extract_local_candidate(raw_pred: str, expected_len: int, phrase: str = "") -> Tuple[Optional[str], str]:
    """
    本地提取策略，不调用模型，不使用标准答案。

    v7 核心调整：
    - 之前优先取“最早出现的等长非原词候选”。
    - 但有些模型会先给一个初稿答案，然后说 Wait / redo / 重新分析，最后再给修订答案。
    - 所以现在会先扫描所有行里的等长候选：
        1. 如果有多个“不等于 phrase”的等长候选，优先取最后一个。
        2. 如果只有一个“不等于 phrase”的等长候选，取它。
        3. 如果只有等于 phrase 的候选，暂存，继续看首尾/截断候选。
        4. 最后实在没有其他候选，才接受等于 phrase 的候选。

    原则：
    - 不使用 replace 标准答案。
    - 只使用 expected_len = len(phrase) 这个任务约束。
    """
    if not raw_pred or expected_len <= 0:
        return None, "no_raw_or_bad_expected_len"

    raw_stripped = raw_pred.strip()
    all_hanzi = strip_common_answer_prefix(clean_hanzi_text(raw_stripped))

    # 如果全文清洗后刚好等长，说明文本里基本只有一个候选。
    if len(all_hanzi) == expected_len:
        return all_hanzi, "local_all_exact"

    lines = [line.strip() for line in raw_pred.splitlines() if line.strip()]

    exact_non_phrase_candidates = []
    phrase_like_candidate = None
    phrase_like_note = ""

    # 先扫描所有行：收集“刚好等长”的候选。
    # 如果有多个非 phrase 候选，选最后一个，适配“初稿 + redo + 最终答案”的模式。
    for idx, line in enumerate(lines):
        candidate = strip_common_answer_prefix(clean_hanzi_text(line))

        if len(candidate) == expected_len:
            if candidate != phrase:
                exact_non_phrase_candidates.append((idx, candidate))
            elif phrase_like_candidate is None:
                phrase_like_candidate = candidate
                phrase_like_note = f"local_line_exact_repeated_phrase_{idx}"

    if exact_non_phrase_candidates:
        idx, candidate = exact_non_phrase_candidates[-1]
        if len(exact_non_phrase_candidates) == 1:
            return candidate, f"local_line_exact_{idx}"
        return candidate, f"local_line_exact_last_of_{len(exact_non_phrase_candidates)}_{idx}"

    # 首尾兜底：如果开头是复述输入，但结尾有答案，会优先取结尾。
    edge_candidate, edge_note = edge_candidate_from_raw_text(raw_pred, expected_len, phrase=phrase)
    if edge_candidate:
        if edge_candidate != phrase:
            return edge_candidate, edge_note
        if phrase_like_candidate is None:
            phrase_like_candidate = edge_candidate
            phrase_like_note = edge_note + "_repeated_phrase"

    # 再处理“单行答案多了字”的情况。
    # 这一步放在等长候选和首尾候选之后，避免被中文解释行误截。
    truncate_non_phrase_candidates = []
    truncate_phrase_candidate = None
    truncate_phrase_note = ""

    for idx, line in enumerate(lines):
        candidate = strip_common_answer_prefix(clean_hanzi_text(line))

        if len(candidate) > expected_len:
            truncated = candidate[:expected_len]

            # 如果这一行就是原始输出的第一行，或者这一行开头带“答案/最终答案”等前缀，
            # 它比普通解释行更像候选答案。
            stripped_line = line.strip()
            looks_like_answer_line = (
                idx == 0
                or stripped_line.startswith("答案")
                or stripped_line.startswith("最终答案")
                or stripped_line.startswith("结果")
                or stripped_line.startswith("输出")
                or stripped_line.startswith("预测结果")
                or stripped_line.startswith("模型输出")
                or stripped_line.startswith("变换结果")
            )

            if not looks_like_answer_line:
                continue

            if truncated != phrase:
                truncate_non_phrase_candidates.append((idx, truncated))
            elif truncate_phrase_candidate is None:
                truncate_phrase_candidate = truncated
                truncate_phrase_note = f"local_line_prefix_truncate_repeated_phrase_{idx}"

    if truncate_non_phrase_candidates:
        idx, candidate = truncate_non_phrase_candidates[-1]
        if len(truncate_non_phrase_candidates) == 1:
            return candidate, f"local_line_prefix_truncate_{idx}"
        return candidate, f"local_line_prefix_truncate_last_of_{len(truncate_non_phrase_candidates)}_{idx}"

    if truncate_phrase_candidate is not None and phrase_like_candidate is None:
        phrase_like_candidate = truncate_phrase_candidate
        phrase_like_note = truncate_phrase_note

    # 如果全篇只找到原 phrase，那就接受它。
    # 这覆盖“模型确实认为保持原词不变”的情况。
    if phrase_like_candidate is not None:
        return phrase_like_candidate, phrase_like_note

    return None, "local_no_candidate"


# ==========================================================================
# 5. 清洗状态判断
# ==========================================================================

def is_terminal_clean_error(cleaned_prediction: str) -> bool:
    """
    有些 ERROR 是终止状态，不需要反复请求：
    - raw 预测本身失败
    - 模型输出没有有效答案
    - 正则清洗后没有汉字

    清洗 API 请求失败则允许下次重跑。
    """
    if not cleaned_prediction:
        return False

    terminal_prefixes = [
        "ERROR: 连续",
        "ERROR: 数据缺失",
        "ERROR: 空预测值",
        "ERROR: 模型未输出有效答案",
        "ERROR: 模型未输出汉字结果",
    ]

    return any(cleaned_prediction.startswith(prefix) for prefix in terminal_prefixes)


def is_valid_clean_item(item: dict) -> bool:
    clean_pred = item.get("cleaned_prediction", "")

    if not clean_pred:
        return False

    # 清洗模型请求失败：下次需要重跑
    if clean_pred.startswith("ERROR: 清洗模型请求失败"):
        return False

    # 其他 ERROR 视为终止状态，保留，避免无意义重复请求
    if clean_pred.startswith("ERROR:"):
        return is_terminal_clean_error(clean_pred)

    return True


# ==========================================================================
# 6. 清洗模型 Prompt
# ==========================================================================

def build_clean_prompt(raw_pred: str, phrase: str, change: str, expected_len: int) -> str:
    return f"""请从下面的模型输出中提取最终答案。

【原始输入】
phrase：{phrase}
change：{change}

【硬性要求】
1. 最终答案必须是一个连续的汉字串。
2. 最终答案必须与 phrase 字数完全一致：{expected_len} 个汉字。
3. 如果输出中有多个候选答案，优先选择最早出现、最像完整答案的那个连续汉字串。
4. 如果模型先给出一行答案，后面又开始解释、反思、推理，优先提取最前面的那行答案。
5. 禁止输出解释、标点、空格、英文、Markdown。
6. 如果完全找不到任何可能的 {expected_len} 字汉字答案，请输出 NULL。

【待清洗的模型输出】
{raw_pred}

请只输出最终答案本体。"""


# ==========================================================================
# 7. 单条文本清洗逻辑
# ==========================================================================

def step2_text_clean(item: dict) -> Tuple[dict, str]:
    raw_pred = item.get("model_prediction", "")
    phrase = item.get("phrase", "")
    change = item.get("change", "")
    expected_len = len(phrase)

    item["cleaning_note"] = ""

    if not raw_pred or raw_pred.startswith("ERROR:"):
        item["cleaned_prediction"] = raw_pred if raw_pred else "ERROR: 空预测值"
        item["cleaning_note"] = "raw_error_or_empty"
        return item, item["cleaned_prediction"]

    # 1. 如果原输出本身字数完全匹配，保留原来的“直通车”思想
    raw_hanzi = clean_hanzi_text(raw_pred)
    if len(raw_pred) == expected_len and len(raw_hanzi) == expected_len:
        item["cleaned_prediction"] = raw_hanzi
        item["cleaning_note"] = "direct_len_match"
        return item, raw_hanzi

    # 2. 先用本地规则提取，避免清洗模型抽错片段
    local_candidate, local_note = extract_local_candidate(raw_pred, expected_len, phrase=phrase)
    if local_candidate and len(local_candidate) == expected_len:
        item["cleaned_prediction"] = local_candidate
        item["cleaning_note"] = local_note
        return item, local_candidate

    # 3. 本地规则没找到，再调用清洗模型
    clean_pred = ""
    active_client = primary_client

    for attempt in range(MAX_RETRIES):
        try:
            response = active_client.chat.completions.create(
                model=CHECK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个答案提取助手。你的任务是从混杂文本中提取最终汉字答案。"
                            "必须严格遵守用户给出的目标字数。"
                            "只能输出答案本体或 NULL。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": build_clean_prompt(raw_pred, phrase, change, expected_len),
                    },
                ],
                temperature=0.0,
                max_tokens=1000, 
                timeout=REQUEST_TIMEOUT,
            )

            clean_pred = (response.choices[0].message.content or "").strip()
            break

        except Exception as e:
            if should_switch_to_backup(e, attempt):
                active_client = backup_client

            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
            else:
                clean_pred = f"ERROR: 清洗模型请求失败: {str(e)}"

    if clean_pred == "NULL":
        item["cleaned_prediction"] = "ERROR: 模型未输出有效答案"
        item["cleaning_note"] = "clean_model_null"
        return item, item["cleaned_prediction"]

    # 4. 清洗模型结果再做本地字数约束
    if not clean_pred.startswith("ERROR:"):
        clean_pred = strip_common_answer_prefix(clean_hanzi_text(clean_pred))

        if not clean_pred:
            item["cleaned_prediction"] = "ERROR: 模型未输出汉字结果"
            item["cleaning_note"] = "clean_model_no_hanzi"
            return item, item["cleaned_prediction"]

        if len(clean_pred) == expected_len:
            item["cleaned_prediction"] = clean_pred
            item["cleaning_note"] = "clean_model_exact"
            return item, clean_pred

        if len(clean_pred) > expected_len and TRUNCATE_OVERLONG_TO_EXPECTED_LEN:
            item["cleaned_prediction"] = clean_pred[:expected_len]
            item["cleaning_note"] = f"clean_model_overlong_truncated_{len(clean_pred)}_to_{expected_len}"
            return item, item["cleaned_prediction"]

        # 如果清洗模型输出过短，最后再尝试全文前缀
        fallback_candidate, fallback_note = extract_local_candidate(raw_pred, expected_len, phrase=phrase)
        if fallback_candidate:
            item["cleaned_prediction"] = fallback_candidate
            item["cleaning_note"] = f"fallback_after_short_model_{fallback_note}"
            return item, fallback_candidate

    item["cleaned_prediction"] = clean_pred if clean_pred else "ERROR: 模型未输出有效答案"
    item["cleaning_note"] = item.get("cleaning_note") or "final_fallback"
    return item, item["cleaned_prediction"]


# ==========================================================================
# 8. 断点续跑逻辑
# ==========================================================================

def load_and_clean_clean_history(clean_file: str, raw_by_id: Dict[Any, dict]) -> Tuple[List[dict], set, int]:
    history_items = read_jsonl(clean_file)
    history_items = dedupe_items_by_id_prefer_valid(history_items, "cleaned_prediction")

    valid_items = []
    processed_ids = set()
    bad_count = 0

    for item in history_items:
        item_id = item.get("id")

        if item_id is None:
            bad_count += 1
            continue

        raw_item = raw_by_id.get(item_id)
        if raw_item is None:
            bad_count += 1
            continue

        if not same_raw_source(item, raw_item):
            # raw 预测发生变化，旧 clean 不能复用
            bad_count += 1
            continue

        if is_valid_clean_item(item):
            if item_id not in processed_ids:
                valid_items.append(item)
                processed_ids.add(item_id)
        else:
            bad_count += 1

    valid_items = sorted(valid_items, key=sort_key_by_id)

    if CLEAN_BAD_HISTORY and os.path.exists(clean_file):
        rewrite_jsonl(clean_file, valid_items)

    return valid_items, processed_ids, bad_count


# ==========================================================================
# 9. 单模型 / 全模型执行入口
# ==========================================================================

def run_text_cleaning_for_model(model_name: str) -> None:
    print(f"\n🧼 [Phase 2 文本清洗] 当前模型：{model_name}")

    raw_file = os.path.join(OUTPUT_DIR, f"raw_{model_name}.jsonl")
    clean_file = os.path.join(OUTPUT_DIR, f"clean_{model_name}.jsonl")

    if not os.path.exists(raw_file):
        print(f"⚠️ 找不到 raw 文件：{raw_file}，请先运行 01_predict.py。")
        return

    # 清洗开始前，先对 raw 结果去重并按 id 排序，然后写回 raw 文件
    raw_items = read_jsonl(raw_file)
    raw_items = dedupe_items_by_id_prefer_valid(raw_items, "model_prediction")
    raw_items = sorted(raw_items, key=sort_key_by_id)
    rewrite_jsonl(raw_file, raw_items)

    raw_by_id = {item.get("id"): item for item in raw_items}

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    clean_items, processed_ids, bad_count = load_and_clean_clean_history(clean_file, raw_by_id)

    print(f"📦 raw 记录：{len(raw_items)} 条，已按 id 排序")
    print(f"🧹 历史 clean 有效/终止记录：{len(clean_items)} 条；清理可重跑失败/坏记录/过期记录：{bad_count} 条")

    pending_items = [item for item in raw_items if item.get("id") not in processed_ids]

    if not pending_items:
        print(f"✅ {model_name} 已无待清洗数据。")
        return

    print(f"⏳ 待清洗/重试：{len(pending_items)} / {len(raw_items)} 条")

    with open(clean_file, "a", encoding="utf-8") as fout:
        with tqdm(total=len(pending_items), desc=f"清洗 {model_name}", unit="条") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_item = {
                    executor.submit(step2_text_clean, item.copy()): item
                    for item in pending_items
                }

                for future in concurrent.futures.as_completed(future_to_item):
                    try:
                        result_item, _ = future.result()
                        append_jsonl_line(fout, result_item)
                    except Exception as e:
                        pbar.set_postfix({"清洗异常": str(e)[:20]})
                    finally:
                        pbar.update(1)

    # 清洗完成后也统一按 id 排序 clean 文件
    final_clean_items = read_jsonl(clean_file)
    final_clean_items = dedupe_items_by_id_prefer_valid(final_clean_items, "cleaned_prediction")
    rewrite_jsonl(clean_file, final_clean_items)

    print(f"🎉 {model_name} 文本清洗阶段完成。clean 已按 id 排序：{clean_file}")


def run_all_text_cleaning() -> None:
    for model_name in MODEL_LIST:
        run_text_cleaning_for_model(model_name)

    print("\n🎉 Phase 2 全部模型文本清洗完成。")


if __name__ == "__main__":
    run_all_text_cleaning()

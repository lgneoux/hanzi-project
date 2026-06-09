# ==========================================================================
# Phase 1: 汉字结构测评系统 —— 预测阶段
# ==========================================================================
# 输入：total_result.json
# 输出：eval_results/raw_{model}.jsonl

# ==========================================================================

import json
import os
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

INPUT_FILE = "total_result.json"
OUTPUT_DIR = "eval_results"

MAX_WORKERS = 10
MAX_TRIES = 3

REQUEST_TIMEOUT = 60

# 是否物理清理已有 raw 文件里的无效/过期记录
CLEAN_BAD_HISTORY = True

# 最后一轮仍失败的样本是否写入 raw 文件为 ERROR
WRITE_FINAL_ERRORS_TO_RAW = True


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

def load_json(filepath: str) -> Any:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到文件：{filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            return json.load(f)


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


def rewrite_jsonl(filepath: str, items: List[dict]) -> None:
    items = sorted(items, key=sort_key_by_id)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl_line(file_handle, item: dict) -> None:
    file_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    file_handle.flush()


def should_switch_to_backup(err: Exception) -> bool:
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

    return any(k in err_str for k in fatal_keywords)


def same_source_item(history_item: dict, current_item: dict) -> bool:
    """
    防止 total_result.json 被修改后，旧 raw 结果被误认为可复用。
    只检查上游输入字段，不检查 model_prediction。
    """
    for key in ["phrase", "change", "replace"]:
        if history_item.get(key) != current_item.get(key):
            return False
    return True


def is_valid_raw_item(item: dict) -> bool:
    pred = item.get("model_prediction", "")
    return bool(pred) and not pred.startswith("ERROR:")


# ==========================================================================
# 4. Prompt
# ==========================================================================

SYSTEM_PROMPT = "你是一个精通汉字结构与部首变换的语言学专家。"


def build_user_prompt(phrase: str, change: str) -> str:
    return f"""任务描述：
给定一个词语（phrase）和一个目标部首（change），请对词语中的**每一个汉字**逐一应用以下规则进行变换，最后拼接成一个新的词语输出。

【变换规则】（按顺序匹配，命中即停止）：
1. 【删】如果该字的当前部首刚好是目标部首：删去该部首。
   - 注意：删去后必须是一个完整的字，否则保持原字不变。
2. 【加】如果该字加上目标部首能组成一个新字：加上该部首。
   - 注意：如果既能加部首成字，又能换部首成字，优先选用【加】。
3. 【换】如果该字的当前部首替换为目标部首能组成一个新字：替换该部首。
4. 【留】如果以上都不满足：保持原字不变。

请注意：请直接输出处理后的完整词语，不要包含任何解释、分析、标点符号或其他多余文字。
请注意：请直接输出处理后的完整词语，不要包含任何解释、分析、标点符号或其他多余文字。
请注意：请直接输出处理后的完整词语，不要包含任何解释、分析、标点符号或其他多余文字。
注意，你输出 字应该跟输入的字一一对应。禁止输出任何推理过程、英文、拼音、Markdown符号或解释性文字。只允许输出最终变换后的词语本体！

【输入数据】
词语：{phrase}
目标部首：{change}

请直接输出处理后的完整词语。"""


# ==========================================================================
# 5. 单条预测逻辑：每次只请求一次，不做内部 sleep 重试
# ==========================================================================

def step1_predict_once(item: dict, model_name: str) -> Tuple[dict, bool, str]:
    phrase = item.get("phrase", "")
    change = item.get("change", "")

    if not phrase or not change:
        error_text = "ERROR: 数据缺失"
        item["model_prediction"] = error_text
        return item, False, error_text

    user_content = build_user_prompt(phrase, change)

    active_client = primary_client

    try:
        response = active_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=2000,
            timeout=REQUEST_TIMEOUT,
        )

        prediction = (response.choices[0].message.content or "").strip()

        if not prediction:
            raise ValueError("模型返回了空字符串")

        item["model_prediction"] = prediction
        return item, True, prediction

    except Exception as e:
        # 只有致命错误才在本次单请求里切到备用 key 再试一次。
        # 这里不 sleep，也不做多轮 retry；普通失败留给下一轮大流程处理。
        if should_switch_to_backup(e):
            try:
                response = backup_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                    timeout=REQUEST_TIMEOUT,
                )
                prediction = (response.choices[0].message.content or "").strip()
                if not prediction:
                    raise ValueError("备用 Key 模型返回了空字符串")

                item["model_prediction"] = prediction
                return item, True, prediction

            except Exception as backup_e:
                error_text = f"ERROR: 本轮预测失败。主 Key 报错: {str(e)}；备用 Key 报错: {str(backup_e)}"
                item["model_prediction"] = error_text
                return item, False, error_text

        error_text = f"ERROR: 本轮预测失败。报错: {str(e)}"
        item["model_prediction"] = error_text
        return item, False, error_text


# ==========================================================================
# 6. 断点续跑逻辑
# ==========================================================================

def load_and_clean_raw_history(raw_file: str, current_data_by_id: Dict[Any, dict]) -> Tuple[List[dict], set, int]:
    history_items = read_jsonl(raw_file)
    valid_by_id = {}
    bad_count = 0

    for item in history_items:
        item_id = item.get("id")

        if item_id is None:
            bad_count += 1
            continue

        current_item = current_data_by_id.get(item_id)
        if current_item is None:
            bad_count += 1
            continue

        if not same_source_item(item, current_item):
            bad_count += 1
            continue

        if is_valid_raw_item(item):
            # 如果重复出现同一个 id，保留最后一个有效结果
            valid_by_id[item_id] = item
        else:
            bad_count += 1

    valid_items = sorted(valid_by_id.values(), key=sort_key_by_id)
    processed_ids = set(valid_by_id.keys())

    if CLEAN_BAD_HISTORY and os.path.exists(raw_file):
        rewrite_jsonl(raw_file, valid_items)

    return valid_items, processed_ids, bad_count


def write_failed_items(model_name: str, failed_items: List[dict], round_idx: int) -> None:
    """
    记录本轮失败项，方便之后单独检查疑难样本。
    每轮覆盖写入，不影响 raw 主结果文件。
    """
    fail_file = os.path.join(OUTPUT_DIR, f"failed_round{round_idx}_{model_name}.jsonl")
    rewrite_jsonl(fail_file, failed_items)


# ==========================================================================
# 7. 单模型 / 全模型执行入口
# ==========================================================================

def run_prediction_for_model(model_name: str) -> None:
    print(f"\n🚀 [Phase 1 预测] 当前模型：{model_name}")

    all_data = load_json(INPUT_FILE)
    current_data_by_id = {item.get("id"): item for item in all_data}

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    raw_file = os.path.join(OUTPUT_DIR, f"raw_{model_name}.jsonl")

    valid_raw_items, processed_ids, bad_count = load_and_clean_raw_history(raw_file, current_data_by_id)

    print(f"🧹 历史 raw 有效记录：{len(valid_raw_items)} 条；清理失败/坏记录/过期记录：{bad_count} 条")

    pending_data = [item for item in all_data if item.get("id") not in processed_ids]

    if not pending_data:
        print(f"✅ {model_name} 已无待预测数据。")
        return

    print(f"⏳ 初始待预测：{len(pending_data)} / {len(all_data)} 条")

    for round_idx in range(1, MAX_TRIES + 1):
        if not pending_data:
            break

        print(f"\n🔁 开始第 {round_idx}/{MAX_TRIES} 轮预测：本轮待处理 {len(pending_data)} 条")

        success_count = 0
        failed_items = []

        with open(raw_file, "a", encoding="utf-8") as fout:
            with tqdm(total=len(pending_data), desc=f"预测 {model_name} Round {round_idx}", unit="条") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    future_to_item = {
                        executor.submit(step1_predict_once, item.copy(), model_name): item
                        for item in pending_data
                    }

                    for future in concurrent.futures.as_completed(future_to_item):
                        try:
                            result_item, ok, _ = future.result()

                            if ok:
                                append_jsonl_line(fout, result_item)
                                success_count += 1
                            else:
                                failed_items.append(result_item)

                        except Exception as e:
                            original_item = future_to_item[future].copy()
                            original_item["model_prediction"] = f"ERROR: Future 突发异常: {str(e)}"
                            failed_items.append(original_item)
                            pbar.set_postfix({"突发异常": str(e)[:20]})
                        finally:
                            pbar.update(1)

        write_failed_items(model_name, failed_items, round_idx)

        print(f"✅ 第 {round_idx} 轮完成：成功 {success_count} 条，失败 {len(failed_items)} 条")

        # 下一轮只处理本轮失败项
        pending_data = [
            {k: v for k, v in item.items() if k != "model_prediction"}
            for item in failed_items
        ]

    # 最后一轮仍失败的样本，写入 raw 为 ERROR，供后续 clean/final 统计为无效
    if pending_data and WRITE_FINAL_ERRORS_TO_RAW:
        print(f"⚠️ 仍有 {len(pending_data)} 条样本在 {MAX_TRIES} 轮后失败，写入 raw 为 ERROR。")
        with open(raw_file, "a", encoding="utf-8") as fout:
            for item in pending_data:
                if not item.get("model_prediction", "").startswith("ERROR:"):
                    item["model_prediction"] = f"ERROR: 连续 {MAX_TRIES} 轮预测失败"
                append_jsonl_line(fout, item)

    # 最后统一按 id 排序 raw 文件
    final_items = read_jsonl(raw_file)
    rewrite_jsonl(raw_file, final_items)

    print(f"🎉 {model_name} 预测阶段完成。raw 已按 id 排序：{raw_file}")


def run_all_predictions() -> None:
    for model_name in MODEL_LIST:
        run_prediction_for_model(model_name)

    print("\n🎉 Phase 1 全部模型预测完成。")


if __name__ == "__main__":
    run_all_predictions()

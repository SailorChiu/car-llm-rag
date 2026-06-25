"""
agent/eval_tools.py — 工具调用准确率评测

评测指标：
  1. JSON 一次解析成功率 = 模型首次输出能被正则+json 成功解析的比例（走兜底=失败）
  2. 工具选对率         = 解析出的工具名 == 标注期望工具 的比例

设计原则：
  - 直接 import agent_demo._extract_json / _generate，与 demo 运行时完全同一套逻辑，
    指标才有意义（不另起一套解析）。
  - 模型只加载一次，循环跑所有 query。
  - Step 1（意图识别）只看工具选择，不执行工具副作用。
  - 报错 catch 并计入失败，真实打印，不编造数字。

用法：
    cd ~/car-llm && source .venv/bin/activate
    python agent/eval_tools.py
"""

import os
import sys
import csv
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────
# 路径设置
# ─────────────────────────────────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (PROJECT_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402

# 复用 agent_demo 的解析函数：保证评测逻辑与 demo 完全一致
from agent_demo import _extract_json, _generate   # noqa: E402
from prompts    import AGENT_SYSTEM               # noqa: E402

# ─────────────────────────────────────────────────────────────
# 标注数据集：30 条，均衡覆盖 4 类工具
# 中文贴近真实车主说法；expected_tool 为黄金标注
# ─────────────────────────────────────────────────────────────
EVAL_DATA: list[tuple[str, str]] = [
    # ── query_manual（车载手册 / RAG 类）× 10 ──────────────
    ("前挡风起雾怎么办",               "query_manual"),
    ("胎压报警灯亮了怎么处理",         "query_manual"),
    ("怎么开启车道保持辅助",           "query_manual"),
    ("安全气囊什么情况下会弹出",       "query_manual"),
    ("如何使用自动泊车功能",           "query_manual"),
    ("发动机机油怎么检查",             "query_manual"),
    ("车辆保养周期是多久",             "query_manual"),
    ("倒车雷达怎么关闭",               "query_manual"),
    ("怎么调节方向盘高度",             "query_manual"),
    ("后视镜加热功能怎么开启",         "query_manual"),

    # ── set_climate（空调温度）× 7 ─────────────────────────
    ("把温度调到22度",                 "set_climate"),
    ("空调开到26度",                   "set_climate"),
    ("车里有点冷，帮我把温度调高",     "set_climate"),
    ("把空调温度设成24度",             "set_climate"),
    ("温度调到20摄氏度",               "set_climate"),
    ("我想把温度设到25度",             "set_climate"),
    ("调高温度，设到28度吧",           "set_climate"),

    # ── control_window（车窗）× 7 ──────────────────────────
    ("打开主驾车窗",                   "control_window"),
    ("把车窗关上",                     "control_window"),
    ("开一下天窗",                     "control_window"),
    ("降下副驾驶侧的窗户",             "control_window"),
    ("所有车窗都关闭",                 "control_window"),
    ("把后排左侧车窗打开",             "control_window"),
    ("车窗升起来",                     "control_window"),

    # ── chitchat（闲聊 / 其他）× 6 ────────────────────────
    ("你好",                           "chitchat"),
    ("今天天气不错",                   "chitchat"),
    ("给我讲个笑话",                   "chitchat"),
    ("你叫什么名字",                   "chitchat"),
    ("谢谢你的帮助",                   "chitchat"),
    ("你都能做什么",                   "chitchat"),
]

# 各类别期望数量（用于分类准确率展示）
_TOOL_COUNTS = {
    "query_manual":   10,
    "set_climate":     7,
    "control_window":  7,
    "chitchat":        6,
}

assert len(EVAL_DATA) == 30, "数据集应有 30 条"
assert sum(_TOOL_COUNTS.values()) == 30


# ─────────────────────────────────────────────────────────────
# 评测主逻辑
# ─────────────────────────────────────────────────────────────
def run_eval(model, tokenizer) -> list[dict]:
    """逐条推理，只跑 Step 1（意图识别），不执行工具副作用。"""
    results = []
    n = len(EVAL_DATA)
    print(f"\n{'─'*62}")
    print(f"  开始评测，共 {n} 条 query …")
    print(f"  ✓=选对  ~=解析成功但选错  ✗=解析失败")
    print(f"{'─'*62}")

    for i, (query, expected) in enumerate(EVAL_DATA, 1):
        # 只用 AGENT_SYSTEM + 单轮用户消息（无历史、无记忆 hint）
        messages = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user",   "content": query},
        ]
        raw = ""
        try:
            raw       = _generate(messages, model, tokenizer, max_new_tokens=64)
            tool_call = _extract_json(raw)
            parse_ok  = tool_call is not None and "tool" in tool_call
            predicted = tool_call.get("tool", "PARSE_FAIL") if parse_ok else "PARSE_FAIL"
            correct   = predicted == expected
        except Exception as exc:
            raw       = f"ERROR: {exc}"
            parse_ok  = False
            predicted = "ERROR"
            correct   = False

        results.append({
            "idx":        i,
            "query":      query,
            "expected":   expected,
            "predicted":  predicted,
            "parse_ok":   parse_ok,
            "correct":    correct,
            "raw_output": raw[:150],
        })

        if correct:
            sym = "✓"
        elif parse_ok:
            sym = "~"
        else:
            sym = "✗"

        print(
            f"  [{i:02d}] {sym}  "
            f"expect={expected:<16s} "
            f"pred={predicted:<16s} "
            f"「{query[:18]}」"
        )

    return results


def print_summary(results: list[dict]) -> None:
    total      = len(results)
    parse_ok_n = sum(1 for r in results if r["parse_ok"])
    correct_n  = sum(1 for r in results if r["correct"])

    print(f"\n{'='*62}")
    print("  工具调用评测汇总")
    print(f"{'='*62}")
    print(f"  总样本数:            {total}")
    print(f"  JSON 一次解析成功率: {parse_ok_n}/{total} = {parse_ok_n/total*100:.1f}%")
    print(f"  工具选对率:          {correct_n}/{total} = {correct_n/total*100:.1f}%")
    print()
    print("  分工具准确率:")
    for tool, count in _TOOL_COUNTS.items():
        tool_rs = [r for r in results if r["expected"] == tool]
        ok      = sum(1 for r in tool_rs if r["correct"])
        bar_len = int(ok / count * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {tool:<20s}  {ok}/{count}  [{bar}]")
    print(f"{'='*62}")

    # 打印错误详情（便于 bad-case 归因）
    bad = [r for r in results if not r["correct"]]
    if bad:
        print(f"\n  ── 未命中详情（{len(bad)} 条）──")
        for r in bad:
            print(
                f"  [{r['idx']:02d}] 「{r['query'][:20]}」 "
                f"期望={r['expected']}  实际={r['predicted']}  "
                f"parse={'OK' if r['parse_ok'] else 'FAIL'}"
            )


def save_csv(results: list[dict], path: str) -> None:
    fieldnames = ["idx", "query", "expected", "predicted", "parse_ok", "correct", "raw_output"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig 让 Excel 正常显示中文
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  逐条结果已写入: {path}")


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────
def main() -> None:
    print(f"\n加载 LLM: {config.LLM_MODEL} …（首次约需 30-60s）")
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    print(f"✓ 模型加载完成（device={next(model.parameters()).device}）")

    results = run_eval(model, tokenizer)
    print_summary(results)

    csv_path = os.path.join(_THIS_DIR, "eval_result.csv")
    save_csv(results, csv_path)


if __name__ == "__main__":
    main()

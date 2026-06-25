"""
agent/tools.py — 工具函数定义

包含三类工具：
  1. query_manual()   — 调用现有 RAG 检索器 + 共享 LLM 实例回答车载手册问题
  2. set_climate()    — ⚠️ MOCK / 演示用，非真实车控
  3. control_window() — ⚠️ MOCK / 演示用，非真实车控

注意：set_climate / control_window 均为演示用 mock 实现，
打印日志并返回假状态，不接入真实车辆控制系统。
真实场景需对接 CAN 总线 / OEM 车控 API（如 AUTOSAR 标准接口）。

记忆机制：
  MEMORY dict 记录用户上一次设置的温度 / 窗户操作，
  由 agent_demo.py 在生成最终回复时注入 system prompt，支持多轮引用。
"""

import os
import sys
import json

# ─────────────────────────────────────────────────────────────
# 路径设置：让 tools.py 能找到项目根目录的 config 和 rag/retrieve
# ─────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
RAG_DIR = os.path.join(PROJECT_ROOT, "rag")
for _p in (PROJECT_ROOT, RAG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
from retrieve import Retriever  # noqa: E402

# ─────────────────────────────────────────────────────────────
# 模块级记忆：记录用户上一次的设置偏好，支持多轮引用
# ─────────────────────────────────────────────────────────────
MEMORY: dict = {
    "last_temperature": None,    # 上次设置的温度（摄氏度）
    "last_window_action": None,  # 上次的窗户操作
}

# ─────────────────────────────────────────────────────────────
# RAG 检索器（延迟初始化，避免 import 时加载）
# ─────────────────────────────────────────────────────────────
_retriever: "Retriever | None" = None

def _get_retriever() -> "Retriever":
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever

_RAG_SYSTEM = (
    "你是车载用户手册助手。只根据下面提供的【手册资料】回答用户问题，"
    "用简洁中文回答；如果资料里没有相关信息，就直说手册中未提及，不要编造。"
)

def _build_rag_messages(question: str, hits: list) -> list:
    ctx = "\n\n".join(
        f"【资料{i+1}·{c['title']}】\n{c['text']}"
        for i, (c, _) in enumerate(hits)
    )
    user_content = f"【手册资料】\n{ctx}\n\n【问题】{question}"
    return [
        {"role": "system", "content": _RAG_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


# ─────────────────────────────────────────────────────────────
# 工具 1：query_manual — 查询车载手册（RAG）
# 复用现有 bge+FAISS 检索器；使用共享 LLM 实例生成回答（省显存）
# ─────────────────────────────────────────────────────────────
def query_manual(question: str, model, tokenizer) -> dict:
    """
    查询车载用户手册（RAG 检索 + 共享 LLM 生成）。

    参数:
        question: 用户问题
        model, tokenizer: 由 agent_demo.py 传入的共享 LLM 实例
    返回:
        {"status": "ok", "answer": str, "retrieved_chunks": list[str]}
    """
    import torch

    retriever = _get_retriever()
    hits = retriever.search(question)
    messages = _build_rag_messages(question, hits)

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
    chunk_titles = [c["title"] for c, _ in hits]

    return {
        "status": "ok",
        "answer": answer,
        "retrieved_chunks": chunk_titles,
    }


# ─────────────────────────────────────────────────────────────
# 工具 2：set_climate — 设置车内温度
# ⚠️ MOCK / 演示用，非真实车控
# 真实场景需对接 CAN 总线 / OEM 车控 API
# ─────────────────────────────────────────────────────────────
def set_climate(temperature, **_kwargs) -> dict:
    """
    设置车内空调温度。

    ⚠️  MOCK / 演示用，非真实车控
        打印日志并返回假状态，不向任何车控系统发送指令。
        真实接入需对接 OEM CAN 总线 API（如 AUTOSAR Adaptive Platform）。

    参数:
        temperature: 目标温度（摄氏度，int 或 float）
    返回:
        {"status": "mock_ok", "message": str, "temperature": float}
    """
    temp = float(temperature)
    MEMORY["last_temperature"] = temp          # 写入记忆，下轮可引用

    _YELLOW = "\033[33m"
    _RESET  = "\033[0m"
    print(f"{_YELLOW}[MOCK-车控] set_climate → {temp}°C  "
          f"（演示用，非真实车控，不接入任何 CAN/OEM 接口）{_RESET}")

    return {
        "status":      "mock_ok",
        "message":     f"已将车内温度设置为 {temp}°C（演示用，非真实车控）",
        "temperature": temp,
    }


# ─────────────────────────────────────────────────────────────
# 工具 3：control_window — 控制车窗
# ⚠️ MOCK / 演示用，非真实车控
# ─────────────────────────────────────────────────────────────
_WINDOW_ACTION_MAP = {
    "open":  "打开",
    "close": "关闭",
    "up":    "升起",
    "down":  "降下",
}

def control_window(action: str = "open", **_kwargs) -> dict:
    """
    控制车窗开关/升降。

    ⚠️  MOCK / 演示用，非真实车控
        打印日志并返回假状态，不向任何车控系统发送指令。

    参数:
        action: 操作指令，可选 open / close / up / down
    返回:
        {"status": "mock_ok", "message": str, "action": str}
    """
    action = action.lower().strip()
    MEMORY["last_window_action"] = action       # 写入记忆

    zh_action = _WINDOW_ACTION_MAP.get(action, action)
    _YELLOW = "\033[33m"
    _RESET  = "\033[0m"
    print(f"{_YELLOW}[MOCK-车控] control_window → {action}  "
          f"（演示用，非真实车控，不接入任何 CAN/OEM 接口）{_RESET}")

    return {
        "status":  "mock_ok",
        "message": f"车窗已执行操作：{zh_action}（演示用，非真实车控）",
        "action":  action,
    }

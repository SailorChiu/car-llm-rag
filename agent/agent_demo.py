"""
agent/agent_demo.py — 工具调用 Agent 主循环

演示意图识别 → 工具调用（RAG / 车控 Mock）→ 结果回填 → 多轮记忆 的完整闭环。

用法：
    cd ~/car-llm && source .venv/bin/activate
    python agent/agent_demo.py

设计说明：
  - LLM 实例只初始化一次，query_manual 也复用同一份权重（8G 显存约束下的工程取舍）。
  - 第一步调用（工具选择）：greedy decoding + 正则提取 JSON，parse 失败 fallback chitchat。
    针对 1.5B 小模型 JSON 输出不稳定的兜底设计——小模型偶尔夹带多余文字，
    用 re.search 抠出第一个 {...} 再 parse，避免因格式问题直接崩溃。
  - 第二步调用（自然语言回复）：将工具执行结果注入 messages，生成用户可见回复。
  - 多轮记忆：messages 列表维护完整对话历史；MEMORY dict 显式记录温度/窗户偏好，
    并在每次生成时注入 system hint，确保小模型也能跨轮引用用户设置。
"""

import os
import sys
import re
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

import config                                        # noqa: E402
from prompts import AGENT_SYSTEM, FINAL_RESPONSE_SYSTEM, OBSERVATION_TEMPLATE  # noqa: E402
from tools   import query_manual, set_climate, control_window, MEMORY           # noqa: E402

# ─────────────────────────────────────────────────────────────
# 终端颜色（可选，降级时不影响功能）
# ─────────────────────────────────────────────────────────────
_C = {
    "cyan":   "\033[36m",
    "green":  "\033[32m",
    "yellow": "\033[33m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{_C.get(color, '')}{text}{_C['reset']}"


# ─────────────────────────────────────────────────────────────
# LLM 推理工具函数
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def _generate(messages: list, model, tokenizer, max_new_tokens: int = 128) -> str:
    """用 apply_chat_template 格式化 messages，greedy 解码，返回新生成文本。"""
    text   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out    = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _extract_json(raw: str) -> dict | None:
    """
    从 LLM 原始输出中提取第一个合法 JSON 对象。

    针对 1.5B 小模型 JSON 输出不稳定的兜底设计：
      - 小模型偶尔在 JSON 前后夹带解释文字
      - 用 re.search 贪婪匹配第一个 {...}，再 json.loads
      - 任何解析异常都返回 None，由调用方 fallback 到 chitchat

    这是简历里"小模型 Agent 工程经验"的关键细节：
    直接 json.loads(整段输出) 会因格式噪声频繁崩溃；正则兜底显著提升稳定性。
    """
    # 先尝试整段 parse（模型完全按格式输出时最快）
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # fallback：贪婪匹配第一个 { ... }（跨行）
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # 如果嵌套大括号导致贪婪不够，用非贪婪 + 最外层匹配
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return None  # 全部失败，调用方 fallback chitchat


def _build_memory_hint() -> str:
    """把当前 MEMORY 格式化成 system hint，让小模型跨轮可靠引用用户设置。"""
    parts = []
    if MEMORY.get("last_temperature") is not None:
        parts.append(f"用户上次设置的温度：{MEMORY['last_temperature']}°C")
    if MEMORY.get("last_window_action"):
        parts.append(f"用户上次的车窗操作：{MEMORY['last_window_action']}")
    return ("【记忆】" + "；".join(parts)) if parts else ""


# ─────────────────────────────────────────────────────────────
# Agent 单步：意图识别 → 工具调用 → 回填 LLM → 返回最终回复
# ─────────────────────────────────────────────────────────────
def agent_step(
    user_msg: str,
    history: list,
    model,
    tokenizer,
) -> tuple[str, str, dict]:
    """
    ReAct 单跳 Agent 步骤。

    返回：(final_reply, tool_name, tool_result)
    """
    # ── Step 1：意图识别 ──────────────────────────────────────
    # 把记忆提示加进 system，让小模型跨轮也能引用
    memory_hint = _build_memory_hint()
    system_content = AGENT_SYSTEM
    if memory_hint:
        system_content = system_content + "\n\n" + memory_hint

    step1_messages = (
        [{"role": "system", "content": system_content}]
        + history
        + [{"role": "user", "content": user_msg}]
    )

    raw_json = _generate(step1_messages, model, tokenizer, max_new_tokens=64)
    tool_call = _extract_json(raw_json)

    # parse 失败 → fallback chitchat（兜底，不崩溃）
    if tool_call is None or "tool" not in tool_call:
        print(_c("yellow", f"  [解析] JSON 提取失败，原始输出：{raw_json!r}"))
        print(_c("yellow",  "  [解析] fallback → chitchat"))
        tool_call = {"tool": "chitchat", "args": {}}

    tool_name = tool_call.get("tool", "chitchat")
    args      = tool_call.get("args", {})

    print(_c("cyan", f"\n  ▶ 意图识别 → [{tool_name}]  args={json.dumps(args, ensure_ascii=False)}"))

    # ── Step 2：执行工具 ──────────────────────────────────────
    tool_result: dict = {}

    if tool_name == "query_manual":
        question = args.get("question", user_msg)
        tool_result = query_manual(question, model, tokenizer)
        # RAG 已由内部 LLM 生成自然语言答案，直接作为最终回复
        final_reply = tool_result["answer"]
        chunks_hit  = tool_result.get("retrieved_chunks", [])
        print(_c("cyan", f"  ▶ RAG 命中知识块：{chunks_hit}"))
        return final_reply, tool_name, tool_result

    elif tool_name == "set_climate":
        tool_result = set_climate(**args)

    elif tool_name == "control_window":
        tool_result = control_window(**args)

    else:
        # chitchat：跳过工具执行，直接进 Step 3 生成回复
        tool_result = {}

    # ── Step 3：把工具结果喂回 LLM，生成最终自然语言回复 ─────
    obs_text = (
        OBSERVATION_TEMPLATE.format(result=json.dumps(tool_result, ensure_ascii=False))
        if tool_result
        else user_msg     # chitchat 直接用原始用户消息
    )

    # 最终回复的 system 里同样注入记忆
    final_system = FINAL_RESPONSE_SYSTEM
    if memory_hint:
        final_system = final_system + "\n\n" + memory_hint

    step3_messages = (
        [{"role": "system", "content": final_system}]
        + history
        + [{"role": "user", "content": obs_text}]
    )
    final_reply = _generate(step3_messages, model, tokenizer, max_new_tokens=200)

    return final_reply, tool_name, tool_result


# ─────────────────────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────────────────────
def main():
    print(_c("bold", "\n" + "=" * 62))
    print(_c("bold", "  车载智能助手 Agent Demo"))
    print(_c("bold", "  (工具调用 · 多轮对话 · 记忆机制)"))
    print(_c("bold", "=" * 62))
    print("  输入 'q' 或 Ctrl-C 退出\n")

    # ── 加载模型（只加载一次，query_manual 也复用） ──────────
    print(f"加载 LLM：{config.LLM_MODEL} …（首次加载约需 30-60s）")
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    device_info = next(model.parameters()).device
    print(_c("green", f"✓ 模型已加载（device={device_info}）\n"))

    # ── 对话历史（多轮） ──────────────────────────────────────
    history: list = []

    while True:
        # 用户输入
        try:
            user_input = input(_c("bold", "你：")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input or user_input.lower() in ("q", "quit", "exit"):
            print("再见！")
            break

        # Agent 单步
        try:
            final_reply, tool_name, _ = agent_step(
                user_msg=user_input,
                history=history,
                model=model,
                tokenizer=tokenizer,
            )
        except Exception as e:
            print(_c("yellow", f"  [错误] {e}"))
            final_reply = "抱歉，处理时遇到问题，请再试一次。"
            tool_name   = "error"

        # 打印回复
        print(_c("green", f"\n助手：{final_reply}\n"))
        print("─" * 62)

        # 更新历史（追加用户消息 + 助手最终回复）
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": final_reply})

        # 控制历史长度：保留最近 10 轮（20 条），防止超 context window
        if len(history) > 20:
            history = history[-20:]


if __name__ == "__main__":
    main()

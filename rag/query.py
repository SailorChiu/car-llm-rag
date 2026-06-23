"""RAG 问答：bge 检索 top-k 知识块 -> 拼进 prompt -> Qwen 生成答案。

用法：
  python rag/query.py "胎压多少"      # 单次提问
  python rag/query.py                  # 进入交互模式
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from retrieve import Retriever  # noqa: E402

SYSTEM = ("你是车载用户手册助手。只根据下面提供的【手册资料】回答用户问题，"
          "用简洁中文回答；如果资料里没有相关信息，就直说手册中未提及，不要编造。")


def build_prompt(question, hits):
    ctx = "\n\n".join(f"【资料{i+1}·{c['title']}】\n{c['text']}"
                      for i, (c, _) in enumerate(hits))
    user = f"【手册资料】\n{ctx}\n\n【问题】{question}"
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


class RagBot:
    def __init__(self):
        self.retriever = Retriever()
        print(f"加载 LLM: {config.LLM_MODEL} ...")
        self.tok = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

    @torch.no_grad()
    def answer(self, question):
        hits = self.retriever.search(question)
        messages = build_prompt(question, hits)
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        gen = out[0][inputs.input_ids.shape[1]:]
        ans = self.tok.decode(gen, skip_special_tokens=True).strip()
        return ans, hits


def show(question, ans, hits):
    print("\n" + "=" * 60)
    print(f"问：{question}")
    print(f"答：{ans}")
    print("-" * 60)
    print("命中知识块：")
    for c, score in hits:
        print(f"  [{score:.3f}] {c['title']}")
    print("=" * 60)


def main():
    bot = RagBot()
    args = sys.argv[1:]
    if args:
        q = " ".join(args)
        ans, hits = bot.answer(q)
        show(q, ans, hits)
        return
    print("\n进入交互问答（输入 q 退出）")
    while True:
        try:
            q = input("\n请输入问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("q", "quit", "exit", ""):
            break
        ans, hits = bot.answer(q)
        show(q, ans, hits)


if __name__ == "__main__":
    main()

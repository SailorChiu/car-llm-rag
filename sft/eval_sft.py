"""SFT 评测：在 val 集上对比 base 与 SFT 的「函数名准确率」「完全匹配率」。

判分口径来自 schema.py（与造数据同一套）。
- 没有 LoRA 适配器时：只评 base（可在训练前先看 base 到底会不会，验证有没有差距可做）。
- 有 sft/out 适配器时：base vs SFT 并排打印。
"""
import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from schema import score, score_intent  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def load_val():
    rows = []
    with open(config.VAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_inputs(tok, user_text):
    msgs = [{"role": "system", "content": config.SFT_SYSTEM},
            {"role": "user", "content": user_text}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt")


@torch.no_grad()
def predict_all(model, tok, rows):
    preds = []
    for r in rows:
        inp = build_inputs(tok, r["messages_user"]).to(model.device)
        out = model.generate(**inp, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = out[0][inp.input_ids.shape[1]:]
        preds.append(tok.decode(gen, skip_special_tokens=True).strip())
    return preds


def evaluate(preds, rows):
    """返回 (完全匹配率, 函数名全名准确率, 意图宽口径准确率)。"""
    n = len(rows)
    func_ok = exact_ok = intent_ok = 0
    for p, r in zip(preds, rows):
        f_ok, e_ok = score(p, r["func"], r["args"])
        func_ok += f_ok
        exact_ok += e_ok
        intent_ok += score_intent(p, r["func"], r["args"])
    return exact_ok / n, func_ok / n, intent_ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=6, help="打印几条定性对比样例")
    args = ap.parse_args()

    rows = load_val()
    tok = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    print(f"加载 base: {config.LLM_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL, dtype=torch.float16, device_map="auto")
    base.eval()

    base_preds = predict_all(base, tok, rows)
    base_m = evaluate(base_preds, rows)  # (完全匹配, 函数名全名, 意图宽口径)

    sft_preds = None
    if os.path.isdir(config.SFT_OUT_DIR) and os.listdir(config.SFT_OUT_DIR):
        from peft import PeftModel
        print(f"加载 SFT 适配器: {config.SFT_OUT_DIR}")
        sft = PeftModel.from_pretrained(base, config.SFT_OUT_DIR)
        sft.eval()
        sft_preds = predict_all(sft, tok, rows)
        sft_m = evaluate(sft_preds, rows)

    def row(name, m):
        return f"{name:<8}{m[0]*100:>10.1f}%{m[1]*100:>14.1f}%{m[2]*100:>14.1f}%"

    print("\n" + "=" * 60)
    print(f"val 集 {len(rows)} 条")
    print(f"{'':<8}{'完全匹配率':>9}{'函数名(全名)':>13}{'意图(宽口径)':>13}")
    print("-" * 60)
    print(row("base", base_m))
    if sft_preds is not None:
        print(row("SFT", sft_m))
    else:
        print("SFT     (无适配器，先训练再评)")
    print("=" * 60)

    # ---- SFT 错误诊断：逐函数命中 + 全部函数名错 + 全部参数错（对着真实错误补数据用）----
    if sft_preds is not None:
        from collections import defaultdict
        per = defaultdict(lambda: [0, 0])  # func -> [对, 总]
        func_errs, arg_errs = [], []
        for p, r in zip(sft_preds, rows):
            f_ok, e_ok = score(p, r["func"], r["args"])
            per[r["func"]][1] += 1
            per[r["func"]][0] += f_ok
            if not f_ok:
                func_errs.append((r["messages_user"], r["target"], p))
            elif not e_ok:
                arg_errs.append((r["messages_user"], r["target"], p))
        print("\n--- SFT 逐函数 函数名准确率 ---")
        for fn in sorted(per):
            ok, tot = per[fn]
            print(f"  {fn:<26}{ok}/{tot}")
        print(f"\n--- 函数名判错 {len(func_errs)} 条 ---")
        for u, g, p in func_errs:
            print(f"  [{u}]\n    gold: {g}\n    pred: {p}")
        print(f"\n--- 函数对但参数错 {len(arg_errs)} 条 ---")
        for u, g, p in arg_errs:
            print(f"  [{u}]\n    gold: {g}\n    pred: {p}")

    print(f"\n定性样例（前 {args.show} 条）:")
    for i in range(min(args.show, len(rows))):
        print(f"\n[{rows[i]['messages_user']}]  gold: {rows[i]['target']}")
        print(f"  base: {base_preds[i]}")
        if sft_preds is not None:
            print(f"  SFT : {sft_preds[i]}")


if __name__ == "__main__":
    main()

"""RAG 检索评测：对一组问题做检索，看正确知识块有没有进 top-k。

命中判定：期望关键词只要出现在 top-k 命中块文本里，就算命中（hit）。
目标：hit@3 >= 90%。只用检索器，不加载 LLM，几秒出结果。
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieve import Retriever  # noqa: E402

# (问题, 期望命中的关键词列表) —— 关键词出现在命中块里即算命中
EVAL_SET = [
    ("胎压一般打多少合适",        ["235", "2.4"]),
    ("空调温度最高能开到多少度",   ["32℃"]),
    ("怎么打开座椅加热",          ["座椅加热"]),
    ("支持快充吗，充满要多久",     ["直流快充"]),
    ("仪表盘亮红灯是什么意思",     ["红色警告灯"]),
    ("怎么进入洗车模式",          ["洗车模式"]),
    ("雨刮片怎么更换",            ["维护位置"]),
    ("怎么给车机升级系统",        ["OTA"]),
    ("如何开启自适应巡航",        ["自适应巡航"]),
    ("车子坏了不能开能拖走吗",     ["牵引模式"]),
]


def hit_at(hits, keywords, k):
    """top-k 命中块里是否出现任一关键词。"""
    blob = "".join(c["text"] for c, _ in hits[:k])
    return any(kw in blob for kw in keywords)


def main():
    r = Retriever()
    k = 3
    n = len(EVAL_SET)
    hit1 = hit3 = 0
    print(f"{'问题':<22}{'hit@1':<8}{'hit@3':<8}top1命中块")
    print("-" * 70)
    for q, kws in EVAL_SET:
        hits = r.search(q, top_k=k)
        h1 = hit_at(hits, kws, 1)
        h3 = hit_at(hits, kws, 3)
        hit1 += h1
        hit3 += h3
        top1_title = hits[0][0]["title"] if hits else "(空)"
        print(f"{q:<22}{'✓' if h1 else '✗':<8}{'✓' if h3 else '✗':<8}{top1_title}")
    print("-" * 70)
    print(f"hit@1 = {hit1}/{n} = {hit1/n*100:.1f}%")
    print(f"hit@3 = {hit3}/{n} = {hit3/n*100:.1f}%   (目标 >= 90%)")


if __name__ == "__main__":
    main()

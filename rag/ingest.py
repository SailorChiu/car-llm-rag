"""RAG 第一步：把手册按 '## ' 切块 -> bge 向量化(归一化) -> faiss 内积索引存盘。

内积 + 归一化向量 == 余弦相似度，所以用 IndexFlatIP。
"""
import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402


def split_manual(path):
    """按 '## ' 标题切块。返回 [{'title':..., 'text':...}, ...]。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    chunks = []
    title, buf = None, []
    for line in raw.splitlines():
        if line.startswith("## "):
            # 遇到新标题，先把上一块存起来
            if title is not None:
                chunks.append({"title": title, "text": ("\n".join([f"## {title}"] + buf)).strip()})
            title = line[3:].strip()
            buf = []
        elif title is not None:
            buf.append(line)
        # 第一个 '## ' 之前的内容（# 大标题 / 说明）忽略
    if title is not None:
        chunks.append({"title": title, "text": ("\n".join([f"## {title}"] + buf)).strip()})
    return chunks


def main():
    os.makedirs(config.RAG_INDEX_DIR, exist_ok=True)
    chunks = split_manual(config.MANUAL_PATH)
    print(f"切出 {len(chunks)} 个知识块：")
    for c in chunks:
        print(f"  - {c['title']}")

    print(f"\n加载 embedding 模型: {config.EMB_MODEL}")
    model = SentenceTransformer(config.EMB_MODEL)

    texts = [c["text"] for c in chunks]
    # 文档侧不加检索指令前缀（前缀只加在 query 侧）
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=True)
    emb = emb.astype("float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, os.path.join(config.RAG_INDEX_DIR, "manual.index"))
    with open(os.path.join(config.RAG_INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n索引已保存到 {config.RAG_INDEX_DIR}（{len(chunks)} 块, 维度 {dim}）")


if __name__ == "__main__":
    main()

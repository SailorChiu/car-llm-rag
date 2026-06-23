"""共享检索逻辑：加载 faiss 索引 + bge 模型，对 query 做 top-k 检索。

query.py（生成答案）和 eval_rag.py（算命中率）都复用这里，避免重复。
"""
import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

import faiss  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

# bge 中文检索：query 侧要加这个指令前缀，文档侧不加（ingest 里已遵守）
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Retriever:
    def __init__(self):
        index_path = os.path.join(config.RAG_INDEX_DIR, "manual.index")
        chunks_path = os.path.join(config.RAG_INDEX_DIR, "chunks.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"找不到索引 {index_path}，请先运行: bash run.sh rag-build")
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.model = SentenceTransformer(config.EMB_MODEL)

    def search(self, query, top_k=None):
        """返回 [(chunk_dict, score), ...]，按相似度降序。"""
        top_k = top_k or config.TOP_K
        q = BGE_QUERY_INSTRUCTION + query
        emb = self.model.encode([q], normalize_embeddings=True,
                                convert_to_numpy=True).astype("float32")
        scores, idxs = self.index.search(emb, top_k)
        out = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            out.append((self.chunks[i], float(score)))
        return out

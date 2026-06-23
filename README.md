# car-llm —— 车载大模型练习工程（RAG + SFT）

一个对标"SFT+RAG 车载实习岗"的最小可跑工程：在一块 RTX 4060 Laptop (8GB) 上，
用 Qwen2.5-1.5B 同时做两件事——**手册知识问答 (RAG)** 和 **车控指令解析 (QLoRA 微调)**。
重点不在堆指标，而在**方法论是否干净**：可复现的评测、难例诊断、对着真实错误做定向迭代、
以及对测试集的纪律（不反复刷 val）。

## 环境
- WSL2 Ubuntu + RTX 4060 Laptop 8GB，CUDA 直通。
- Python 3.12 venv（非 conda），`torch 2.5.1+cu121`、transformers 5.x、peft、bitsandbytes。
- **模型源用 HF 官方 `huggingface.co`**：本机 `hf-mirror.com` 连不上，直连官方反而可用
  （`run.sh` 已默认官方源，可用 `HF_ENDPOINT` 覆盖）。

```bash
cd /home/kaifan/car-llm && source .venv/bin/activate   # 每次新终端
bash run.sh rag        # = rag-build + rag-eval
bash run.sh rag-chat "胎压多少"
bash run.sh sft        # = sft-data + sft-train + sft-eval
```

---

## Demo A：车辆手册 RAG 问答
**流程**：手册按 `## ` 切 18 块 → `bge-small-zh` 归一化向量 → FAISS 内积索引（=余弦相似度）
→ query 加 bge 检索指令前缀取 top-3 → 拼进 prompt 交 Qwen 生成，系统提示"只据资料回答、
无则说没有"以抑制幻觉。

**结果（10 题检索评测）**：

| 指标 | 数值 |
|---|---|
| Hit@3 | **100.0%** (10/10) |
| Hit@1 | 90.0% (9/10) |

`bash run.sh rag-chat "胎压多少"` → `标准冷态胎压为 2.4 bar，约合 235 kPa，前后轮一致。`（命中"胎压"块）

---

## Demo B：车控指令 SFT（QLoRA）
**任务**：把中文口语指令解析成**严格 JSON 函数调用**，针对一套固定的 10 函数车控 API
（`climate.set_temperature` / `media.set_volume` / `window.set_position` / `system.reject` …），
只输出紧凑 JSON、参数名/枚举锁死、域外请求走 `system.reject`。

**方法**：Qwen2.5-1.5B + 4bit(nf4) QLoRA，仅训练 LoRA 适配器（**18.4M / 1.56B = 1.18%**），
**label mask 只在答案 JSON 上算 loss**；batch=2 + 梯度累积 + gradient_checkpointing +
paged_adamw_8bit，单卡约 2 分钟/轮。

**评测纪律**：val 与 train **按措辞模板切分、严格不重叠**（程序校验交集为 0），测的是 schema
泛化而非背句子；判分口径统一在 `sft/schema.py`，train/eval 共用。报三个指标避免被质疑成稻草人：

| 指标 | 含义 |
|---|---|
| 完全匹配率 | 函数名 + 参数字典都严格一致（头条指标）|
| 函数名准确率(全名) | 带域前缀的 API 合规度 |
| 意图准确率(宽口径) | 函数只比末段、参数只比值，证明 base "听懂了意图、只是不合规" |

**结果（54 条 held-out val，base vs SFT）**：

| | 完全匹配率 | 函数名(全名) | 意图(宽口径) |
|---|---|---|---|
| base | 0.0% | 0.0% | 9.3% |
| SFT（首轮 117 条） | 74.1% | 88.9% | 83.3% |
| **SFT（定向迭代后 187 条）** | **94.4%** | **96.3%** | **96.3%** |

**迭代是数据驱动的**：首轮后读 `sft-eval` 的逐条错误，发现 (a) `media.set_volume` 把参数名
串成 window 的 `percent`、(b) `set_fan_speed` 没学过"风速"一词、(c) reject 主题太窄。
**只往 train 侧补对比/泛化样例**（措辞仍与 val 不重叠），func-name 88.9%→96.3%、exact 74.1%→94.4%。

### 局限性（如实记录，仅一次重训额度，不反复刷 val 以免过拟合测试集）
残余 3/54 错误：
1. `推荐一部好看的电影` → `media.control{play}`：OOD 拒答边界在贴近域内话题（电影/音乐）时仍模糊。
2. `帮我写封请假邮件` → `mail.reject`：模型正确拒答，但把域前缀**幻觉**成 `mail.`（应 `system.`），
   故"意图对、全名错"。
3. `所有车窗都升上去` → `percent:100`（应 0）：方向词"升上去"歧义未根除。

可继续改进的方向（未做，以保持评测纪律）：更大且更均衡的拒答/方向语料、把可用函数 schema
显式放进 system prompt 做 grounded function-calling、或上更大基座。

---

## 英文简历 bullet（真实指标）
- **Retrieval-Augmented QA**: Built a Chinese vehicle-manual RAG assistant
  (BAAI/bge-small-zh embeddings + FAISS cosine retrieval + Qwen2.5-1.5B generation),
  reaching **100% Hit@3 / 90% Hit@1** on a 10-question eval set and grounding answers in
  retrieved manual sections to suppress hallucination.
- **Instruction Fine-Tuning (QLoRA)**: Fine-tuned Qwen2.5-1.5B with **4-bit QLoRA**
  (LoRA, 18M trainable params ≈ 1.2%) to map Chinese voice commands to strict JSON function
  calls for a fixed 10-function car-control API; lifted held-out **exact-match 0%→94%** and
  **function-name accuracy 0%→96%** over the base model, using a **template-disjoint val split**
  and one **error-driven data iteration** (contrastive examples to fix argument-name / intent
  confusions) on a single 8 GB RTX 4060.

---

## 目录
```
config.py            模型名/路径/SFT 系统提示词（环境变量可覆盖）
run.sh               阶段运行器：rag-build/rag-eval/rag-chat/sft-data/sft-train/sft-eval
data/manual.md       中文车辆手册（18 段）
data/train|val.jsonl SFT 数据（措辞模板级 train/val 切分）
rag/ingest.py        切块→bge 向量→FAISS 索引
rag/retrieve.py      共享检索器（query 加 bge 指令前缀）
rag/query.py         RAG 生成（检索 top-k → Qwen）
rag/eval_rag.py      Hit@k 评测
sft/schema.py        固定函数 schema + 判分（造数据/评测共用）
sft/make_data.py     造数据（含难例：口语模糊、域外拒答；定向补例）
sft/train_qlora.py   QLoRA 训练（label mask 只在答案算 loss）
sft/eval_sft.py      base vs SFT 三指标 + 逐条错误诊断
```

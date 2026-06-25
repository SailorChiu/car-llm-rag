# car-llm-rag

> 车载场景大模型助手 —— 在消费级硬件（RTX 4060 Laptop 8G / WSL2）上，基于 **Qwen2.5-1.5B** 实现「**指令解析 SFT + 车辆手册 RAG 问答**」两个 demo，核心是一套可复现、防泄漏的**评测方法论闭环**。

面向「车载大模型 SFT + RAG」方向的求职作品。重点不在堆模型规模，而在于：**在算力受限的真实约束下，把一个端侧可部署的车载问答/指令系统做扎实，并用三指标量化每一步优化的收益。**

---

## 🚀 快速开始

```bash
cd /home/kaifan/car-llm && source .venv/bin/activate   # 每次新终端
bash run.sh rag                 # = rag-build + rag-eval
bash run.sh rag-chat "胎压多少"  # 单条问答
bash run.sh sft                 # = sft-data + sft-train + sft-eval
```

> 模型源默认走 HF 官方 `huggingface.co`（本机 `hf-mirror.com` 连不上，直连官方反而可用），可用 `HF_ENDPOINT` 覆盖。

---

## ✨ 核心成果

### 主线 A — 指令解析 SFT（面试主打）
中文语音指令 → **严格 JSON 函数调用**（固定 10 个车控 API，如 `climate.set_temperature` / `media.set_volume` / `window.set_position` / `system.reject`），只输出紧凑 JSON、参数名/枚举锁死、域外请求走 `system.reject`。

- **4bit(nf4) QLoRA** 微调，仅训练 LoRA adapter（**18.4M / 1.56B = 1.18%** 参数，base 冻结）
- **label mask 只在答案 JSON 上算 loss**；batch=2 + 梯度累积 + gradient_checkpointing + paged_adamw_8bit，单卡约 2 分钟/轮
- val 与 train **按措辞模板严格切分、交集为 0**，测的是 schema 泛化而非背句子

**结果（54 条 held-out val，base vs SFT）**：

| | 完全匹配率 | 函数名(全名) | 意图(宽口径) |
|---|---|---|---|
| base | 0.0% | 0.0% | 9.3% |
| SFT（首轮 117 条） | 74.1% | 88.9% | 83.3% |
| **SFT（定向迭代后 187 条）** | **94.4%** | **96.3%** | **96.3%** |

> 关键不是「0→94」这个跳变本身，而是**这个差距是我故意用严格 schema 造出来的** —— base 在严格 JSON 约束下必然失败，从而把「意图识别」和「参数合规」拆成两个可独立优化的指标。

### 主线 B — 车辆手册 RAG 问答
- **切块**：手册按 `## ` 切 **18 块**
- **检索**：`bge-small-zh` 归一化向量 + **FAISS 内积索引（=余弦相似度）**，query 加 bge 检索指令前缀取 top-3
- **生成**：检索上下文 + Qwen2.5-1.5B，系统提示「只据资料回答、无则说没有」以抑制幻觉
- 10 题自建 eval：**Hit@3 = 100% (10/10) / Hit@1 = 90% (9/10)**

```
bash run.sh rag-chat "胎压多少"
→ 标准冷态胎压为 2.4 bar，约合 235 kPa，前后轮一致。   # 命中「胎压」块
```

---

## 🎯 核心亮点：评测方法论闭环

本项目的真正差异点不是模型，而是**工程方法论** —— 这也是面试主线：

| 维度 | 做法 |
|------|------|
| **故意造差距** | 用严格 JSON schema 让 base 必然失败，制造 base→SFT 的可测差距，证明微调的真实增益 |
| **三指标拆解** | 不只看「对不对」，而是把 **意图识别（函数名）** 与 **参数合规（完全匹配）** 拆开，分别归因 |
| **防数据泄漏** | 按**措辞模板**切 train / val，同一指令的不同说法不跨集，避免「背答案」式虚高 |
| **错误驱动迭代** | 读**完整错误清单**做定向归因，针对性补数据，而非盲目加样本 |
| **防过拟合测试集** | 全程只卡**一次**评测额度、不反复刷 val，杜绝对测试集过拟合 |

**一次数据驱动迭代的实例**：首轮后读 `sft-eval` 的逐条错误，发现 (a) `media.set_volume` 把参数名串成 window 的 `percent`、(b) `set_fan_speed` 没学过「风速」一词、(c) reject 主题太窄。**只往 train 侧补对比 / 泛化样例**（措辞仍与 val 不重叠），func-name 88.9%→96.3%、exact 74.1%→94.4%。

---

## 🏗️ 系统架构

```
[主线 A 指令解析]                    [主线 B 手册问答]
中文语音指令                          用户提问
   │                                   │
   ▼                                   ▼
Qwen2.5-1.5B + QLoRA adapter      bge-small-zh + FAISS Top-K
   │                                   │
   ▼                                   ▼
严格 JSON 函数调用                 检索上下文 + Qwen2.5-1.5B
   │                                   │
   └──────────► 三指标评测 + bad case 归因 ◄──────────┘
                       │
                       ▼
                  错误驱动迭代
```

---

## 🛠️ 技术栈与环境

`Qwen2.5-1.5B` · `QLoRA / PEFT` · `bitsandbytes (4bit nf4)` · `transformers` · `bge-small-zh + FAISS` · `WSL2` · `RTX 4060 Laptop 8G`

- **环境**：WSL2 Ubuntu + Python 3.12 venv（非 conda），`torch 2.5.1+cu121`、`transformers`、`peft`、`bitsandbytes`，CUDA 直通
- **踩坑记录**：Windows 本机 Python 3.14 无 torch 轮子，训练务必在 WSL；`bitsandbytes` / QLoRA 在 Linux 原生顺畅、Windows 上多坑

---

## ⚠️ 局限性（如实记录，仅一次重训额度，不反复刷 val 以免过拟合测试集）

残余 3/54 错误：
1. `推荐一部好看的电影` → `media.control{play}`：OOD 拒答边界在贴近域内话题（电影 / 音乐）时仍模糊。
2. `帮我写封请假邮件` → `mail.reject`：模型正确拒答，但把域前缀**幻觉**成 `mail.`（应 `system.`），故「意图对、全名错」。
3. `所有车窗都升上去` → `percent:100`（应 0）：方向词「升上去」歧义未根除。

可继续改进（未做，以保持评测纪律）：更大且更均衡的拒答 / 方向语料、把可用函数 schema 显式放进 system prompt 做 grounded function-calling、或上更大基座。

---

## 📁 目录结构

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

---

## 🗺️ Roadmap

- [x] 指令解析 QLoRA SFT（完全匹配 94.4% / 函数名 96.3%）
- [x] 手册 RAG 问答（Hit@3 100% / Hit@1 90%）
- [x] 三指标评测 + 防泄漏切分 + 错误驱动迭代
- [ ] **端侧部署优化（进行中）**：llama.cpp / GGUF 量化（Q8 / Q5 / Q4_K_M 对照 benchmark），测端侧显存与 tok/s
- [ ] vLLM 服务端推理对照
- [ ] 扩充评测集规模（当前 10 题 → 50+）

---

## 📝 English résumé bullets（真实指标）

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

## 🤖 Agent / 工具调用 Demo

> **一句话总结**：本 demo 演示了 意图识别 → 工具调用（RAG / mock 车控）→ 结果回填 → 多轮记忆 的闭环。

详见 [agent/README.md](agent/README.md)。

```bash
cd ~/car-llm && source .venv/bin/activate
python agent/agent_demo.py
```

实现要点：

- **意图识别**：引导 Qwen2.5-1.5B 输出 function-calling 风格 JSON（`query_manual` / `set_climate` / `control_window` / `chitchat`）
- **工具调用**：RAG 检索（复用现有 bge+FAISS）/ 车控 Mock（⚠️ MOCK / 演示用，非真实车控）
- **多轮记忆**：`messages` 历史列表 + `MEMORY` dict 跨轮记录用户偏好
- **小模型容错**：正则抠 JSON + parse 失败 fallback，1.5B 模型格式不稳时不崩溃

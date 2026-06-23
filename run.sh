#!/usr/bin/env bash
# 车载大模型练习工程：RAG + SFT 阶段化运行器
# 用法: bash run.sh <stage>
#   rag-build  构建手册向量索引
#   rag-eval   评测检索命中率 hit@k
#   rag-chat   进入手册问答（可直接 bash run.sh rag-chat "胎压多少"）
#   sft-data   生成车控指令微调数据
#   sft-train  QLoRA 微调 Qwen2.5-1.5B
#   sft-eval   对比微调前后准确率
#   rag        = build + eval
#   sft        = data + train + eval
set -e
cd "$(dirname "$0")"

# 自动激活本项目 venv（没激活时），避免误用系统 python
if [ -z "$VIRTUAL_ENV" ] && [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

# 拉模型用。注意：本机 hf-mirror.com 连不上，直连官方源反而可用，故默认官方源。
# 如需换镜像，运行前 export HF_ENDPOINT=... 覆盖即可。
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
# 关掉 tokenizers 多进程告警
export TOKENIZERS_PARALLELISM=false

PY=${PY:-python}

case "$1" in
  rag-build) $PY rag/ingest.py ;;
  rag-eval)  $PY rag/eval_rag.py ;;
  rag-chat)  shift; $PY rag/query.py "$@" ;;
  sft-data)  $PY sft/make_data.py ;;
  sft-train) $PY sft/train_qlora.py ;;
  sft-eval)  $PY sft/eval_sft.py ;;
  rag)       $PY rag/ingest.py && $PY rag/eval_rag.py ;;
  sft)       $PY sft/make_data.py && $PY sft/train_qlora.py && $PY sft/eval_sft.py ;;
  *) echo "用法: bash run.sh [rag-build|rag-eval|rag-chat|sft-data|sft-train|sft-eval|rag|sft]" ;;
esac

import os

# ====== 模型（可用环境变量覆盖）======
# 国内拉不动 HuggingFace 时，run.sh 里已设 HF_ENDPOINT=https://hf-mirror.com
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
EMB_MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-small-zh-v1.5")

# ====== 路径 ======
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RAG_INDEX_DIR = os.path.join(ROOT, "rag", "index")
SFT_OUT_DIR = os.path.join(ROOT, "sft", "out")

MANUAL_PATH = os.path.join(DATA_DIR, "manual.md")
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH = os.path.join(DATA_DIR, "val.jsonl")

# ====== 超参 ======
TOP_K = 3

# SFT / 指令解析统一使用的系统提示词（训练、评测必须一致）
SFT_SYSTEM = "你是车载指令解析器。把用户的话解析成一个JSON，包含 function 和 arguments 两个字段，只输出JSON，不要多余文字。"

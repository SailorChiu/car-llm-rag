#!/usr/bin/env bash
# 第0步：在 WSL 里建 venv 并安装依赖（torch 走 cu121），最后做 CUDA 自检。
# 全程日志写到 setup.log，可后台跑 + tail 观察。
set -e
cd /home/kaifan/car-llm

echo "==== [0/4] python & 目录 ===="
python3 --version
pwd

echo "==== [1/4] 建 venv ===="
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip

echo "==== [2/4] 装 torch (cu121) ===="
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "==== [3/4] 装其余依赖 (PyPI) ===="
pip install transformers sentence-transformers faiss-cpu peft datasets accelerate bitsandbytes

echo "==== [4/4] CUDA 自检 ===="
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch:", torch.__version__)
print("cuda_available:", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0))
    print("CUDA_CHECK_PASS")
else:
    print("CUDA_CHECK_FAIL")
PY

echo "==== DONE ===="

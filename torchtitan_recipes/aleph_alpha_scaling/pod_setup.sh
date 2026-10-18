#!/usr/bin/bash
# One-time setup after each RunPod start (the container disk is wiped on stop).
# Run on the pod from the repo root (/root/torchtitan) after rsyncing it over.
set -euo pipefail

command -v uv >/dev/null || pip install -q --break-system-packages uv
uv venv -q -p 3.12 /root/venv
source /root/venv/bin/activate

uv pip install -q -r requirements.txt
uv pip install -q -e . --no-deps
# requirements.txt can pull a stable torch over the nightly; install nightly last.
# cu130 matches the host driver (580.x).
uv pip install -q --pre --reinstall-package torch torch \
    --index-url https://download.pytorch.org/whl/nightly/cu130
# FlashAttention 3: the attention kernel torchtitan selects on H100 (sm90).
uv pip install -q --pre flash_attn_3 --no-deps \
    --index-url https://download.pytorch.org/whl/nightly/cu130
uv pip install -q matplotlib

# Tokenizers only (no weights: these runs train from random init).
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-1.7B --assets tokenizer
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-30B-A3B --assets tokenizer

grep -q "source /root/venv/bin/activate" ~/.bashrc ||
    echo "source /root/venv/bin/activate; cd /root/torchtitan" >>~/.bashrc

python -c "import torch, flash_attn_interface; print(torch.__version__, torch.cuda.get_device_name())"
which runpodctl && echo "pod id: ${RUNPOD_POD_ID:-<unset>}"

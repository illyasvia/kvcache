#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/main.py" \
    --model_type qwen35 \
    --kv_mode origin \
    --prompt_compression longllmlingua \
    --compressor_model_path microsoft/phi-2 \
    --compressor_device auto \
    --compression_rate 0.5 \
    --compression_chunk_tokens 512 \
    --compression_iterative_size 200 \
    --compression_reorder sort \
    --compression_dynamic_ratio 0.3 \
    --lang en \
    --context_lengths 2000 16000 \
    --depths 0 52 100

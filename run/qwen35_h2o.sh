#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/main.py" \
    --model_type qwen35 \
    --kv_mode h2o \
    --h2o_heavy_hitter_size 1024 \
    --h2o_recent_size 1024 \
    --h2o_chunk_size 1024 \
    --lang en \
    --context_lengths 32000 48000 100000 300000 \
    --depths 0 52 100

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/main.py" \
    --model_type qwen35 \
    --kv_mode h2o \
    --h2o_heavy_hitter_size 512 \
    --h2o_recent_size 512 \
    --h2o_chunk_size 512 \
    --lang en \
    --dataset_dir "${ROOT_DIR}/dataset/needlebench_v2" \
    --dataset_suffix 1000k \
    --scoring needlebench_v2 \
    --output_dir "${ROOT_DIR}/output/needlebench_v2_1000k_h2o" \
    --context_lengths 1000000 \
    --depths 0 50 100 \
    --max_new_tokens 128

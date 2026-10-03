#!/usr/bin/env bash
set -euo pipefail

python multimodal_eval.py \
    --dataset docvqa \
    --model_type qwen3vl \
    --limit "${LIMIT:-100}" \
    --max_new_tokens 128 \
    --resume

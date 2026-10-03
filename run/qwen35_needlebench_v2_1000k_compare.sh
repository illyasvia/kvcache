#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
DATASET_DIR="${ROOT_DIR}/dataset/needlebench_v2"
LOG_DIR="${ROOT_DIR}/output/needlebench_v2_1000k_logs"
REPORT_SCRIPT="${ROOT_DIR}/dataset/needle/generate_comparison_report.py"
mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
    --model_type qwen35
    --lang en
    --dataset_dir "${DATASET_DIR}"
    --dataset_suffix 1000k
    --scoring needlebench_v2
    --context_lengths 1000000
    --depths 0 50 100
    --max_new_tokens 128
    --prefill_chunk_size 512
    --resume
)

run_mode() {
    local mode="$1"
    shift
    local log_path="${LOG_DIR}/${mode}.log"
    echo "[$(date '+%F %T')] 启动 ${mode}" | tee -a "${log_path}"
    "${PYTHON}" "${ROOT_DIR}/main.py" "${COMMON_ARGS[@]}" "$@" 2>&1 | tee -a "${log_path}"
    local exit_code=${PIPESTATUS[0]}
    printf '%s\n' "${exit_code}" > "${LOG_DIR}/${mode}.exit_code"
    echo "[$(date '+%F %T')] ${mode} 结束，exit_code=${exit_code}" | tee -a "${log_path}"
    "${PYTHON}" "${REPORT_SCRIPT}"
}

run_mode h2o \
    --kv_mode h2o \
    --h2o_heavy_hitter_size 512 \
    --h2o_recent_size 512 \
    --h2o_chunk_size 512 \
    --output_dir "${ROOT_DIR}/output/needlebench_v2_1000k_h2o"

run_mode longllmlingua \
    --kv_mode origin \
    --prompt_compression longllmlingua \
    --compressor_model_path microsoft/phi-2 \
    --compressor_device auto \
    --compression_rate 0.5 \
    --compression_chunk_tokens 512 \
    --compression_iterative_size 200 \
    --compression_reorder sort \
    --compression_dynamic_ratio 0.3 \
    --output_dir "${ROOT_DIR}/output/needlebench_v2_1000k_longllmlingua"

run_mode origin \
    --kv_mode origin \
    --output_dir "${ROOT_DIR}/output/needlebench_v2_1000k_origin"

"${PYTHON}" "${REPORT_SCRIPT}"

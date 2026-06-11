#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"
# Reproduce the original split sweep layout used for this series.
SEQ_LEN=256 \
MODEL_TAG=nl2_do01 \
TRAIN_EPOCHS=15 \
DISTILL_STEP_VALUES=400 \
K_VALUES="4 8 16 32" \
K_TAG=k4_8_16_32 \
"${script_dir}/run_tm_topk_gumbel_sweep_local.sh"

SEQ_LEN=256 \
MODEL_TAG=nl2_do01 \
TRAIN_EPOCHS=15 \
DISTILL_STEP_VALUES=400 \
K_VALUES="64 128 256" \
K_TAG=k64_128_256 \
"${script_dir}/run_tm_topk_gumbel_sweep_local.sh"

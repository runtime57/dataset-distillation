#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"
SEQ_LEN=128 \
MODEL_TAG=nl2_do01 \
TRAIN_EPOCHS=15 \
DISTILL_STEP_VALUES="100 200 300 400 500 600" \
K_VALUES="64 128 256" \
"${script_dir}/run_tm_topk_gumbel_sweep_local.sh"

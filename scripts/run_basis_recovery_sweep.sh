#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

BASE_CMD=(
  python3 distill.py
  --config-name distill_soft_tokens_gm_10k_seq256_n256_local_bpe_1024
  distillation.n_steps=200
  distillation.log_step=20
  distillation.save_period=50
  distillation.optimizer.lr=0.0003
  distillation.entropy_weight=0.01
  dataloader.num_workers=0
  distillation.synthetic.init_mode=kmeans
)

run_variant() {
  local run_name="$1"
  shift
  echo "===== ${run_name} ====="
  "${BASE_CMD[@]}" "distillation.run_name=${run_name}" "$@"
}

run_variant \
  "distill256_local_bpe_1024_gm_anchors64_recovery_kmeans_lr3e4_ent1e2" \
  distillation.synthetic.parameterization=anchors \
  +distillation.synthetic.num_anchors=64

run_variant \
  "distill256_local_bpe_1024_gm_concepts64_recovery_kmeans_scale8_lr3e4_ent1e2" \
  distillation.synthetic.parameterization=concepts \
  +distillation.synthetic.num_concepts=64 \
  +distillation.synthetic.concept_input_mode=concepts \
  +distillation.synthetic.concept_logit_scale=8.0

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
  distillation.n_steps=400
  distillation.log_step=20
  distillation.save_period=50
  dataloader.num_workers=0
  distillation.synthetic.parameterization=concepts
  +distillation.synthetic.num_concepts=64
)

run_variant() {
  local run_name="$1"
  shift
  echo "===== ${run_name} ====="
  "${BASE_CMD[@]}" "distillation.run_name=${run_name}" "$@"
}

run_variant \
  "distill256_local_bpe_1024_gm_concepts64_debug_probs_scale32_lr1e3" \
  distillation.optimizer.lr=0.001 \
  +distillation.synthetic.concept_input_mode=probs \
  +distillation.synthetic.concept_logit_scale=32.0

run_variant \
  "distill256_local_bpe_1024_gm_concepts64_debug_concepts_scale32_lr1e3" \
  distillation.optimizer.lr=0.001 \
  +distillation.synthetic.concept_input_mode=concepts \
  +distillation.synthetic.concept_logit_scale=32.0

run_variant \
  "distill256_local_bpe_1024_gm_concepts64_debug_concepts_scale8_lr1e3" \
  distillation.optimizer.lr=0.001 \
  +distillation.synthetic.concept_input_mode=concepts \
  +distillation.synthetic.concept_logit_scale=8.0

run_variant \
  "distill256_local_bpe_1024_gm_concepts64_debug_concepts_scale8_lr3e4" \
  distillation.optimizer.lr=0.0003 \
  +distillation.synthetic.concept_input_mode=concepts \
  +distillation.synthetic.concept_logit_scale=8.0

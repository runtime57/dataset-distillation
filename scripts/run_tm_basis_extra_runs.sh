#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

COMMON_ARGS=(
  --config-name distill_soft_tokens_tm_10k_seq256_n256_local_bpe_1024
  dataloader.num_workers=0
  distillation.n_steps=200
  distillation.log_step=20
  distillation.save_period=50
  distillation.outer_batches=4
  distillation.n_inner_steps=5
  distillation.inner_lr=0.01
  distillation.optimizer.lr=0.001
)

python3 distill.py \
  "${COMMON_ARGS[@]}" \
  distillation.run_name=distill256_local_bpe_1024_tm_anchors64_extra_runs_s200 \
  distillation.synthetic.parameterization=anchors \
  +distillation.synthetic.num_anchors=64

python3 distill.py \
  "${COMMON_ARGS[@]}" \
  distillation.run_name=distill256_local_bpe_1024_tm_concepts64_extra_runs_s200 \
  distillation.synthetic.parameterization=concepts \
  +distillation.synthetic.num_concepts=64 \
  +distillation.synthetic.concept_input_mode=concepts \
  +distillation.synthetic.concept_logit_scale=32.0

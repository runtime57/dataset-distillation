#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <distill_run_dir_or_name> [n_epochs]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ARG="$1"
N_EPOCHS="${2:-5}"

if [[ "$RUN_ARG" = /* || "$RUN_ARG" == saved/* ]]; then
  DISTILL_DIR="$RUN_ARG"
else
  DISTILL_DIR="saved/distillation/$RUN_ARG"
fi

BEST_CKPT="$DISTILL_DIR/full_soft_tokens_best.pth"
if [[ ! -f "$BEST_CKPT" ]]; then
  BEST_CKPT="$DISTILL_DIR/full_soft_tokens.pth"
fi

if [[ ! -f "$BEST_CKPT" ]]; then
  echo "could not find distilled checkpoint in: $DISTILL_DIR" >&2
  exit 1
fi

RUN_BASENAME="$(basename "$DISTILL_DIR")"
EVAL_RUN_NAME="${RUN_BASENAME}_ppl_eval_${N_EPOCHS}ep"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

python3 train.py \
  --config-name tinystories_lm_softdistill_local_bpe_1024 \
  writer.run_name="${EVAL_RUN_NAME}" \
  trainer.save_dir="saved/softdistill_eval" \
  trainer.n_epochs="${N_EPOCHS}" \
  trainer.early_stop="${N_EPOCHS}" \
  trainer.override=true \
  +trainer.compile_enabled=true \
  +trainer.compile_mode=default \
  +trainer.compile_dynamic=false \
  model.max_seq_len=256 \
  model.d_model=128 \
  model.n_heads=4 \
  model.n_layers=4 \
  model.dim_feedforward=512 \
  model.dropout=0.0 \
  datasets.val.sequence_length='${model.max_seq_len}' \
  datasets.test.sequence_length='${model.max_seq_len}' \
  dataloader.num_workers=0 \
  datasets.train.checkpoint_path="${BEST_CKPT}"

echo
echo "summary log:"
echo "  ${ROOT_DIR}/saved/softdistill_eval/${EVAL_RUN_NAME}/info.log"

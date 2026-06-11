#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Sweep TM sparse soft-token parameterizations and evaluate each distilled
# checkpoint by training a fresh LM on it, then collecting val/test metrics.
#
# Common overrides:
#   K_VALUES="64 128 256" ./scripts/run_tm_topk_gumbel_sweep.sh
#   PARAMETERIZATIONS="topk" ./scripts/run_tm_topk_gumbel_sweep.sh
#   SKIP_PREPARE=1 ./scripts/run_tm_topk_gumbel_sweep.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONUNBUFFERED=1

log_stage() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

K_VALUES="${K_VALUES:-64 128 256}"
PARAMETERIZATIONS="${PARAMETERIZATIONS:-topk topk_gumbel}"
K_TAG="${K_TAG:-}"
MODEL_TAG="${MODEL_TAG:-nl2_do01}"

NSEQ="${NSEQ:-256}"
SEQ_LEN="${SEQ_LEN:-128}"
INIT_CONFIDENCE="${INIT_CONFIDENCE:-12.5}"
N_STEPS="${N_STEPS:-400}"
DISTILL_STEP_VALUES="${DISTILL_STEP_VALUES:-${N_STEPS}}"
DISTILL_CONFIG="${DISTILL_CONFIG:-distill_soft_tokens_tm_10k_n256_local_bpe_1024_seq${SEQ_LEN}}"
DISTILL_SAVE_ROOT="${DISTILL_SAVE_ROOT:-saved/distillation_sweep_tm_topk_gumbel_seq${SEQ_LEN}_${MODEL_TAG}}"
EVAL_SAVE_ROOT="${EVAL_SAVE_ROOT:-saved/eval_tm_topk_gumbel_seq${SEQ_LEN}_${MODEL_TAG}}"
RESULTS_ROOT="${RESULTS_ROOT:-saved/results_seq${SEQ_LEN}_${MODEL_TAG}}"
SOFT_TRAIN_CONFIG="${SOFT_TRAIN_CONFIG:-tinystories_lm_softdistill_local_bpe_1024}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
TRAIN_EPOCH_LEN="${TRAIN_EPOCH_LEN:-64}"
TRAIN_LR="${TRAIN_LR:-3e-4}"
TRAIN_LR_STEP_SIZE="${TRAIN_LR_STEP_SIZE:-}"
TRAIN_LR_GAMMA="${TRAIN_LR_GAMMA:-}"

DATA_PATH="${DATA_PATH:-data/tinystories_train_15k.txt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-artifacts/tokenizers/tinystories_bpe_1024/tokenizer.json}"
EXPERT_CONFIG="${EXPERT_CONFIG:-expert_trajectory_local_bpe_1024_seq${SEQ_LEN}}"
if [[ -z "${EXPERT_PATH:-}" ]]; then
  if [[ "${SEQ_LEN}" == "256" ]]; then
    EXPERT_PATH="saved/expert_trajectories/tiny_lm_local_bpe_1024_seq256_defaultmodel_sp5_lr001/expert_checkpoints.pth"
  else
    EXPERT_PATH="saved/expert_trajectories/tiny_lm_local_bpe_1024_seq${SEQ_LEN}_sp5_lr001/expert_checkpoints.pth"
  fi
fi
EXPERT_DIR="${EXPERT_PATH%/*}"

CONF_TAG="${INIT_CONFIDENCE//./}"
GROUP_SUFFIX=""
if [[ -n "${K_TAG}" ]]; then
  GROUP_SUFFIX="_${K_TAG}"
fi

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  if [[ ! -f "${DATA_PATH}" ]]; then
    log_stage "Exporting TinyStories text to ${DATA_PATH}"
    "${PYTHON_BIN}" export_tinystories_text.py \
      +dataset_name=roneneldan/TinyStories \
      +split=train \
      +max_texts=15000 \
      "+output_path=${DATA_PATH}" \
      +streaming=true
  else
    log_stage "Using existing TinyStories text: ${DATA_PATH}"
  fi

  if [[ ! -f "${TOKENIZER_PATH}" ]]; then
    log_stage "Training local BPE tokenizer at ${TOKENIZER_PATH}"
    "${PYTHON_BIN}" train_local_bpe_tokenizer.py \
      "+input_path=${DATA_PATH}" \
      +output_dir=artifacts/tokenizers/tinystories_bpe_1024 \
      +vocab_size=1024
  else
    log_stage "Using existing tokenizer: ${TOKENIZER_PATH}"
  fi

  if [[ ! -f "${EXPERT_PATH}" ]]; then
    log_stage "Computing TM expert trajectory at ${EXPERT_PATH}"
    "${PYTHON_BIN}" -m src.compute_expert_trajectory \
      --config-name "${EXPERT_CONFIG}" \
      "expert.save_dir=${EXPERT_DIR}" \
      "dataloader.num_workers=0" \
      "dataloader.pin_memory=false"
  else
    log_stage "Using existing expert trajectory: ${EXPERT_PATH}"
  fi
fi

for distill_steps in ${DISTILL_STEP_VALUES}; do
  DISTILL_SAVE_DIR="${DISTILL_SAVE_ROOT}_distil${distill_steps}_e${TRAIN_EPOCHS}${GROUP_SUFFIX}"
  EVAL_SAVE_DIR="${EVAL_SAVE_ROOT}_distil${distill_steps}_e${TRAIN_EPOCHS}${GROUP_SUFFIX}"
  RESULTS_DIR="${RESULTS_ROOT}_distil${distill_steps}_e${TRAIN_EPOCHS}${GROUP_SUFFIX}"

  mkdir -p "${RESULTS_DIR}"
  log_stage "Starting sweep for seq_len=${SEQ_LEN}, distillation.n_steps=${distill_steps}, trainer.n_epochs=${TRAIN_EPOCHS}"

  for parameterization in ${PARAMETERIZATIONS}; do
    case "${parameterization}" in
      topk)
        label="topk"
        ;;
      topk_gumbel)
        label="gumbel"
        ;;
      *)
        echo "Unknown parameterization '${parameterization}'. Use topk or topk_gumbel." >&2
        exit 2
        ;;
    esac

    for k in ${K_VALUES}; do
      run_name="tm_${label}_k${k}_n${NSEQ}_conf${CONF_TAG}"
      checkpoint_path="${DISTILL_SAVE_DIR}/${run_name}/full_soft_tokens_best.pth"

      if [[ -f "${checkpoint_path}" && "${RERUN_DISTILL:-0}" != "1" ]]; then
        log_stage "Skipping existing distillation checkpoint: ${checkpoint_path}"
      else
        log_stage "Distilling ${run_name}"
        "${PYTHON_BIN}" distill.py --config-name "${DISTILL_CONFIG}" \
          "distillation.save_dir=${DISTILL_SAVE_DIR}" \
          "distillation.run_name=${run_name}" \
          "distillation.n_steps=${distill_steps}" \
          "distillation.expert_trajectory_path=${EXPERT_PATH}" \
          "distillation.synthetic.num_sequences=${NSEQ}" \
          "distillation.synthetic.init_confidence=${INIT_CONFIDENCE}" \
          "distillation.synthetic.parameterization=${parameterization}" \
          "++distillation.synthetic.topk=${k}" \
          "dataloader.num_workers=0" \
          "dataloader.pin_memory=false" \
          "hydra.run.dir=.hydra/distill_tm_topk_gumbel/${run_name}_distil${distill_steps}_e${TRAIN_EPOCHS}"
      fi

      eval_run_name="eval_${run_name}"
      if [[ -f "${EVAL_SAVE_DIR}/${eval_run_name}/info.log" && "${RERUN_EVAL:-0}" != "1" ]]; then
        log_stage "Skipping existing evaluation log: ${EVAL_SAVE_DIR}/${eval_run_name}/info.log"
      else
        log_stage "Evaluating ${run_name} with downstream LM training"
        train_cmd=(
          "${PYTHON_BIN}" train.py --config-name "${SOFT_TRAIN_CONFIG}"
          "datasets.train.checkpoint_path=${checkpoint_path}"
          "datasets.val.local_text_path=${DATA_PATH}"
          "datasets.test.local_text_path=${DATA_PATH}"
          "model.max_seq_len=${SEQ_LEN}"
          "dataloader.num_workers=0"
          "dataloader.pin_memory=false"
          "trainer.save_dir=${EVAL_SAVE_DIR}"
          "trainer.n_epochs=${TRAIN_EPOCHS}"
          "trainer.epoch_len=${TRAIN_EPOCH_LEN}"
          "optimizer.lr=${TRAIN_LR}"
          "writer.run_name=${eval_run_name}"
          "hydra.run.dir=.hydra/eval_tm_topk_gumbel/${eval_run_name}_distil${distill_steps}_e${TRAIN_EPOCHS}"
        )
        if [[ -n "${TRAIN_LR_STEP_SIZE}" ]]; then
          train_cmd+=("lr_scheduler.step_size=${TRAIN_LR_STEP_SIZE}")
        fi
        if [[ -n "${TRAIN_LR_GAMMA}" ]]; then
          train_cmd+=("lr_scheduler.gamma=${TRAIN_LR_GAMMA}")
        fi
        "${train_cmd[@]}"
      fi
    done
  done

  log_stage "Collecting sweep results"
  "${PYTHON_BIN}" scripts/collect_tm_results.py \
    --distill-dir "${DISTILL_SAVE_DIR}" \
    --eval-dir "${EVAL_SAVE_DIR}" \
    --output "${RESULTS_DIR}/tm_topk_gumbel_results.csv" \
    --markdown "${RESULTS_DIR}/tm_topk_gumbel_results.md" \
    --best-output "${RESULTS_DIR}/best_res.csv"

  echo "Wrote:"
  echo "  ${RESULTS_DIR}/tm_topk_gumbel_results.csv"
  echo "  ${RESULTS_DIR}/tm_topk_gumbel_results.md"
  echo "  ${RESULTS_DIR}/best_res.csv"
done

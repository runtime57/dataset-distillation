#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"
SEQ_LEN=256 \
MODEL_TAG=nl4_do00_fixed15k_expertadam \
DISTILL_CONFIG=distill_soft_tokens_tm_10k_n256_local_bpe_1024_seq256_nl4_do00_fixed15k \
SOFT_TRAIN_CONFIG=tinystories_lm_softdistill_local_bpe_1024_seq256_nl4_do00_fixed15k \
EXPERT_CONFIG=expert_trajectory_local_bpe_1024_seq256_nl4_do00_fixed15k \
EXPERT_PATH=saved/expert_trajectories/tiny_lm_local_bpe_1024_seq256_nl4_do00_fixed15k_adam_sp5_lr001/expert_checkpoints.pth \
DATA_PATH=data/tinystories/tinystories_train_first15k.txt \
TRAIN_EPOCHS=15 \
TRAIN_EPOCH_LEN=64 \
TRAIN_LR_STEP_SIZE=64 \
DISTILL_STEP_VALUES=400 \
K_VALUES="4 8 16 32 64 128 256" \
K_TAG=k4_8_16_32_64_128_256 \
"${script_dir}/run_tm_topk_gumbel_sweep_local.sh"

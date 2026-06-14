# MNIST Dataset Distillation Smoke Results

Date: 2026-06-15

## One-Step, Raw Synthetic Pixels, Noise Init

Run:
`results/mnist_raw_noise_onestep_ssnlish_1000_check/one_step`

Command:
`python3 scripts/mnist_distill_smoke.py --objective one_step --init-mode noise --image-param raw --lr-schedule per_update --synthetic-lr 0.01 --adam-beta1 0.5 --adam-beta2 0.999 --synthetic-lr-scheduler step --step-lr-size 400 --step-lr-gamma 0.5 --iterations 1000 --output-root results/mnist_raw_noise_onestep_ssnlish_1000_check`

Final metrics:

- `best_step = 836`
- `best_train_loss = 0.593015`
- `eval_accuracy_mean = 0.7835`
- `eval_loss_mean = 0.691142`

Notes:

- This is a no-real-init run.
- The key change versus the earlier stuck runs is `--image-param raw`: synthetic tensors are passed to the model unconstrained, matching the original `ssnl/dataset-distillation` style more closely.
- `sigmoid`-parameterized runs stayed near random or weak accuracy, while this raw-pixel run reaches clearly useful MNIST accuracy.

## Same Settings On Other Objectives

All runs below use the same no-real-init setup as the working one-step run:
`--init-mode noise --image-param raw --lr-schedule per_update --synthetic-lr 0.01 --adam-beta1 0.5 --adam-beta2 0.999 --synthetic-lr-scheduler step --step-lr-size 400 --step-lr-gamma 0.5 --iterations 1000`.

| objective | run | best step | best train loss | eval accuracy | eval loss |
| --- | --- | ---: | ---: | ---: | ---: |
| `feature_matching` | `results/mnist_raw_noise_fm_ssnlish_1000_check/feature_matching` | 337 | 0.000503 | 0.1405 | 2.285529 |
| `gradient_matching` | `results/mnist_raw_noise_gm_ssnlish_1000_check/gradient_matching` | 941 | 0.076203 | 0.1330 | 2.296010 |
| `trajectory_matching` | `results/mnist_raw_noise_tm_ssnlish_1000_check/trajectory_matching` | 331 | 0.218052 | 0.1405 | 2.298351 |
| `trajectory_matching`, full synthetic updates | `results/mnist_raw_noise_tm_fullupdates_1000_check/trajectory_matching` | 616 | 0.201343 | 0.1240 | 2.303862 |

Notes:

- FM, GM, and TM all optimize their surrogate losses, but stay near random MNIST eval accuracy in this smoke protocol.
- The fixed TM rerun matches the full `distill_epochs * distill_steps` synthetic update protocol inside the TM loss and saves expert checkpoints at the same interval. This improves trajectory loss slightly, but not eval accuracy.
- This makes the one-step raw-pixel run the first clearly working no-real-init MNIST baseline.
- Next debugging target is the surrogate/evaluation protocol for FM, GM, and TM rather than only pixel parameterization or optimizer scheduling.

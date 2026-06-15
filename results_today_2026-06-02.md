# Dataset Distillation Results, 2026-06-02

Lower PPL is better.

All main numbers below use the same valid LM setup:

- local TinyStories subset: train rows `0..9999`, val rows `10000..12999`, test rows `13000..14999`
- tokenizer: local BPE vocab `1024`
- model: `max_seq_len=256`, `d_model=128`, `n_heads=4`, `n_layers=4`, dropout `0.0`
- synthetic size unless stated otherwise: `n=256` sequences, length `256`
- eval trains from scratch on the synthetic checkpoint and reports real held-out val/test PPL

Important: synthetic training is soft-token training for soft checkpoints. Val/test PPL is measured on real held-out hard-token rows.

## Main Comparable Results

These are the most useful current results to compare.

| rank | method | param | n | k | conf | eval run | best epoch | best val PPL | best test PPL | final epoch | final test PPL | note |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0 | Real baseline | real text | 10k rows | - | - | `real_10krows_seq256_bpe1024` | 3 | 23.310 | 21.777 | 3 | 21.777 | upper target, not distilled |
| 1 | One-step learned LR | fixed top-k | 256 | 16 | 12.5 | `eval8_os_topk16_n256_conf125_lr001` | 7 | 63.283 | 60.677 | 8 | 62.044 | new best synthetic, quick eval8 |
| 2 | One-step learned LR | fixed top-k | 256 | 32 | 12.5 | `eval8_os_topk32_n256_conf125_lr001` | 7 | 63.636 | 61.033 | 8 | 61.749 | quick eval8; worse than k16 |
| 3 | TM | full soft | 256 | - | 12.5 | `eval20_final_tm_n256_conf125_opt01_ob4` | 6 | 64.138 | 61.467 | 20 | 104.729 | best full-soft synthetic |
| 4 | One-step fixed LR | full soft | 256 | - | 12.5 | `eval20_final_os_n256_conf125_lr001_fixed_opt01_ob1` | 6 | 64.149 | 61.470 | 20 | 105.042 | fixed inner lr |
| 5 | GM | full soft | 256 | - | 12.5 | `eval20_final_gm_n256_conf125_opt01_ob4` | 6 | 64.149 | 61.470 | 20 | 104.575 | very close to TM |
| 6 | One-step learned LR | full soft | 256 | - | 12.5 | `eval20_final_os_n256_conf125_lr001_opt01_ob1` | 6 | 64.191 | 61.513 | 20 | 104.611 | learned inner lr |
| 7 | One-step learned LR | fixed top-k | 256 | 8 | 12.5 | `eval20_os_topk8_n256_conf125_lr001` | 6 | 63.980 | 61.922 | 20 | 106.590 | compact, almost full-soft quality |
| 8 | FM | full soft | 256 | - | 12.5 | `eval20_final_fm_n256_conf125_opt10_ob4` | 7 | 77.976 | 74.851 | 20 | 136.303 | weaker objective |

Takeaway: the current best synthetic run is one-step fixed `top-k16`, `test PPL = 60.677` in a quick 8-epoch eval. Real train-10k is still much better at `21.777`, so the remaining gap is large.

## Earlier Same-Day Runs

These were also run today, but before the later `conf=12.5` full-soft sweep. They are useful as baselines, not as the final current setup.

| method | param | n | k | conf | eval run | best epoch | best val PPL | best test PPL | note |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| GM | top-k | 256 | 8 | 5.0 | `eval_gm_topk8_n256_seq256_10krows` | 5 | 69.274 | 66.307 | early sparse baseline |
| GM | top-k | 256 | 32 | 5.0 | `eval_gm_topk32_n256_seq256_10krows` | 5 | 78.101 | 75.324 | early sparse baseline |
| FM | full soft | 256 | - | 5.0 | `eval_fm_full_n256_seq256_10krows` | 5 | 159.012 | 154.182 | early conf=5 |
| TM | full soft | 256 | - | 5.0 | `eval_tm_full_n256_seq256_10krows` | 5 | 190.987 | 185.585 | early conf=5 |
| GM | full soft | 256 | - | 5.0 | `eval_gm_full_n256_seq256_10krows` | 5 | 319.784 | 313.721 | early conf=5 |
| GM | anchors | 256 | - | 5.0 | `eval_gm_anchors64_n256_seq256_10krows` | 5 | 641.822 | 631.608 | weak/failed |

## Current Sparse Probe

| method | param | n | k | conf | status | observed result |
|---|---|---:|---:|---:|---|---|
| One-step learned LR | fixed top-k | 256 | 8 | 12.5 | completed distill + eval | best epoch 6: val `63.980`, test `61.922`; overfits after that |
| One-step learned LR | fixed top-k | 256 | 16 | 12.5 | completed distill + quick eval8 | best epoch 7: val `63.283`, test `60.677`; epoch 8 worsens to test `62.044` |
| One-step learned LR | fixed top-k | 256 | 32 | 12.5 | completed distill + quick eval8 | best epoch 7: val `63.636`, test `61.033`; worse than k16 |
| One-step learned LR | fixed top-k | 256 | 8 | 5.0 | completed distill + quick eval8 | best epoch 6: val `73.044`, test `70.961`; lower confidence was worse |
| One-step learned LR | fixed top-k | 512 | 8 | 12.5 | aborted during distill at step 270/400 | proxy was not clearly better; no eval |

For high-confidence fixed top-k, entropy stays almost zero, so these are basically hard-token-like synthetic sets within a fixed support. Wider support helped up to `k=16`; `k=32` had a better distill proxy late in training but worse eval PPL than `k=16`.

## Rough 2-Epoch Screening Runs

These are only rough screening results with `n=128` and eval for 2 epochs. They are not directly comparable with the 20-epoch final runs.

| method | param | n | conf | eval run | best val PPL | best test PPL |
|---|---|---:|---:|---|---:|---:|
| GM | full soft | 128 | 20.0 | `eval2_screen_gm_n128_conf20_opt01_ob4` | 125.830 | 120.626 |
| One-step | full soft | 128 | 20.0 | `eval2_screen_os_n128_conf20_lr001_opt01_ob1` | 125.830 | 120.626 |
| FM | full soft | 128 | 20.0 | `eval2_screen_fm_n128_conf20_opt10_ob4` | 125.830 | 120.626 |
| TM | full soft | 128 | 20.0 | `eval2_screen_tm_n128_conf20_opt01_ob4` | 125.830 | 120.626 |
| One-step | full soft | 128 | 17.5 | `eval2_screen_os_n128_conf175_lr001_opt01_ob1` | 125.831 | 120.626 |
| One-step | full soft | 128 | 15.0 | `eval2_screen_os_n128_conf15_lr001_opt01_ob1` | 125.842 | 120.638 |
| One-step | full soft | 128 | 12.5 | `eval2_screen_os_n128_conf125_lr001_opt01_ob1` | 125.983 | 120.781 |
| TM | full soft | 128 | 10.0 | `eval2_screen_tm_n128_conf10_opt01_ob4` | 127.804 | 122.628 |
| One-step | full soft | 128 | 10.0 | `eval2_screen_os_n128_conf10_lr001_opt01_ob1` | 127.863 | 122.677 |
| GM | full soft | 128 | 10.0 | `eval2_screen_gm_n128_conf10_opt01_ob4` | 127.886 | 122.713 |
| FM | full soft | 128 | 10.0 | `eval2_screen_fm_n128_conf10_opt10_ob4` | 134.242 | 128.954 |
| FM | full soft | 128 | 5.0 | `eval2_screen_fm_n128_conf5_opt10_ob4` | 367.618 | 356.716 |
| GM | full soft | 128 | 5.0 | `eval2_screen_gm_n128_conf5_opt01_ob4` | 526.045 | 515.720 |
| TM | full soft | 128 | 5.0 | `eval2_screen_tm_n128_conf5_opt01_ob4` | 529.883 | 518.965 |
| One-step | full soft | 128 | 5.0 | `eval2_screen_os_n128_conf5_lr001_opt01_ob1` | 532.919 | 521.907 |

## Recommended Next Experiments

Most promising directions from current evidence:

| priority | experiment | why |
|---:|---|---|
| 1 | Confirm `one-step top-k16 conf=12.5` with a 12/20-epoch eval and maybe another seed | it is the new best, but current number is from quick eval8 |
| 2 | One-step top-k16 with `conf=10.0/15.0` | `conf=5.0` was too soft; k16 may have a nearby confidence optimum |
| 3 | One-step top-k24 or top-k12 | k16 beat k8 and k32, so the sweet spot may be between them |
| 4 | One-step Gumbel top-k32 with `conf=10.0/12.5` | learned/dynamic support may beat fixed top-k if stable |
| 5 | TM full soft with slightly larger `n`, e.g. `n=512`, short eval to epoch 8 | current best full-soft method; more synthetic data may help |

Do not compare against `eval_gm_full_n256_seq256_10krows_e20_bad_seq128`; it is invalid because it used the wrong setup.

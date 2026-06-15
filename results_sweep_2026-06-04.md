# Model-matched n256 result audit, 2026-06-13

This table supersedes the mixed sweep notes from 2026-06-04. It keeps only
results that are comparable to the current dataset-distillation setting.

## Scope

In-scope synthetic rows must satisfy all of the following:

- eval model and distill model configs are both present and identical;
- model is `TinyTransformerLM(vocab_size=1024, max_seq_len=256, d_model=128, n_heads=4, n_layers=4, dim_feedforward=512, dropout=0.0, pad_token_id=0)`;
- distilled dataset budget is `num_sequences=256`, `sequence_length=256`;
- metric is held-out hard-token val/test PPL from local `data/tinystories_train_15k.txt`.

Hydra values like `sequence_length: ${model.max_seq_len}` were resolved through
the saved distillation model config before filtering.

## Audit Summary

| bucket | count | decision |
|---|---:|---|
| synthetic, model-matched, canonical `n=256/seq=256` | 95 | main comparable pool |
| real baselines, canonical model | 3 | baseline/control only |
| model-matched but `n=512` | 4 | excluded from current `n=256` table |
| eval/distill matched each other but non-canonical model | 8 | excluded |
| teacher/hybrid/real-first256 diagnostics | 10 | excluded from dataset-distillation claims |
| old eval rows without available distill config | 15 | excluded; model match cannot be verified |

No available saved eval row had eval/distill model configs that directly
disagreed with each other. The problematic high-ranking rows were mostly
matched to a different architecture (`2` layers, `dropout=0.1`, sometimes
`max_seq_len=128/512`) or used a different data budget (`n=512`).

## Current Comparable Targets

| run | data | best epoch | val PPL | test PPL | note |
|---|---|---:|---:|---:|---|
| `real10k_seq256_4l_adamw_lr1e3_wd001_e5` | real 10k rows | 5 | 11.278 | 10.567 | canonical real baseline |
| `real_10krows_seq256_bpe1024` | real 10k rows | 3 | 23.310 | 21.777 | older optimizer baseline |
| `eval_ngram_herding256_ub1_bb4096_hard_local_e12` | distilled hard-token coreset, `n=256` | 4 | 43.156 | 41.168 | current best valid teacher-free `n=256` result |
| `real256_seq256_4l_adamw_lr1e3_wd001_e8_epochlen64` | real first 256 rows | 3 | 67.023 | 70.137 | same data budget as synthetic, hard real tokens |
| `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr35e4_wd11_gamma092_e8` | distilled `n=256`, one-step gumbel top-k64 | 4 | 45.679 | 43.462 | best valid learned-soft-token result from the audited sweep |

## 2026-06-15 Teacher-Free Hard-Token Coreset Probes

These rows are pure dataset-distillation/coreset probes: no teacher targets, no
hybrid relabeling, same canonical model, `n=256`, `max_seq_len=256`, and the
same held-out local val/test split. Checkpoints store only hard token ids.

| test PPL | val PPL | epoch | run | selection | eval |
|---:|---:|---:|---|---|---|
| 41.168 | 43.156 | 4 | `eval_ngram_herding256_ub1_bb4096_hard_local_e12` | unigram+hashed-bigram herding, `bigram_bins=4096`, `unigram_weight=1.0`, `bigram_weight=1.0` | lr=0.0035, wd=1.1, gamma=0.92 |
| 41.235 | 43.241 | 4 | `eval_ngram_herding256_ub1_bb4096_ls64_hard_local_e6` | same herding plus feature-objective swap local search | lr=0.0035, wd=1.1, gamma=0.92 |
| 41.268 | 43.061 | 5 | `eval_ngram_herding256_ub1_bw11_bb4096_hard_local_e6` | unigram+hashed-bigram herding, `bigram_weight=1.1`; best val in this probe set | lr=0.0035, wd=1.1, gamma=0.92 |
| 41.525 | 43.422 | 4 | `eval_ngram_herding256_ub1_bw20_bb4096_hard_local_e4` | unigram+hashed-bigram herding, `bigram_weight=2.0` | lr=0.0035, wd=1.1, gamma=0.92 |
| 41.518 | 43.303 | 4 | `eval_ngram_herding256_ub1_bw15_bb4096_hard_local_e6` | unigram+hashed-bigram herding, `bigram_weight=1.5` | lr=0.0035, wd=1.1, gamma=0.92 |
| 41.820 | 43.524 | 4 | `eval_ngram_herding256_ub1_bw08_bb4096_hard_local_e6` | unigram+hashed-bigram herding, `bigram_weight=0.8` | lr=0.0035, wd=1.1, gamma=0.92 |
| 41.902 | 43.665 | 4 | `eval_ngram_herding256_ub1_bw1_bb8192_hard_local_e4` | unigram+hashed-bigram herding, `bigram_bins=8192` | lr=0.0035, wd=1.1, gamma=0.92 |
| 42.274 | 44.242 | 4 | `eval_ngram_posbigram_herding256_b4096_p4096_pb4096_hard_local_e4` | adds position unigram+bigram hashes at weight 1.0 | lr=0.0035, wd=1.1, gamma=0.92 |
| 42.351 | 44.248 | 4 | `eval_ngram_tri_herding256_ub1_bw1_tw05_bb4096_tb4096_hard_local_e6` | adds hashed trigrams with `trigram_weight=0.5` | lr=0.0035, wd=1.1, gamma=0.92 |
| 42.414 | 44.411 | 5 | `eval_ngram_pos_herding256_b4096_p4096_pw01_hard_local_e6` | adds position unigram hashes with `position_weight=0.1` | lr=0.0035, wd=1.1, gamma=0.92 |
| 42.605 | 44.546 | 4 | `eval_ngram_herding256_ub1_bw05_bb4096_hard_local_e4` | unigram+hashed-bigram herding, `bigram_weight=0.5` | lr=0.0035, wd=1.1, gamma=0.92 |
| 42.903 | 44.627 | 4 | `eval_ngram_pos_herding256_b4096_p4096_hard_local_e4` | adds position unigram hashes at weight 1.0 | lr=0.0035, wd=1.1, gamma=0.92 |
| 43.722 | 46.201 | 4 | `eval_random256_seed1_hard_local_e4` | random 256 hard-token sequences control | lr=0.0035, wd=1.1, gamma=0.92 |
| 54.177 | 53.548 | 4 | `eval_rare_top256_hard_local_e4` | rare-token top-256 control | lr=0.0035, wd=1.1, gamma=0.92 |

Train-recipe probes on the best coreset did not improve test PPL: `lr=0.0040,
wd=1.1` reached `val=43.398/test=41.465`, `lr=0.0030, wd=1.1` reached
`val=43.837/test=41.917`, stronger weight decay (`1.5` or `2.0`) was worse, and
batch sizes `8`/`32` were worse than the canonical batch size `16`.

## In-Scope Synthetic Results

Sorted by test PPL. This is the useful band of the 95 valid canonical rows;
the remaining valid rows are lower-quality smoke/random/concepts runs and do
not change the current best.

| test PPL | val PPL | epoch | run | synthetic | eval |
|---:|---:|---:|---|---|---|
| 43.462 | 45.679 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr35e4_wd11_gamma092_e8` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.92 |
| 43.486 | 45.695 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr35e4_wd11_gamma09_e8` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.9 |
| 43.489 | 45.702 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr35e4_wd11_gamma093_e8` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.93 |
| 43.524 | 45.724 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr35e4_wd11_e6` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.95 |
| 43.558 | 45.794 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_hard_lr35e4_wd11_gamma092_e8` | one_step, topk_gumbel64 hard labels, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.92 |
| 43.558 | 45.744 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr35e4_wd115_gamma092_e8` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.15, gamma=0.92 |
| 43.590 | 45.832 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_temp08_lr35e4_wd11_gamma092_e8` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.92 |
| 43.593 | 45.814 | 4 | `eval_os_gumbel32_n256_conf125_lr001_detckpt_lr35e4_wd11_e6` | one_step, topk_gumbel32, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.95 |
| 43.620 | 45.819 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr35e4_wd11_gamma085_e8` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.85 |
| 43.672 | 45.845 | 4 | `eval_ms4_gumbel64_n256_conf100_fixedlr03_lr35e4_wd11_gamma092_e8` | multi_step, topk_gumbel64, conf=10.0, steps=80 | lr=0.0035, wd=1.1, gamma=0.92 |
| 43.674 | 45.902 | 4 | `eval_os_gumbel16_n256_conf125_lr001_detckpt_lr35e4_wd11_e6` | one_step, topk_gumbel16, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.95 |
| 43.679 | 45.851 | 4 | `eval_os_gumbel128_n256_conf125_lr001_detckpt_lr35e4_wd11_e6` | one_step, topk_gumbel128, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.95 |
| 43.696 | 45.969 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr35e4_wd105_gamma092_e8` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.05, gamma=0.92 |
| 43.710 | 45.933 | 4 | `eval_os_gumbel64_n256_conf125_lr001_s800_cap2_lr35e4_wd11_e6` | one_step, topk_gumbel64, conf=12.5, steps=800 | lr=0.0035, wd=1.1, gamma=0.95 |
| 43.739 | 45.837 | 9 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr35e4_wd11_gamma095916_ep32_e12` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.9591663046625439, ep_len=32 |
| 43.796 | 45.973 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_temp12_lr35e4_wd11_gamma092_e8` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.92 |
| 43.811 | 46.103 | 4 | `eval_distill_os_topk16_n256_conf125_lr001_ob4_lr35e4_wd11_e6` | one_step, topk16, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.95 |
| 43.811 | 46.103 | 4 | `eval_os_topk16_n256_lr35e4_wd11_e6` | one_step, topk16, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.95 |
| 43.816 | 46.022 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr35e4_wd12_e6` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.2, gamma=0.95 |
| 43.835 | 46.100 | 4 | `eval_os_gumbel64_n256_conf100_lr001_detckpt_lr35e4_wd11_e6` | one_step, topk_gumbel64, conf=10.0, steps=400 | lr=0.0035, wd=1.1, gamma=0.95 |
| 43.843 | 46.051 | 4 | `eval_os_topk16_n256_lr35e4_wd115_e6` | one_step, topk16, conf=12.5, steps=400 | lr=0.0035, wd=1.15, gamma=0.95 |
| 43.869 | 45.865 | 8 | `eval_os_topk16_n256_lr35e4_wd11_epochlen32_e10` | one_step, topk16, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.95, ep_len=32 |
| 43.882 | 46.086 | 4 | `eval_gm_full_n256_lr35e4_wd11_e6` | gradient_matching, full, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.95 |
| 43.885 | 46.088 | 4 | `eval_os_full_fixed_n256_lr35e4_wd11_e6` | one_step, full, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.95 |
| 43.891 | 46.267 | 4 | `eval_os_topk16_n256_lr35e4_wd10_e6` | one_step, topk16, conf=12.5, steps=400 | lr=0.0035, wd=1, gamma=0.95 |
| 43.903 | 46.087 | 4 | `eval_tm_full_n256_lr35e4_wd11_e6` | trajectory_matching, full, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.95 |
| 43.905 | 46.146 | 4 | `eval_tm_full_n256_lr35e4_wd10_e6` | trajectory_matching, full, conf=12.5, steps=400 | lr=0.0035, wd=1, gamma=0.95 |
| 43.905 | 46.116 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr35e4_wd11_gamma092_beta2_095_e8` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.92 |
| 43.941 | 46.174 | 4 | `eval_os_full_fixed_n256_lr35e4_wd10_e6` | one_step, full, conf=12.5, steps=400 | lr=0.0035, wd=1, gamma=0.95 |
| 43.961 | 46.321 | 4 | `eval_tm_full_n256_lr3e3_wd10_e5` | trajectory_matching, full, conf=12.5, steps=400 | lr=0.003, wd=1, gamma=0.95 |
| 44.008 | 46.223 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr34e4_wd11_gamma092_e8` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0034, wd=1.1, gamma=0.92 |
| 44.015 | 46.322 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr3e3_wd11_e7` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.003, wd=1.1, gamma=0.95 |
| 44.069 | 46.348 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_seed3_lr35e4_wd11_gamma092_e8` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.92, seed=3 |
| 44.134 | 46.346 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr36e4_wd11_gamma092_e8` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0036, wd=1.1, gamma=0.92 |
| 44.140 | 46.416 | 4 | `eval_tm_full_n256_lr3e3_wd11_e6` | trajectory_matching, full, conf=12.5, steps=400 | lr=0.003, wd=1.1, gamma=0.95 |
| 44.185 | 46.475 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr35e4_wd10_e6` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1, gamma=0.95 |
| 44.271 | 46.513 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_seed0_lr35e4_wd11_gamma092_e8` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.92, seed=0 |
| 44.277 | 46.510 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr4e3_wd11_e6` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.004, wd=1.1, gamma=0.95 |
| 44.301 | 46.525 | 5 | `eval_gumbel64_realinit_conf10_lr001_s400_lr35e4_wd11_gamma092_e8` | one_step, topk_gumbel64, conf=10.0, steps=400 | lr=0.0035, wd=1.1, gamma=0.92 |
| 44.421 | 46.653 | 4 | `eval_tm_full_n256_lr3e3_wd12_e6` | trajectory_matching, full, conf=12.5, steps=400 | lr=0.003, wd=1.2, gamma=0.95 |
| 44.474 | 46.698 | 4 | `eval_tm_full_n256_lr4e3_wd10_e6` | trajectory_matching, full, conf=12.5, steps=400 | lr=0.004, wd=1, gamma=0.95 |
| 44.681 | 46.935 | 3 | `eval_tm_full_n256_lr3e3_wd08_full_e5` | trajectory_matching, full, conf=12.5, steps=400 | lr=0.003, wd=0.8, gamma=0.95 |
| 44.684 | 46.991 | 6 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_lr35e4_wd11_gamma092_bs32_ep32_e8` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.92, ep_len=32 |
| 44.869 | 47.133 | 3 | `eval_tm_full_n256_lr3e3_wd07_e5` | trajectory_matching, full, conf=12.5, steps=400 | lr=0.003, wd=0.7, gamma=0.95 |
| 44.885 | 47.109 | 4 | `eval_os_gumbel64_n256_conf125_lr001_detckpt_seed2_lr35e4_wd11_e6` | one_step, topk_gumbel64, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.95, seed=2 |
| 45.140 | 46.944 | 4 | `eval_os_topk8_conf125_n256_lr35e4_wd11_e6` | one_step, topk8, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.95 |
| 45.151 | 47.453 | 3 | `eval_tm_full_n256_lr3e3_wd06_e4` | trajectory_matching, full, conf=12.5, steps=400 | lr=0.003, wd=0.6, gamma=0.95 |
| 45.433 | 47.443 | 4 | `eval_os_topk32_n256_lr35e4_wd11_e6` | one_step, topk32, conf=12.5, steps=400 | lr=0.0035, wd=1.1, gamma=0.95 |
| 45.803 | 48.165 | 4 | `eval_gumbel64_realinit_conf8_lr001_s400_lr35e4_wd11_gamma092_e8` | one_step, topk_gumbel64, conf=8.0, steps=400 | lr=0.0035, wd=1.1, gamma=0.92 |
| 45.880 | 48.254 | 4 | `eval_os_gumbel64_n256_conf80_lr001_detckpt_lr35e4_wd11_e6` | one_step, topk_gumbel64, conf=8.0, steps=400 | lr=0.0035, wd=1.1, gamma=0.95 |
| 46.573 | 48.958 | 2 | `eval_tm_full_n256_lr3e3_wd04_e4` | trajectory_matching, full, conf=12.5, steps=400 | lr=0.003, wd=0.4, gamma=0.95 |
| 46.736 | 49.213 | 2 | `eval_tm_full_n256_lr3e3_wd02_e4` | trajectory_matching, full, conf=12.5, steps=400 | lr=0.003, wd=0.2, gamma=0.95 |
| 47.028 | 49.557 | 2 | `eval_tm_full_n256_lr3e3_wd01_e4` | trajectory_matching, full, conf=12.5, steps=400 | lr=0.003, wd=0.1, gamma=0.95 |
| 47.233 | 49.785 | 2 | `eval_tm_full_n256_lr3e3_wd005_e4` | trajectory_matching, full, conf=12.5, steps=400 | lr=0.003, wd=0.05, gamma=0.95 |
| 47.437 | 50.010 | 2 | `eval_tm_full_n256_lr3e3_wd001_e4` | trajectory_matching, full, conf=12.5, steps=400 | lr=0.003, wd=0.01, gamma=0.95 |

## Excluded: Model-Matched But Outside Current Budget

These rows are not wrong internally, but they use `n=512`, so they are not part
of the current `n=256` comparison.

| test PPL | val PPL | epoch | run | reason |
|---:|---:|---:|---|---|
| 39.130 | 41.404 | 5 | `eval8_os_topk16_n512_adamw_lr1e3_wd001` | `n=512` |
| 39.148 | 41.426 | 5 | `eval6_os_topk16_n512_adamw_lr1e3_wd0` | `n=512` |
| 42.137 | 44.429 | 8 | `eval10_os_topk16_n512_adamw_lr5e4_wd001` | `n=512` |
| 44.347 | 46.715 | 13 | `eval8_os_topk16_n512_conf125_lr001` | `n=512` |

## Excluded: Non-Canonical Model

These rows had eval/distill configs that matched each other, but the matched
model was not the current canonical 4-layer, seq256, dropout0 setup.

| test PPL | val PPL | epoch | run | model reason |
|---:|---:|---:|---|---|
| 37.789 | 40.045 | 7 | `d_seq256_2l_dropout01_os_topk16_n512_conf125_lr1e3_wd001_gamma095_e8` | `n_layers=2`, `dropout=0.1`, `n=512` |
| 37.995 | 39.836 | 8 | `g_seq512_2l_dropout01_os_topk16_n256_conf125_lr35e4_wd11_gamma092_e8` | `max_seq_len=512`, `n_layers=2`, `dropout=0.1` |
| 39.701 | 41.567 | 4 | `g_seq512_2l_dropout01_os_topk16_n256_conf125_lr35e4_wd11_gamma092_e14` | `max_seq_len=512`, `n_layers=2`, `dropout=0.1` |
| 40.803 | 42.832 | 5 | `g_seq512_2l_dropout01_os_topk16_n256_conf125_lr1e3_wd001_gamma095_e8` | `max_seq_len=512`, `n_layers=2`, `dropout=0.1` |
| 40.854 | 43.164 | 8 | `a_seq128_2l_dropout01_os_gumbel64_n512_conf125_lr35e4_wd11_gamma092_e8` | `max_seq_len=128`, `n_layers=2`, `dropout=0.1`, `n=512` |
| 45.914 | 47.523 | 4 | `b_seq256_2l_dropout01_os_gumbel64_n256_conf125_lr35e4_wd11_gamma092_e8` | `n_layers=2`, `dropout=0.1` |
| 49.245 | 51.497 | 4 | `e_seq256_2l_dropout01_os_topk16_n256_conf125_lr1e3_wd001_gamma095_e8` | `n_layers=2`, `dropout=0.1` |
| 747.304 | 753.726 | 1 | `smoke_matched_a_eval` | smoke, `max_seq_len=128`, `n_layers=2`, `dropout=0.1`, `n=512` |

## Excluded: Diagnostics Only

These are useful sanity checks, but they are not distilled-dataset results in
the current setting.

| test PPL | val PPL | epoch | run | reason |
|---:|---:|---:|---|---|
| 10.567 | 11.278 | 5 | `teacher_real10k_local_seq256_4l_lr1e3_wd001_e5_full` | duplicate teacher/real baseline |
| 23.526 | 24.938 | 34 | `eval_hybrid_teacher_on_distilled_gumbel64_best_hardinputs_lr35e4_wd11_gamma092_e20` | hybrid teacher/real-distill control |
| 29.562 | 29.155 | 20 | `eval_teacher_soft_first256_temp1_lr35e4_wd11_gamma092_e20` | teacher-soft first-256 control |
| 30.280 | 29.783 | 19 | `eval_teacher_soft_first256_temp1_hard010_lr35e4_wd11_gamma092_e20` | teacher-soft first-256 control |
| 34.879 | 36.598 | 5 | `teacher_real10k_local_seq256_4l_lr1e3_wd001_e5` | partial teacher/real control |
| 35.152 | 34.512 | 8 | `eval_teacher_soft_first256_temp1_lr35e4_wd11_gamma092_e8` | teacher-soft first-256 control |
| 39.995 | 41.766 | 5 | `teacher_real10k_seq256_4l_lr1e3_wd001_e5` | partial teacher/real control |
| 59.821 | 57.249 | 4 | `eval_msadamw4_gumbel64_n256_conf125_lr35e4_wd11_opt1e3_s160_rerun_lr35e4_wd11_gamma092_e8` | later experimental diagnostic; weights deleted |
| 59.891 | 57.317 | 4 | `eval_decoupled_gumbel64_n256_s80_lr35e4_wd11_gamma092_e8` | later experimental diagnostic; weights deleted |
| 70.137 | 67.023 | 3 | `hard_real_first256_lr1e3_wd001_gamma095_e20` | hard real first-256 control |

## Excluded: Cannot Verify Distill Model

The 15 old `saved/eval_sweep_full_soft/eval2_screen_*` rows are excluded from
the comparable table because their original distillation directories/configs
were already cleaned. Without the distill-side model config, model match cannot
be audited.

## Reproduction Configs

Canonical configs retained for future runs:

- `src/configs/distill_soft_tokens_os_gumbel64_matched_local_bpe_1024.yaml`
- `src/configs/tinystories_lm_softdistill_matched_local_bpe_1024.yaml`
- `src/configs/distill_soft_tokens_one_step_10k_seq256_gumbel32_n256_local_bpe_1024.yaml`
- `src/configs/distill_soft_tokens_one_step_10k_seq256_topk8_n256_local_bpe_1024.yaml`
- `src/configs/distill_soft_tokens_tm_10k_seq256_gumbel32_n256_local_bpe_1024.yaml`
- `src/configs/distill_soft_tokens_tm_10k_seq256_topk8_n256_local_bpe_1024.yaml`

Example command pair for the current best family:

```bash
python3 distill.py -cn distill_soft_tokens_os_gumbel64_matched_local_bpe_1024
python3 train.py -cn tinystories_lm_softdistill_matched_local_bpe_1024 \
  datasets.train.checkpoint_path=saved/distillation/<run>/full_soft_tokens_best.pth
```

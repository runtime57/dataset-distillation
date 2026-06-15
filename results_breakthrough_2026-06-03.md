# Breakthrough results, 2026-06-03

Setup for all evals below:
- dataset split: local `data/tinystories_train_15k.txt`, train rows 0..9999, val rows 10000..12999, test rows 13000..14999
- model: TinyTransformerLM, `max_seq_len=256`, `d_model=128`, `n_layers=4`, `n_heads=4`, `dropout=0.0`
- main synthetic checkpoint: `saved/distillation_sweep_breakthrough/os_topk16_n512_conf125_lr001/full_soft_tokens_best.pth`
- main synthetic data: one-step, fixed top-k16, `num_sequences=512`, `init_confidence=12.5`

## Best result

| rank | eval run | optimizer | best epoch | val PPL | test PPL | notes |
|---:|---|---|---:|---:|---:|---|
| 1 | `eval8_os_topk16_n512_adamw_lr1e3_wd001` | AdamW lr `1e-3`, wd `0.01` | 5 | 41.404 | 39.130 | best overall; overfits after epoch 5 |
| 2 | `eval6_os_topk16_n512_adamw_lr1e3_wd0` | AdamW lr `1e-3`, wd `0.0` | 5 | 41.426 | 39.148 | essentially same, slightly worse |
| 3 | `eval10_os_topk16_n512_adamw_lr5e4_wd001` | AdamW lr `5e-4`, wd `0.01` | 8 | 44.429 | 42.137 | slower and worse; stopped after it lagged |
| 4 | `eval8_os_topk16_n512_conf125_lr001` | AdamW lr `3e-4`, wd `0.01` | 13 | 46.715 | 44.347 | old eval lr; clear plateau/overfit after epoch 13 |
| 5 | `eval8_os_topk16_n256_adamw_lr1e3_wd001` | AdamW lr `1e-3`, wd `0.01` | 3 | 54.057 | 51.619 | n=256 comparison; overfits after epoch 3 |

## Distill run

| run | objective | n | top-k | steps | best outer loss | best step | final inner lr |
|---|---|---:|---:|---:|---:|---:|---:|
| `os_topk16_n512_conf125_lr001` | one-step | 512 | 16 | 400 | 6.38185 | 387 | 1.89877 |

## Reproduce best eval

```bash
HYDRA_FULL_ERROR=1 python3 train.py --config-name tinystories_lm_softdistill_local_bpe_1024 \
  datasets.train.checkpoint_path=saved/distillation_sweep_breakthrough/os_topk16_n512_conf125_lr001/full_soft_tokens_best.pth \
  datasets.val.local_text_path=data/tinystories_train_15k.txt \
  datasets.test.local_text_path=data/tinystories_train_15k.txt \
  model.max_seq_len=256 model.d_model=128 model.n_heads=4 model.n_layers=4 model.dim_feedforward=512 model.dropout=0.0 \
  dataloader.batch_size=16 \
  optimizer.lr=1e-3 optimizer.weight_decay=0.01 \
  trainer.n_epochs=8 trainer.epoch_len=64 trainer.log_step=64 trainer.early_stop=3 trainer.seed=1 trainer.override=true \
  trainer.save_dir=saved/eval_sweep_breakthrough_lr \
  writer.run_name=eval8_os_topk16_n512_adamw_lr1e3_wd001 \
  hydra.run.dir=.hydra/eval_breakthrough_lr/eval8_os_topk16_n512_adamw_lr1e3_wd001
```

## Notes

- The 20-40 target was reached on test PPL: best `39.130`.
- `lr=1e-3` is materially better than the previous eval lr `3e-4`; the synthetic checkpoint itself did not change between these evals.
- Best eval overfits fast after epoch 5: epoch 6 test PPL `39.472`, epoch 7 `40.954`, epoch 8 `44.590`.
- n=256 top-k16 with the same eval hyperparams reaches best test PPL `51.619`, so the n=512 jump is real; old n=256 default-lr eval was test PPL `60.677`.
- The plain resume path was fixed in `src/trainer/base_trainer.py` for PyTorch 2.6+ by loading full local checkpoints with `weights_only=False`.

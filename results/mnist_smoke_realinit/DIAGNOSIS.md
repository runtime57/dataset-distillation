# MNIST Smoke Diagnosis

The first MNIST smoke run in `results/mnist_smoke` reached about `0.12`
accuracy, which is random-baseline behavior for 10 classes. The run still
verified that all four objective paths execute, but it did not verify useful
distillation quality.

The cause was the smoke configuration, not the MNIST data path or LeNet:

- LeNet trained on 2048 real MNIST examples reaches about `0.71` accuracy after
  3 real-data epochs in the same harness.
- The initial synthetic images in the first run were nearly flat gray noise:
  `sigmoid(0.1 * N(0, 1))`, with pixel std around `0.025`.
- The distilled learning rate was initialized to `0.02`; in this CPU-sized
  smoke setting, one real image per class already needs about `0.1` to get
  clearly above random after 10 synthetic steps and 3 epochs.
- With only 20 outer iterations, the synthetic images and distilled learning
  rates barely moved.

This directory reruns the four objectives with a diagnostic setting:

```bash
python3 scripts/mnist_distill_smoke.py \
  --objective <objective> \
  --iterations 100 \
  --distill-steps 10 \
  --distill-epochs 3 \
  --train-size 2048 \
  --test-size 2000 \
  --batch-size 128 \
  --output-root results/mnist_smoke_realinit \
  --init-mode real \
  --distilled-lr-init 0.1
```

The results are above random baseline, confirming that the objective code paths
produce a training signal in the MNIST harness. This is still a CPU smoke test,
not a paper-scale reproduction.

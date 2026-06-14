# Real-Image Baseline Comparison

The `real-init` diagnostic run proves that the MNIST harness has a learning
signal, but it does not prove useful dataset distillation quality. A simple
real-image baseline is competitive or better.

All rows below use the same LeNet evaluation protocol, the same 2000-image test
subset, 10 synthetic steps, 3 epochs over the small training set, and 5 random
model initializations.

## Distilled Checkpoints

| objective | eval_accuracy_mean | eval_accuracy_std |
| --- | ---: | ---: |
| feature_matching | 0.3604 | 0.0701 |
| gradient_matching | 0.4030 | 0.1072 |
| trajectory_matching | 0.4040 | 0.0980 |
| one_step | 0.3987 | 0.0845 |

## Real-Image Baselines

`10_total` repeats the same one real MNIST image per class at every synthetic
step. `100_total` uses 10 real MNIST images per class, one per class at each of
the 10 synthetic steps.

| baseline | lr | eval_accuracy_mean | eval_accuracy_std |
| --- | ---: | ---: | ---: |
| 10_total | 0.02 | 0.1399 | 0.0310 |
| 10_total | 0.05 | 0.2895 | 0.0626 |
| 10_total | 0.10 | 0.4724 | 0.0675 |
| 10_total | 0.20 | 0.3594 | 0.1265 |
| 10_total | 0.30 | 0.1925 | 0.0774 |
| 100_total | 0.02 | 0.1608 | 0.0437 |
| 100_total | 0.05 | 0.2419 | 0.0741 |
| 100_total | 0.10 | 0.3498 | 0.0984 |
| 100_total | 0.20 | 0.4618 | 0.0852 |
| 100_total | 0.30 | 0.2661 | 0.1625 |

Conclusion: these short CPU runs are diagnostic only. They show that the code
paths execute and receive gradients, but they do not yet beat a tuned real-image
baseline. Beating that baseline likely requires a closer reproduction of the
paper setup: many more outer iterations, learned LR schedule per epoch/step,
better initialization and/or random-initialization batching, and evaluation over
more held-out initializations.

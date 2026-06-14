# MNIST GM After `sample_indices` Fix

Command used random/uniform synthetic initialization, no real-init, 1000 GM
iterations, 10 synthetic steps x 10 classes, 3 epochs, and 5 eval seeds.

Result:

- `eval_accuracy_mean = 0.11552`
- `eval_accuracy_std = 0.01899`

This remains random-baseline behavior. The
`BaseSyntheticTokenDataset.sample_indices` fix does not directly affect this
standalone MNIST harness; it affects the text synthetic-token runner.

# Text Init Matrix, 2026-06-16

Canonical setup: TinyTransformerLM 4-layer, local BPE-1024, `n=256`, `max_seq_len=256`. `topk` means topk-gumbel64 soft-forward; `argmax` means the same parameterization with `hard_forward=true`.

`real-many` uses all 10k train sequences, about 39-40 source texts per synthetic sequence. `real-many-4` caps the mixture at 1024 source sequences, i.e. about 4 source texts per synthetic sequence.

| objective | init | param | val PPL | test PPL | epoch | best inner | entropy |
|---|---|---|---:|---:|---:|---:|---:|
| gm | real | topk | 45.695 | 43.482 | 4 | 0.416994 | 0.003 |
| gm | real | argmax | 45.794 | 43.558 | 4 | 0.417013 | 0.000 |
| gm | noise | topk | 789.552 | 780.794 | 2 | 0.890446 | 4.157 |
| gm | noise | argmax | 526.331 | 512.059 | 3 | 0.440493 | 0.000 |
| gm | real-many | topk | 311.603 | 299.918 | 6 | 0.500923 | 3.496 |
| gm | real-many | argmax | 1333.973 | 1284.501 | 1 | 0.475723 | 0.000 |
| fm | real | topk | 45.684 | 43.469 | 4 | 1.377552 | 0.004 |
| fm | real | argmax | 45.794 | 43.558 | 4 | 1.377668 | 0.000 |
| fm | noise | topk | 1741.165 | 1741.066 | 2 | 1.813266 | 4.157 |
| fm | noise | argmax | 968.455 | 955.870 | 1 | 1.517193 | 0.000 |
| fm | real-many | topk | 317.878 | 308.102 | 5 | 2.417480 | 3.497 |
| fm | real-many | argmax | 1008.659 | 982.594 | 4 | 1.901592 | 0.000 |
| tm | real | topk | pending | pending | pending | pending | pending |
| tm | real | argmax | pending | pending | pending | pending | pending |
| tm | noise | topk | pending | pending | pending | pending | pending |
| tm | noise | argmax | pending | pending | pending | pending | pending |
| tm | real-many | topk | pending | pending | pending | pending | pending |
| tm | real-many | argmax | pending | pending | pending | pending | pending |
| os | real | topk | pending | pending | pending | pending | pending |
| os | real | argmax | pending | pending | pending | pending | pending |
| os | noise | topk | pending | pending | pending | pending | pending |
| os | noise | argmax | pending | pending | pending | pending | pending |
| os | real-many | topk | pending | pending | pending | pending | pending |
| os | real-many | argmax | pending | pending | pending | pending | pending |
| fm | real-many-4 | topk | 75.979 | 73.140 | 3 | 1.394969 | 1.379 |
| fm | real-many-4 | argmax | 164.597 | 159.187 | 4 | 1.417828 | 0.000 |

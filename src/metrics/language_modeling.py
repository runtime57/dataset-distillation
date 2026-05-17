import math

import torch
from torch.nn import functional as F

from src.metrics.base_metric import BaseMetric


class PerplexityMetric(BaseMetric):
    """
    Batch-level perplexity for causal language modeling.
    """

    def __init__(self, ignore_index=-100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor | None = None,
        target_probs: torch.Tensor | None = None,
        **kwargs,
    ):
        shift_logits = logits[:, :-1].contiguous()

        if target_probs is not None:
            shifted_targets = target_probs[:, 1:].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            loss = -(shifted_targets * log_probs).sum(dim=-1).mean()
            return math.exp(min(loss.item(), 20.0))

        if labels is None:
            raise ValueError(
                "PerplexityMetric expects labels for hard-target evaluation "
                "or target_probs for soft-target evaluation."
            )

        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
        )
        return math.exp(min(loss.item(), 20.0))

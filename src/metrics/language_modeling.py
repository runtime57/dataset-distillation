import math

import torch
from torch.nn import functional as F

from src.metrics.base_metric import BaseMetric


def _token_count(values, attention_mask):
    if attention_mask is None:
        return values.new_tensor(values.numel())
    mask = attention_mask[:, 1:].to(device=values.device, dtype=values.dtype)
    return mask.sum().clamp_min(1.0)


def _masked_mean(values, attention_mask):
    count = _token_count(values, attention_mask)
    if attention_mask is None:
        return values.mean(), count
    mask = attention_mask[:, 1:].to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / count, count


class PerplexityMetric(BaseMetric):
    """
    Token-weighted negative log-likelihood for causal language modeling.
    MetricTracker applies exp() to the aggregated mean NLL.
    """

    def __init__(self, ignore_index=-100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index
        self.transform = lambda nll: math.exp(min(float(nll), 20.0))

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor | None = None,
        target_probs: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        shift_logits = logits[:, :-1].contiguous()

        if target_probs is not None:
            shifted_targets = target_probs[:, 1:].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            per_token_loss = -(shifted_targets * log_probs).sum(dim=-1)
            loss, count = _masked_mean(per_token_loss, attention_mask)
            return loss.item(), count.item()

        if labels is None:
            raise ValueError(
                "PerplexityMetric expects labels for hard-target evaluation "
                "or target_probs for soft-target evaluation."
            )

        shift_labels = labels[:, 1:].contiguous()
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction="none",
        ).view_as(shift_labels)
        valid_mask = (shift_labels != self.ignore_index).to(per_token_loss.dtype)
        if attention_mask is not None:
            valid_mask = valid_mask * attention_mask[:, 1:].to(
                device=per_token_loss.device,
                dtype=per_token_loss.dtype,
            )
        count = valid_mask.sum().clamp_min(1.0)
        loss = (per_token_loss * valid_mask).sum() / count
        return loss.item(), count.item()

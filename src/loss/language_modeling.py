import torch
from torch import nn
from torch.nn import functional as F


def _masked_mean(values, attention_mask):
    if attention_mask is None:
        return values.mean()
    mask = attention_mask[:, 1:].to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class LanguageModelingLoss(nn.Module):
    """
    Next-token cross-entropy for causal language modeling.
    """

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor | None = None,
        target_probs: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **batch,
    ):
        shift_logits = logits[:, :-1].contiguous()

        if target_probs is not None:
            shifted_targets = target_probs[:, 1:].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            per_token_loss = -(shifted_targets * log_probs).sum(dim=-1)
            loss = _masked_mean(per_token_loss, attention_mask)
            return {"loss": loss}

        if labels is None:
            raise ValueError(
                "LanguageModelingLoss expects labels for hard-target training "
                "or target_probs for soft-target training."
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
        loss = (per_token_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        return {"loss": loss}

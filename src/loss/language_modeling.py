import torch
from torch import nn
from torch.nn import functional as F


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
        **batch,
    ):
        shift_logits = logits[:, :-1].contiguous()

        if target_probs is not None:
            shifted_targets = target_probs[:, 1:].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            loss = -(shifted_targets * log_probs).sum(dim=-1).mean()
            return {"loss": loss}

        if labels is None:
            raise ValueError(
                "LanguageModelingLoss expects labels for hard-target training "
                "or target_probs for soft-target training."
            )

        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
        )
        return {"loss": loss}

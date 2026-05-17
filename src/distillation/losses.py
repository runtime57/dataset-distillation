import torch
from torch.nn import functional as F


def hard_lm_loss(logits, labels, ignore_index=-100):
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def soft_lm_loss(logits, target_probs):
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    shifted_targets = target_probs[:, 1:].contiguous()
    return -(shifted_targets * log_probs).sum(dim=-1).mean()


def entropy(probs):
    log_probs = torch.log(probs.clamp_min(1e-8))
    return -(probs * log_probs).sum(dim=-1).mean()

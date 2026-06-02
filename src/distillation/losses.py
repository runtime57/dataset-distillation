import torch
from torch.nn import functional as F


def _masked_mean(values, attention_mask):
    if attention_mask is None:
        return values.mean()
    mask = attention_mask[:, 1:].to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def hard_lm_loss(logits, labels, attention_mask=None, ignore_index=-100):
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(shift_labels)
    valid_mask = (shift_labels != ignore_index).to(per_token_loss.dtype)
    if attention_mask is not None:
        valid_mask = valid_mask * attention_mask[:, 1:].to(
            device=per_token_loss.device,
            dtype=per_token_loss.dtype,
        )
    return (per_token_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


def soft_lm_loss(logits, target_probs, attention_mask=None):
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    shifted_targets = target_probs[:, 1:].contiguous()
    per_token_loss = -(shifted_targets * log_probs).sum(dim=-1)
    return _masked_mean(per_token_loss, attention_mask)


def entropy(probs):
    log_probs = torch.log(probs.clamp_min(1e-8))
    return -(probs * log_probs).sum(dim=-1).mean()

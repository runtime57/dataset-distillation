import torch


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def resolve_inner_lr(log_inner_lr, config):
    inner_lr = torch.exp(log_inner_lr)
    min_inner_lr = config.distillation.get("min_inner_lr")
    max_inner_lr = config.distillation.get("max_inner_lr")
    if min_inner_lr is not None or max_inner_lr is not None:
        clamp_min = 0.0 if min_inner_lr is None else float(min_inner_lr)
        clamp_max = float("inf") if max_inner_lr is None else float(max_inner_lr)
        inner_lr = inner_lr.clamp(min=clamp_min, max=clamp_max)
    return inner_lr

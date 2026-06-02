import torch
from torch.func import functional_call

from src.distillation.losses import hard_lm_loss
from src.distillation.utils import move_batch_to_device


def outer_loss_on_real_batches(
    model,
    params,
    buffers,
    real_loader,
    device,
    outer_batches,
):
    losses = []
    for _ in range(outer_batches):
        real_batch = move_batch_to_device(next(real_loader), device)
        real_outputs = functional_call(
            model,
            (params, buffers),
            args=(),
            kwargs={
                "input_ids": real_batch["input_ids"],
                "attention_mask": real_batch.get("attention_mask"),
            },
        )
        losses.append(
            hard_lm_loss(
                real_outputs["logits"],
                real_batch["labels"],
                attention_mask=real_batch.get("attention_mask"),
            )
        )
    return torch.stack(losses).mean()

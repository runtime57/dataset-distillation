import torch
from torch.func import functional_call

from src.distillation.losses import hard_lm_loss, soft_lm_loss
from src.distillation.utils import move_batch_to_device


def gradient_matching_loss(
    model,
    params,
    buffers,
    real_loader,
    synth_batch,
    outer_batches,
    device,
):
    real_grads_accum = None
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
        real_loss = hard_lm_loss(real_outputs["logits"], real_batch["labels"])
        real_grads = torch.autograd.grad(
            real_loss,
            tuple(params.values()),
            create_graph=False,
            allow_unused=True,
        )
        if real_grads_accum is None:
            real_grads_accum = [
                grad.detach().clone() if grad is not None else None
                for grad in real_grads
            ]
        else:
            for index, grad in enumerate(real_grads):
                if grad is not None and real_grads_accum[index] is not None:
                    real_grads_accum[index] += grad.detach()
    real_grads_mean = [
        grad / outer_batches if grad is not None else None
        for grad in real_grads_accum
    ]

    synth_outputs = functional_call(
        model,
        (params, buffers),
        args=(),
        kwargs={
            "input_embeds": synth_batch["input_embeds"],
            "attention_mask": synth_batch["attention_mask"],
        },
    )
    synth_loss = soft_lm_loss(synth_outputs["logits"], synth_batch["target_probs"])
    synth_grads = torch.autograd.grad(
        synth_loss,
        tuple(params.values()),
        create_graph=True,
        allow_unused=True,
    )

    total = torch.tensor(0.0, device=device)
    n_matched = 0
    for real_grad, synth_grad in zip(real_grads_mean, synth_grads):
        if real_grad is not None and synth_grad is not None:
            real_flat = real_grad.flatten()
            synth_flat = synth_grad.flatten()
            real_norm = real_flat.norm()
            synth_norm = synth_flat.norm()
            if real_norm > 1e-8 and synth_norm > 1e-8:
                cosine_similarity = (real_flat * synth_flat).sum() / (
                    real_norm * synth_norm
                )
                total = total + (1.0 - cosine_similarity)
                n_matched += 1
    return total / max(n_matched, 1), synth_loss

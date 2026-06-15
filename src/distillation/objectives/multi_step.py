import torch
from torch.func import functional_call

from src.distillation.data import synthetic_batch
from src.distillation.losses import soft_lm_loss


def multi_step_parameters(
    model,
    params,
    buffers,
    synthetic_data,
    config,
    device,
    inner_lr,
):
    n_inner_steps = int(config.distillation.get("n_inner_steps", 4))
    if n_inner_steps < 1:
        raise ValueError(f"n_inner_steps must be positive, got {n_inner_steps}.")

    updated_params = params
    inner_losses = []
    for _ in range(n_inner_steps):
        synth_batch = synthetic_batch(
            synthetic_data=synthetic_data,
            model_params=updated_params,
            batch_size=config.distillation.synthetic.batch_size,
            device=device,
        )
        outputs = functional_call(
            model,
            (updated_params, buffers),
            args=(),
            kwargs={
                "input_embeds": synth_batch["input_embeds"],
                "attention_mask": synth_batch["attention_mask"],
            },
        )
        inner_loss = soft_lm_loss(
            outputs["logits"],
            synth_batch["target_probs"],
            attention_mask=synth_batch["attention_mask"],
        )
        gradients = torch.autograd.grad(
            inner_loss,
            tuple(updated_params.values()),
            create_graph=True,
            allow_unused=True,
        )
        updated_params = {
            name: parameter if gradient is None else parameter - inner_lr * gradient
            for (name, parameter), gradient in zip(updated_params.items(), gradients)
        }
        inner_losses.append(inner_loss)

    return updated_params, torch.stack(inner_losses).mean()

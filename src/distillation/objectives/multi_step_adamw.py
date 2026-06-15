import torch
from torch.func import functional_call

from src.distillation.data import synthetic_batch
from src.distillation.losses import soft_lm_loss
from src.distillation.objectives.optim import adamw_step


def multi_step_adamw_parameters(
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

    beta1 = float(config.distillation.get("inner_beta1", 0.9))
    beta2 = float(config.distillation.get("inner_beta2", 0.999))
    eps = float(config.distillation.get("inner_eps", 1e-8))
    weight_decay = float(config.distillation.get("inner_weight_decay", 0.0))

    updated_params = params
    first_moments = {name: torch.zeros_like(param) for name, param in params.items()}
    second_moments = {name: torch.zeros_like(param) for name, param in params.items()}
    inner_losses = []

    for step in range(1, n_inner_steps + 1):
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

        updated_params, first_moments, second_moments = adamw_step(
            params=updated_params,
            gradients=gradients,
            first_moments=first_moments,
            second_moments=second_moments,
            step=step,
            lr=inner_lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )
        inner_losses.append(inner_loss)

    return updated_params, torch.stack(inner_losses).mean()

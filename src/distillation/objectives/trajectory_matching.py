import torch
from torch.func import functional_call

from src.distillation.data import synthetic_batch
from src.distillation.losses import soft_lm_loss


def trajectory_matching_loss(
    model,
    buffers,
    synthetic_data,
    expert_checkpoints,
    n_inner_steps,
    inner_lr,
    config,
    device,
):
    canonical_keys = set(dict(model.named_parameters()).keys())
    num_experts = len(expert_checkpoints)
    step_index = torch.randint(0, num_experts - 1, (1,)).item()

    expert_start = {
        key: value.to(device)
        for key, value in expert_checkpoints[step_index].items()
        if key in canonical_keys
    }
    expert_end = {
        key: value.to(device)
        for key, value in expert_checkpoints[step_index + 1].items()
        if key in canonical_keys
    }

    current_params = {
        key: value.detach().clone().requires_grad_(True)
        for key, value in expert_start.items()
    }

    for _ in range(n_inner_steps):
        synth_batch_data = synthetic_batch(
            synthetic_data=synthetic_data,
            model_params=current_params,
            batch_size=config.distillation.synthetic.batch_size,
            device=device,
        )
        outputs = functional_call(
            model,
            (current_params, buffers),
            args=(),
            kwargs={
                "input_embeds": synth_batch_data["input_embeds"],
                "attention_mask": synth_batch_data["attention_mask"],
            },
        )
        loss = soft_lm_loss(outputs["logits"], synth_batch_data["target_probs"])
        grads = torch.autograd.grad(
            loss,
            tuple(current_params.values()),
            create_graph=True,
            allow_unused=True,
        )
        updated_params = {}
        for (name, parameter), grad in zip(current_params.items(), grads):
            updated_params[name] = (
                parameter if grad is None else parameter - inner_lr * grad
            )
        current_params = updated_params

    param_names = list(expert_start.keys())
    student_flat = torch.cat([current_params[name].flatten() for name in param_names])
    expert_end_flat = torch.cat(
        [expert_end[name].detach().flatten() for name in param_names]
    )
    expert_start_flat = torch.cat(
        [expert_start[name].detach().flatten() for name in param_names]
    )
    
    numerator = ((student_flat - expert_end_flat) ** 2).sum()
    denominator = ((expert_start_flat - expert_end_flat) ** 2).sum().clamp(min=1e-8)
    return numerator / denominator
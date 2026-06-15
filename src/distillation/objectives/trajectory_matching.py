import torch
from torch.func import functional_call

from src.distillation.data import synthetic_batch
from src.distillation.losses import soft_lm_loss
from src.distillation.objectives.optim import adamw_step


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _checkpoint_optimizer_state(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint.get("optimizer_state", {})
    return {}


def _optimizer_state_tensor(optimizer_state, name, key, parameter):
    value = optimizer_state.get(name, {}).get(key)
    if torch.is_tensor(value):
        return value.to(device=parameter.device, dtype=parameter.dtype).detach().clone()
    return torch.zeros_like(parameter)


def _optimizer_step(optimizer_state, param_names):
    steps = []
    for name in param_names:
        value = optimizer_state.get(name, {}).get("step")
        if torch.is_tensor(value):
            steps.append(int(value.detach().cpu().item()))
        elif value is not None:
            steps.append(int(value))
    return max(steps) if steps else 0


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
    if num_experts < 2:
        raise ValueError("Trajectory matching requires at least two checkpoints.")
    step_index = torch.randint(0, num_experts - 1, (1,)).item()
    start_state = _checkpoint_state_dict(expert_checkpoints[step_index])
    end_state = _checkpoint_state_dict(expert_checkpoints[step_index + 1])
    start_optimizer_state = _checkpoint_optimizer_state(expert_checkpoints[step_index])

    expert_start = {
        key: value.to(device)
        for key, value in start_state.items()
        if key in canonical_keys
    }
    expert_end = {
        key: value.to(device)
        for key, value in end_state.items()
        if key in canonical_keys
    }

    current_params = {
        key: value.detach().clone().requires_grad_(True)
        for key, value in expert_start.items()
    }
    inner_optimizer = str(config.distillation.get("inner_optimizer", "sgd")).lower()
    beta1 = float(config.distillation.get("inner_beta1", 0.9))
    beta2 = float(config.distillation.get("inner_beta2", 0.999))
    eps = float(config.distillation.get("inner_eps", 1e-8))
    weight_decay = float(config.distillation.get("inner_weight_decay", 0.0))
    first_moments = {
        name: _optimizer_state_tensor(start_optimizer_state, name, "exp_avg", param)
        for name, param in current_params.items()
    }
    second_moments = {
        name: _optimizer_state_tensor(start_optimizer_state, name, "exp_avg_sq", param)
        for name, param in current_params.items()
    }
    start_optimizer_step = _optimizer_step(
        start_optimizer_state,
        current_params.keys(),
    )

    for inner_step in range(1, n_inner_steps + 1):
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
        loss = soft_lm_loss(
            outputs["logits"],
            synth_batch_data["target_probs"],
            attention_mask=synth_batch_data["attention_mask"],
        )
        grads = torch.autograd.grad(
            loss,
            tuple(current_params.values()),
            create_graph=True,
            allow_unused=True,
        )
        if inner_optimizer == "sgd":
            current_params = {
                name: parameter if grad is None else parameter - inner_lr * grad
                for (name, parameter), grad in zip(current_params.items(), grads)
            }
        elif inner_optimizer == "adamw":
            current_params, first_moments, second_moments = adamw_step(
                params=current_params,
                gradients=grads,
                first_moments=first_moments,
                second_moments=second_moments,
                step=start_optimizer_step + inner_step,
                lr=inner_lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(
                "trajectory_matching inner_optimizer must be 'sgd' or 'adamw', "
                f"got {inner_optimizer!r}."
            )

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

import torch
from torch.func import functional_call

from src.distillation.data import synthetic_batch
from src.distillation.losses import hard_lm_loss, soft_lm_loss
from src.distillation.utils import move_batch_to_device


def _matched_parameters(params, parameter_names=None):
    if parameter_names is None:
        return tuple(params.values())

    matched_names = list(parameter_names)
    missing = [name for name in matched_names if name not in params]
    if missing:
        raise KeyError(
            "Unknown gradient matching parameter names: "
            f"{', '.join(missing)}"
        )
    return tuple(params[name] for name in matched_names)


def _real_gradients(model, params, buffers, real_batch, matched_params):
    real_outputs = functional_call(
        model,
        (params, buffers),
        args=(),
        kwargs={
            "input_ids": real_batch["input_ids"],
            "attention_mask": real_batch.get("attention_mask"),
        },
    )
    real_loss = hard_lm_loss(
        real_outputs["logits"],
        real_batch["labels"],
        attention_mask=real_batch.get("attention_mask"),
    )
    return torch.autograd.grad(
        real_loss,
        matched_params,
        create_graph=False,
        allow_unused=True,
    )


def _synthetic_gradients(model, params, buffers, synth_batch, matched_params):
    synth_outputs = functional_call(
        model,
        (params, buffers),
        args=(),
        kwargs={
            "input_embeds": synth_batch["input_embeds"],
            "attention_mask": synth_batch["attention_mask"],
        },
    )
    synth_loss = soft_lm_loss(
        synth_outputs["logits"],
        synth_batch["target_probs"],
        attention_mask=synth_batch["attention_mask"],
    )
    synth_grads = torch.autograd.grad(
        synth_loss,
        matched_params,
        create_graph=True,
        allow_unused=True,
    )
    return synth_loss, synth_grads


def _mean_real_gradients(real_grads_accum, outer_batches):
    return [
        grad / outer_batches if grad is not None else None
        for grad in real_grads_accum
    ]


def _accumulate_gradients(accum, gradients):
    if accum is None:
        return [
            grad.detach().clone() if grad is not None else None
            for grad in gradients
        ]

    for index, grad in enumerate(gradients):
        if grad is not None and accum[index] is not None:
            accum[index] += grad.detach()
    return accum


def _gradient_distance(real_grads, synth_grads, device):
    total = torch.tensor(0.0, device=device)
    total_weight = 0
    for real_grad, synth_grad in zip(real_grads, synth_grads):
        if real_grad is not None and synth_grad is not None:
            real_flat = real_grad.flatten()
            synth_flat = synth_grad.flatten()
            real_norm = real_flat.norm()
            synth_norm = synth_flat.norm()
            if real_norm > 1e-8 and synth_norm > 1e-8:
                cosine_similarity = (real_flat * synth_flat).sum() / (
                    real_norm * synth_norm
                )
                weight = real_flat.numel()
                total = total + weight * (1.0 - cosine_similarity)
                total_weight += weight
    return total / max(total_weight, 1)


def gradient_matching_loss(
    model,
    params,
    buffers,
    real_loader,
    synth_batch,
    outer_batches,
    device,
    parameter_names=None,
):
    matched_params = _matched_parameters(params, parameter_names)
    real_grads_accum = None
    for _ in range(outer_batches):
        real_batch = move_batch_to_device(next(real_loader), device)
        real_grads_accum = _accumulate_gradients(
            real_grads_accum,
            _real_gradients(model, params, buffers, real_batch, matched_params),
        )
    real_grads_mean = _mean_real_gradients(real_grads_accum, outer_batches)

    synth_loss, synth_grads = _synthetic_gradients(
        model,
        params,
        buffers,
        synth_batch,
        matched_params,
    )
    return _gradient_distance(real_grads_mean, synth_grads, device), synth_loss


def grouped_gradient_matching_loss(
    model,
    params,
    buffers,
    group_matcher,
    synthetic_data,
    outer_batches,
    device,
    parameter_names=None,
):
    matched_params = _matched_parameters(params, parameter_names)
    total = torch.tensor(0.0, device=device)
    synth_losses = []
    group_ids = group_matcher.sample_groups(group_matcher.groups_per_step)

    for group_id in group_ids:
        real_grads_accum = None
        for _ in range(outer_batches):
            real_batch = group_matcher.sample_real_batch(
                group_id,
                group_matcher.real_batch_size,
                device,
            )
            real_grads_accum = _accumulate_gradients(
                real_grads_accum,
                _real_gradients(model, params, buffers, real_batch, matched_params),
            )
        real_grads_mean = _mean_real_gradients(real_grads_accum, outer_batches)

        synth_indices = group_matcher.sample_synthetic_indices(
            group_id,
            group_matcher.synth_batch_size,
            device,
        )
        synth_batch_data = synthetic_batch(
            synthetic_data=synthetic_data,
            model_params=params,
            batch_size=None,
            device=device,
            indices=synth_indices,
        )
        synth_loss, synth_grads = _synthetic_gradients(
            model,
            params,
            buffers,
            synth_batch_data,
            matched_params,
        )
        total = total + _gradient_distance(real_grads_mean, synth_grads, device)
        synth_losses.append(synth_loss)

    return total / len(group_ids), torch.stack(synth_losses).mean()

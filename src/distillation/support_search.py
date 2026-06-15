import math

import torch
from torch.func import functional_call

from src.distillation.losses import hard_lm_loss, soft_lm_loss
from src.distillation.objectives.common import outer_loss_on_real_batches
from src.distillation.objectives.optim import adamw_step
from src.distillation.utils import move_batch_to_device


def _get_search_config(config):
    return config.distillation.get("support_search", {})


def support_search_enabled(config, synthetic_data):
    search_config = _get_search_config(config)
    return bool(search_config.get("enabled", False)) and hasattr(
        synthetic_data,
        "replace_hard_tokens",
    )


def should_run_support_search(config, step, synthetic_data):
    if not support_search_enabled(config, synthetic_data):
        return False
    period = int(_get_search_config(config).get("period", 20))
    if period <= 0:
        raise ValueError(f"support_search.period must be positive, got {period}.")
    return step % period == 0


def _one_step_parameters_from_probs(
    model,
    params,
    buffers,
    probe_probs,
    embedding_weight,
    inner_lr,
    create_graph=True,
):
    attention_mask = torch.ones(
        probe_probs.shape[:2],
        dtype=torch.long,
        device=probe_probs.device,
    )
    outputs = functional_call(
        model,
        (params, buffers),
        args=(),
        kwargs={
            "input_embeds": probe_probs @ embedding_weight,
            "attention_mask": attention_mask,
        },
    )
    inner_loss = soft_lm_loss(
        outputs["logits"],
        probe_probs,
        attention_mask=attention_mask,
    )
    gradients = torch.autograd.grad(
        inner_loss,
        tuple(params.values()),
        create_graph=create_graph,
        allow_unused=True,
    )
    updated_params = {}
    for (name, parameter), gradient in zip(params.items(), gradients):
        updated_params[name] = (
            parameter if gradient is None else parameter - inner_lr * gradient
        )
    return updated_params, inner_loss


def _multi_step_adamw_parameters_from_probs(
    model,
    params,
    buffers,
    probe_probs,
    embedding_weight,
    inner_lr,
    n_inner_steps,
    beta1,
    beta2,
    eps,
    weight_decay,
    create_graph=False,
    detach_between_steps=True,
):
    attention_mask = torch.ones(
        probe_probs.shape[:2],
        dtype=torch.long,
        device=probe_probs.device,
    )
    updated_params = {
        name: parameter.detach().requires_grad_(True)
        for name, parameter in params.items()
    }
    first_moments = {
        name: torch.zeros_like(parameter)
        for name, parameter in updated_params.items()
    }
    second_moments = {
        name: torch.zeros_like(parameter)
        for name, parameter in updated_params.items()
    }
    inner_losses = []

    for step in range(1, int(n_inner_steps) + 1):
        outputs = functional_call(
            model,
            (updated_params, buffers),
            args=(),
            kwargs={
                "input_embeds": probe_probs @ embedding_weight,
                "attention_mask": attention_mask,
            },
        )
        inner_loss = soft_lm_loss(
            outputs["logits"],
            probe_probs,
            attention_mask=attention_mask,
        )
        gradients = torch.autograd.grad(
            inner_loss,
            tuple(updated_params.values()),
            create_graph=create_graph,
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
        if detach_between_steps:
            updated_params = {
                name: parameter.detach().requires_grad_(True)
                for name, parameter in updated_params.items()
            }
            first_moments = {
                name: moment.detach()
                for name, moment in first_moments.items()
            }
            second_moments = {
                name: moment.detach()
                for name, moment in second_moments.items()
            }
        inner_losses.append(inner_loss if create_graph else inner_loss.detach())

    return updated_params, torch.stack(inner_losses).mean()


def _multi_step_adamw_settings(config, settings):
    return {
        "n_inner_steps": int(
            settings.get(
                "n_inner_steps",
                config.distillation.get("n_inner_steps", 4),
            )
        ),
        "beta1": float(
            settings.get(
                "inner_beta1",
                config.distillation.get("inner_beta1", 0.9),
            )
        ),
        "beta2": float(
            settings.get(
                "inner_beta2",
                config.distillation.get("inner_beta2", 0.999),
            )
        ),
        "eps": float(
            settings.get(
                "inner_eps",
                config.distillation.get("inner_eps", 1e-8),
            )
        ),
        "weight_decay": float(
            settings.get(
                "inner_weight_decay",
                config.distillation.get("inner_weight_decay", 0.0),
            )
        ),
    }


def _proposal_parameters_from_probs(
    model,
    params,
    buffers,
    probe_probs,
    embedding_weight,
    inner_lr,
    config,
    search_config,
    acceptance_config,
):
    proposal_objective = str(search_config.get("proposal_objective", "one_step")).lower()
    if proposal_objective == "one_step":
        return _one_step_parameters_from_probs(
            model=model,
            params=params,
            buffers=buffers,
            probe_probs=probe_probs,
            embedding_weight=embedding_weight,
            inner_lr=inner_lr,
        )
    if proposal_objective == "multi_step_adamw":
        proposal_settings = search_config.get("proposal", {})
        settings = _multi_step_adamw_settings(config, proposal_settings)
        settings["n_inner_steps"] = int(
            proposal_settings.get(
                "n_inner_steps",
                acceptance_config.get(
                    "n_inner_steps",
                    config.distillation.get("n_inner_steps", 4),
                ),
            )
        )
        return _multi_step_adamw_parameters_from_probs(
            model=model,
            params=params,
            buffers=buffers,
            probe_probs=probe_probs,
            embedding_weight=embedding_weight,
            inner_lr=inner_lr,
            create_graph=True,
            detach_between_steps=False,
            **settings,
        )
    raise ValueError(
        "support_search.proposal_objective must be 'one_step' or "
        f"'multi_step_adamw', got {proposal_objective!r}."
    )


def _collect_fixed_real_batches(real_loader, device, outer_batches):
    return [move_batch_to_device(next(real_loader), device) for _ in range(outer_batches)]


def _outer_loss_on_fixed_real_batches(model, params, buffers, real_batches):
    losses = []
    for real_batch in real_batches:
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


def _synthetic_hard_and_support_ids(synthetic_data, indices):
    if hasattr(synthetic_data, "support_ids"):
        support_ids = synthetic_data.support_ids[indices]
        return support_ids[..., 0], support_ids

    logits = getattr(synthetic_data, "logits", None)
    if logits is None or logits.dim() != 3:
        raise ValueError(
            "Accepted support search needs synthetic_data.support_ids or "
            "full-vocab synthetic_data.logits."
        )
    logits = logits[indices]
    k = int(getattr(synthetic_data, "k", 1))
    support_ids = logits.topk(k, dim=-1).indices
    return logits.argmax(dim=-1), support_ids


def _capture_rows(synthetic_data, sequence_indices, position_indices):
    sequence_indices = sequence_indices.detach().to(dtype=torch.long).cpu()
    position_indices = position_indices.detach().to(dtype=torch.long).cpu()
    device = next(synthetic_data.parameters()).device
    sequence_device = sequence_indices.to(device=device)
    position_device = position_indices.to(device=device)
    snapshot = {
        "sequence_indices": sequence_device,
        "position_indices": position_device,
    }
    if hasattr(synthetic_data, "support_ids"):
        snapshot["support_ids"] = synthetic_data.support_ids[
            sequence_device,
            position_device,
        ].detach().clone()
    if hasattr(synthetic_data, "logits"):
        snapshot["logits"] = synthetic_data.logits[
            sequence_device,
            position_device,
        ].detach().clone()
    return snapshot


@torch.no_grad()
def _restore_rows(synthetic_data, snapshot):
    sequence_indices = snapshot["sequence_indices"]
    position_indices = snapshot["position_indices"]
    if "support_ids" in snapshot:
        synthetic_data.support_ids[sequence_indices, position_indices] = snapshot[
            "support_ids"
        ]
    if "logits" in snapshot:
        synthetic_data.logits[sequence_indices, position_indices] = snapshot["logits"]


def _score_one_step_proxy(
    model,
    params,
    buffers,
    synthetic_data,
    indices,
    real_batches,
    inner_lr,
    embedding_weight,
):
    token_probs = synthetic_data.token_probs(
        indices=indices,
        embedding_weight=embedding_weight,
    ).detach()
    updated_params, inner_loss = _one_step_parameters_from_probs(
        model=model,
        params=params,
        buffers=buffers,
        probe_probs=token_probs,
        embedding_weight=embedding_weight,
        inner_lr=inner_lr,
        create_graph=False,
    )
    outer_loss = _outer_loss_on_fixed_real_batches(
        model=model,
        params=updated_params,
        buffers=buffers,
        real_batches=real_batches,
    )
    return float(outer_loss.detach().cpu()), float(inner_loss.detach().cpu())


def _score_multi_step_adamw_proxy(
    model,
    params,
    buffers,
    synthetic_data,
    indices,
    real_batches,
    inner_lr,
    embedding_weight,
    n_inner_steps,
    beta1,
    beta2,
    eps,
    weight_decay,
):
    token_probs = synthetic_data.token_probs(
        indices=indices,
        embedding_weight=embedding_weight,
    ).detach()
    updated_params, inner_loss = _multi_step_adamw_parameters_from_probs(
        model=model,
        params=params,
        buffers=buffers,
        probe_probs=token_probs,
        embedding_weight=embedding_weight,
        inner_lr=inner_lr,
        n_inner_steps=n_inner_steps,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )
    outer_loss = _outer_loss_on_fixed_real_batches(
        model=model,
        params=updated_params,
        buffers=buffers,
        real_batches=real_batches,
    )
    return float(outer_loss.detach().cpu()), float(inner_loss.detach().cpu())


def _score_acceptance_proxy(
    model,
    params,
    buffers,
    synthetic_data,
    indices,
    real_batches,
    inner_lr,
    embedding_weight,
    config,
    acceptance_config,
):
    proxy_objective = str(
        acceptance_config.get("proxy_objective", "one_step")
    ).lower()
    if proxy_objective == "one_step":
        return _score_one_step_proxy(
            model=model,
            params=params,
            buffers=buffers,
            synthetic_data=synthetic_data,
            indices=indices,
            real_batches=real_batches,
            inner_lr=inner_lr,
            embedding_weight=embedding_weight,
        )
    if proxy_objective == "multi_step_adamw":
        return _score_multi_step_adamw_proxy(
            model=model,
            params=params,
            buffers=buffers,
            synthetic_data=synthetic_data,
            indices=indices,
            real_batches=real_batches,
            inner_lr=inner_lr,
            embedding_weight=embedding_weight,
            n_inner_steps=int(
                acceptance_config.get(
                    "n_inner_steps",
                    config.distillation.get("n_inner_steps", 4),
                )
            ),
            beta1=float(
                acceptance_config.get(
                    "inner_beta1",
                    config.distillation.get("inner_beta1", 0.9),
                )
            ),
            beta2=float(
                acceptance_config.get(
                    "inner_beta2",
                    config.distillation.get("inner_beta2", 0.999),
                )
            ),
            eps=float(
                acceptance_config.get(
                    "inner_eps",
                    config.distillation.get("inner_eps", 1e-8),
                )
            ),
            weight_decay=float(
                acceptance_config.get(
                    "inner_weight_decay",
                    config.distillation.get("inner_weight_decay", 0.0),
                )
            ),
        )
    raise ValueError(
        "support_search.acceptance.proxy_objective must be 'one_step' or "
        f"'multi_step_adamw', got {proxy_objective!r}."
    )


def _apply_masks(
    candidate_grad,
    support_ids,
    exclude_token_ids,
    hard_ids=None,
    exclude_support=True,
):
    candidate_grad = candidate_grad.clone()
    if exclude_support:
        candidate_grad.scatter_(-1, support_ids, math.inf)
    elif hard_ids is not None:
        candidate_grad.scatter_(-1, hard_ids.unsqueeze(-1), math.inf)
    for token_id in exclude_token_ids:
        if 0 <= int(token_id) < candidate_grad.shape[-1]:
            candidate_grad[..., int(token_id)] = math.inf
    return candidate_grad


def hard_em_topk_step(
    model,
    params,
    buffers,
    synthetic_data,
    real_loader,
    config,
    device,
    inner_lr,
    embedding_weight,
):
    """
    Conservative hard-EM E-step for fixed top-k supports.

    A dense probability probe is built from the current sparse top-k batch.
    The gradient of the one-step outer loss with respect to that probe gives a
    first-order score for replacing the current high-probability token at each
    position. Positions with the largest positive hard-token gain are swapped.
    """
    search_config = _get_search_config(config)
    batch_size = int(
        search_config.get(
            "batch_size",
            config.distillation.synthetic.get("batch_size", 16),
        )
    )
    max_replacements = int(search_config.get("max_replacements", 256))
    min_gain = float(search_config.get("min_gain", 0.0))
    outer_batches = int(
        search_config.get(
            "outer_batches",
            config.distillation.get("outer_batches", 1),
        )
    )
    acceptance_config = search_config.get("acceptance", {})
    acceptance_enabled = bool(acceptance_config.get("enabled", False))
    proposal_objective = str(search_config.get("proposal_objective", "one_step")).lower()
    proxy_objective = str(
        acceptance_config.get("proxy_objective", "one_step")
    ).lower()
    proposal_count = int(
        acceptance_config.get(
            "proposal_count",
            search_config.get("proposal_count", max_replacements),
        )
    )
    group_size = int(acceptance_config.get("group_size", max_replacements))
    min_improvement = float(acceptance_config.get("min_improvement", 0.0))
    keep_old_hard = bool(search_config.get("keep_old_hard", True))
    reset_logits = bool(search_config.get("reset_logits", True))
    exclude_support = bool(search_config.get("exclude_support", True))
    confidence = search_config.get(
        "confidence",
        config.distillation.synthetic.get("init_confidence"),
    )
    exclude_token_ids = search_config.get("exclude_token_ids", [0])

    if batch_size <= 0:
        raise ValueError(f"support_search.batch_size must be positive, got {batch_size}.")
    if max_replacements <= 0:
        return {
            "support_replacements": 0,
            "support_mean_gain": 0.0,
            "support_max_gain": 0.0,
            "support_inner_loss": 0.0,
            "support_outer_loss": 0.0,
        }

    if group_size <= 0:
        raise ValueError(f"support_search.acceptance.group_size must be positive, got {group_size}.")

    was_training = synthetic_data.training
    synthetic_data.eval()
    try:
        indices = synthetic_data.sample_indices(batch_size, device=device)
        fixed_real_batches = (
            _collect_fixed_real_batches(real_loader, device, outer_batches)
            if acceptance_enabled
            else None
        )
        current_probs = synthetic_data.token_probs(
            indices=indices,
            embedding_weight=embedding_weight,
        ).detach()
        probe_probs = current_probs.clone().requires_grad_(True)

        updated_params, inner_loss = _proposal_parameters_from_probs(
            model=model,
            params=params,
            buffers=buffers,
            probe_probs=probe_probs,
            embedding_weight=embedding_weight,
            inner_lr=inner_lr,
            config=config,
            search_config=search_config,
            acceptance_config=acceptance_config,
        )
        if fixed_real_batches is None:
            outer_loss = outer_loss_on_real_batches(
                model=model,
                params=updated_params,
                buffers=buffers,
                real_loader=real_loader,
                device=device,
                outer_batches=outer_batches,
            )
        else:
            outer_loss = _outer_loss_on_fixed_real_batches(
                model=model,
                params=updated_params,
                buffers=buffers,
                real_batches=fixed_real_batches,
            )
        (grad_probs,) = torch.autograd.grad(outer_loss, probe_probs)

        with torch.no_grad():
            hard_ids, support_ids = _synthetic_hard_and_support_ids(
                synthetic_data,
                indices,
            )
            hard_grad = grad_probs.gather(-1, hard_ids.unsqueeze(-1)).squeeze(-1)
            candidate_grad = _apply_masks(
                candidate_grad=grad_probs,
                support_ids=support_ids,
                exclude_token_ids=exclude_token_ids,
                hard_ids=hard_ids,
                exclude_support=exclude_support,
            )
            best_candidate_grad, best_candidate_ids = candidate_grad.min(dim=-1)
            gains = hard_grad - best_candidate_grad
            valid = torch.isfinite(best_candidate_grad) & (gains > min_gain)
            if not valid.any():
                return {
                    "support_replacements": 0,
                    "support_proposals": 0,
                    "support_rejected": 0,
                    "support_mean_gain": 0.0,
                    "support_max_gain": 0.0,
                    "support_inner_loss": float(inner_loss.detach().cpu()),
                    "support_outer_loss": float(outer_loss.detach().cpu()),
                }

            flat_gains = gains.reshape(-1)
            flat_valid = valid.reshape(-1)
            valid_indices = flat_valid.nonzero(as_tuple=True)[0]
            valid_gains = flat_gains[valid_indices]
            replacement_count = min(
                proposal_count if acceptance_enabled else max_replacements,
                int(valid_indices.numel()),
            )
            selected_order = valid_gains.topk(replacement_count).indices
            selected_flat = valid_indices[selected_order]
            selected_gains = flat_gains[selected_flat]

            sequence_length = gains.shape[1]
            batch_positions = selected_flat // sequence_length
            position_indices = selected_flat % sequence_length
            sequence_indices = indices[batch_positions]
            token_ids = best_candidate_ids.reshape(-1)[selected_flat]

        if not acceptance_enabled:
            with torch.no_grad():
                changed = synthetic_data.replace_hard_tokens(
                    sequence_indices=sequence_indices,
                    position_indices=position_indices,
                    token_ids=token_ids,
                    confidence=confidence,
                    keep_old_hard=keep_old_hard,
                    reset_logits=reset_logits,
                )
            rejected = 0
            proxy_before = float(outer_loss.detach().cpu())
            proxy_after = proxy_before
        else:
            proxy_before, _ = _score_acceptance_proxy(
                model=model,
                params=params,
                buffers=buffers,
                synthetic_data=synthetic_data,
                indices=indices,
                real_batches=fixed_real_batches,
                inner_lr=inner_lr,
                embedding_weight=embedding_weight,
                config=config,
                acceptance_config=acceptance_config,
            )
            proxy_after = proxy_before
            changed = 0
            rejected = 0
            accepted_gains = []
            max_groups = math.ceil(
                min(max_replacements, selected_flat.numel()) / group_size
            )
            for group_index in range(max_groups):
                start = group_index * group_size
                end = min(start + group_size, selected_flat.numel(), max_replacements)
                if start >= end:
                    break
                group_sequences = sequence_indices[start:end]
                group_positions = position_indices[start:end]
                group_tokens = token_ids[start:end]
                snapshot = _capture_rows(
                    synthetic_data,
                    group_sequences,
                    group_positions,
                )
                with torch.no_grad():
                    group_changed = synthetic_data.replace_hard_tokens(
                        sequence_indices=group_sequences,
                        position_indices=group_positions,
                        token_ids=group_tokens,
                        confidence=confidence,
                        keep_old_hard=keep_old_hard,
                        reset_logits=reset_logits,
                    )
                if group_changed == 0:
                    continue
                candidate_score, _ = _score_acceptance_proxy(
                    model=model,
                    params=params,
                    buffers=buffers,
                    synthetic_data=synthetic_data,
                    indices=indices,
                    real_batches=fixed_real_batches,
                    inner_lr=inner_lr,
                    embedding_weight=embedding_weight,
                    config=config,
                    acceptance_config=acceptance_config,
                )
                if candidate_score < proxy_after - min_improvement:
                    changed += int(group_changed)
                    accepted_gains.append(selected_gains[start:end].detach())
                    proxy_after = candidate_score
                else:
                    _restore_rows(synthetic_data, snapshot)
                    rejected += int(group_changed)
            if accepted_gains:
                selected_gains = torch.cat(accepted_gains)
            else:
                selected_gains = selected_gains[:0]
    finally:
        synthetic_data.train(was_training)

    return {
        "support_replacements": int(changed),
        "support_proposals": int(selected_flat.numel()),
        "support_rejected": int(rejected),
        "support_mean_gain": (
            0.0 if changed == 0 else float(selected_gains.mean().cpu())
        ),
        "support_max_gain": (
            0.0 if changed == 0 else float(selected_gains.max().cpu())
        ),
        "support_inner_loss": float(inner_loss.detach().cpu()),
        "support_outer_loss": float(outer_loss.detach().cpu()),
        "support_proxy_before": float(proxy_before),
        "support_proxy_after": float(proxy_after),
        "support_proposal_objective": proposal_objective,
        "support_proxy_objective": proxy_objective,
    }

import math
from copy import deepcopy
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import trange

from src.datasets.data_utils import build_dataloaders, build_datasets, inf_loop
from src.distillation.checkpoints import (
    append_metrics,
    prepare_save_dir,
    save_checkpoint,
    snapshot_synthetic_data,
)
from src.distillation.data import (
    collect_initial_tokens,
    collect_initial_tokens_from_dataset,
    collect_token_mixture_logits_from_dataset,
    synthetic_batch,
)
from src.distillation.factory import build_synthetic_data
from src.distillation.grouping import TextGroupMatcher
from src.distillation.losses import entropy
from src.distillation.objectives import (
    feature_matching_loss,
    gradient_matching_loss,
    grouped_feature_matching_loss,
    grouped_gradient_matching_loss,
    multi_step_adamw_parameters,
    multi_step_parameters,
    one_step_parameters,
    outer_loss_on_real_batches,
    trajectory_matching_loss,
)
from src.distillation.support_search import (
    hard_em_topk_step,
    should_run_support_search,
)
from src.distillation.utils import resolve_inner_lr
from src.tokenizers import build_tokenizer_from_dataset_config
from src.utils.init_utils import resolve_device, set_random_seed
from src.utils.io_utils import ROOT_PATH


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _collect_synthetic_init_tokens(config, dataloaders, datasets, device):
    init_sequence_offset = config.distillation.synthetic.get("init_sequence_offset")
    if init_sequence_offset is None:
        return collect_initial_tokens(
            dataloaders["train"],
            num_sequences=config.distillation.synthetic.num_sequences,
            device=device,
        )
    return collect_initial_tokens_from_dataset(
        datasets["train"],
        num_sequences=config.distillation.synthetic.num_sequences,
        offset=init_sequence_offset,
        device=device,
    )


def _plain_config(config_node):
    return OmegaConf.to_container(config_node, resolve=True)


def _resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT_PATH / path
    return path


def _initialize_synthetic_from_checkpoint(
    config, synthetic_data, embedding_weight, device
):
    checkpoint_path = config.distillation.synthetic.get("init_checkpoint_path")
    if checkpoint_path is None:
        return False
    if not hasattr(synthetic_data, "initialize_from_checkpoint"):
        raise ValueError(
            "distillation.synthetic.init_checkpoint_path requires a synthetic "
            "parameterization with initialize_from_checkpoint()."
        )
    checkpoint = torch.load(
        _resolve_path(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    synthetic_data.initialize_from_checkpoint(
        checkpoint,
        confidence=config.distillation.synthetic.get("init_checkpoint_confidence"),
        embedding_weight=embedding_weight,
    )
    return True


def _assert_metadata_equal(name, expected, actual):
    if expected != actual:
        raise ValueError(
            f"Expert trajectory {name} does not match current TM config. "
            f"Expected {expected!r}, got {actual!r}."
        )


def _assert_dataloader_compatible(expected, actual):
    expected_batch_size = expected.get("batch_size")
    actual_batch_size = actual.get("batch_size") if isinstance(actual, dict) else None
    if expected_batch_size != actual_batch_size:
        raise ValueError(
            "Expert trajectory dataloader batch_size does not match current "
            "TM config. "
            f"Expected {expected_batch_size!r}, got {actual_batch_size!r}."
        )


def _metadata_optimizer(metadata):
    optimizer = metadata.get("optimizer")
    if isinstance(optimizer, dict):
        return optimizer
    return {
        "name": "SGD",
        "lr": metadata.get("lr"),
        "momentum": metadata.get("momentum", 0.0),
    }


def _assert_close(name, expected, actual, tolerance=1e-12):
    if actual is None:
        raise ValueError(f"Expert trajectory metadata is missing {name}.")
    if abs(float(expected) - float(actual)) > tolerance:
        raise ValueError(
            f"Expert trajectory {name} does not match current TM config. "
            f"Expected {expected!r}, got {actual!r}."
        )


def _validate_expert_trajectory(payload, config):
    if not isinstance(payload, dict) or "metadata" not in payload:
        raise ValueError(
            "Expert trajectory must use the metadata/checkpoints format. "
            "Regenerate it with src/compute_expert_trajectory.py after the "
            "TM protocol fix."
        )
    checkpoints = payload.get("checkpoints")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(checkpoints, list):
        raise ValueError(
            "Malformed expert trajectory: expected dict metadata and list "
            "checkpoints."
        )
    if len(checkpoints) < 2:
        raise ValueError("Expert trajectory must contain at least two checkpoints.")

    steps = []
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Malformed expert checkpoint #{index}: not a dict.")
        if "step" not in checkpoint or "state_dict" not in checkpoint:
            raise ValueError(
                f"Malformed expert checkpoint #{index}: expected 'step' and "
                "'state_dict'."
            )
        steps.append(int(checkpoint["step"]))
    if steps[0] != 0:
        raise ValueError(f"Expert trajectory must start at step 0, got {steps[0]}.")

    n_inner_steps = int(config.distillation.get("n_inner_steps", 5))
    step_diffs = [right - left for left, right in zip(steps, steps[1:])]
    if any(diff != n_inner_steps for diff in step_diffs):
        raise ValueError(
            "Expert checkpoint intervals must equal distillation.n_inner_steps. "
            f"Got steps={steps[:10]}{'...' if len(steps) > 10 else ''}, "
            f"diffs={step_diffs[:10]}{'...' if len(step_diffs) > 10 else ''}, "
            f"n_inner_steps={n_inner_steps}."
        )

    _assert_metadata_equal("model", _plain_config(config.model), metadata.get("model"))
    _assert_metadata_equal(
        "train dataset",
        _plain_config(config.datasets.train),
        metadata.get("datasets", {}).get("train"),
    )
    _assert_dataloader_compatible(
        _plain_config(config.dataloader),
        metadata.get("dataloader"),
    )
    _assert_metadata_equal(
        "save_period",
        n_inner_steps,
        int(metadata.get("save_period")),
    )

    optimizer_metadata = _metadata_optimizer(metadata)
    expert_optimizer = str(optimizer_metadata.get("name", "sgd")).lower()
    inner_optimizer = str(config.distillation.get("inner_optimizer", "sgd")).lower()
    if expert_optimizer != inner_optimizer:
        raise ValueError(
            "Expert trajectory optimizer does not match "
            "distillation.inner_optimizer. "
            f"Expected {inner_optimizer!r}, got {expert_optimizer!r}."
        )
    if inner_optimizer == "sgd":
        momentum = float(optimizer_metadata.get("momentum", 0.0))
        if momentum != 0.0:
            raise ValueError(
                "SGD trajectory matching currently supports only "
                "momentum-free expert trajectories. "
                f"Got expert momentum={momentum}."
            )
    elif inner_optimizer == "adamw":
        missing_state_steps = [
            checkpoint["step"]
            for checkpoint in checkpoints
            if "optimizer_state" not in checkpoint
        ]
        if missing_state_steps:
            raise ValueError(
                "AdamW trajectory matching requires optimizer_state in every "
                "expert checkpoint. Regenerate the trajectory with the current "
                "src/compute_expert_trajectory.py. "
                f"Missing at steps={missing_state_steps[:10]}"
                f"{'...' if len(missing_state_steps) > 10 else ''}."
            )
        betas = optimizer_metadata.get("betas", (0.9, 0.999))
        _assert_close(
            "inner_beta1",
            config.distillation.get("inner_beta1", 0.9),
            betas[0],
        )
        _assert_close(
            "inner_beta2",
            config.distillation.get("inner_beta2", 0.999),
            betas[1],
        )
        _assert_close(
            "inner_eps",
            config.distillation.get("inner_eps", 1e-8),
            optimizer_metadata.get("eps", 1e-8),
        )
        _assert_close(
            "inner_weight_decay",
            config.distillation.get("inner_weight_decay", 0.0),
            optimizer_metadata.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(
            "trajectory_matching inner_optimizer must be 'sgd' or 'adamw', "
            f"got {inner_optimizer!r}."
        )

    if not bool(config.distillation.get("learn_inner_lr", False)):
        _assert_close(
            "lr",
            config.distillation.inner_lr,
            optimizer_metadata.get("lr", metadata.get("lr")),
        )
    return checkpoints, metadata


def _load_expert_checkpoints(config, objective):
    if objective != "trajectory_matching":
        return None
    expert_path = ROOT_PATH / config.distillation.expert_trajectory_path
    payload = torch.load(expert_path, map_location="cpu")
    expert_checkpoints, metadata = _validate_expert_trajectory(payload, config)
    print(
        f"Loaded {len(expert_checkpoints)} expert checkpoints "
        f"(step {metadata['save_period']}, "
        f"optimizer {_metadata_optimizer(metadata).get('name')}, "
        f"lr {_metadata_optimizer(metadata).get('lr')}) from {expert_path}"
    )
    return expert_checkpoints


def _regularized_loss(loss, synthetic_data, embedding_weight, config):
    synth_probs = synthetic_data.token_probs(embedding_weight=embedding_weight)
    entropy_value = entropy(synth_probs)
    regularizer = float(config.distillation.get("entropy_weight", 0.0)) * entropy_value

    target_entropy_value = None
    target_entropy_weight = float(config.distillation.get("target_entropy_weight", 0.0))
    if target_entropy_weight != 0.0:
        if getattr(synthetic_data, "uses_decoupled_targets", False):
            target_probs = synthetic_data.target_probs(
                embedding_weight=embedding_weight,
            )
        else:
            target_probs = synth_probs
        target_entropy_value = entropy(target_probs)
        regularizer = regularizer + target_entropy_weight * target_entropy_value

    return loss - regularizer, entropy_value, target_entropy_value


def run_distillation(config):
    set_random_seed(config.distillation.seed)
    device = resolve_device(config.distillation.device)

    save_dir = prepare_save_dir(config)
    requested_partitions = ["train", config.distillation.real_split]
    datasets = build_datasets(config, partitions=requested_partitions)
    dataloaders = build_dataloaders(config, datasets)
    real_loader = inf_loop(dataloaders[config.distillation.real_split])
    tokenizer = build_tokenizer_from_dataset_config(config.datasets.train)

    model = instantiate(config.model).to(device)
    model.train()
    initial_state = deepcopy(model.state_dict())

    synthetic_data = build_synthetic_data(config).to(device)
    init_mode = config.distillation.synthetic.get("init_mode", "real")
    synthetic_init_tokens = None
    initialized_from_checkpoint = _initialize_synthetic_from_checkpoint(
        config,
        synthetic_data,
        model.token_embedding.weight,
        device,
    )
    if initialized_from_checkpoint:
        pass
    elif init_mode == "real_mixture":
        if not hasattr(synthetic_data, "initialize_from_full_logits"):
            raise ValueError(
                "distillation.synthetic.init_mode=real_mixture requires a "
                "synthetic parameterization with initialize_from_full_logits()."
            )
        full_logits = collect_token_mixture_logits_from_dataset(
            datasets["train"],
            num_sequences=config.distillation.synthetic.num_sequences,
            sequence_length=config.distillation.synthetic.sequence_length,
            vocab_size=config.distillation.synthetic.vocab_size,
            eps=config.distillation.synthetic.get("init_mixture_eps", 1e-4),
            offset=config.distillation.synthetic.get("init_mixture_offset", 0),
            max_source_sequences=config.distillation.synthetic.get(
                "init_mixture_max_sequences"
            ),
            device=device,
        )
        synthetic_data.initialize_from_full_logits(
            full_logits,
            embedding_weight=model.token_embedding.weight,
        )
    elif init_mode == "kmeans" and hasattr(synthetic_data, "initialize_from_kmeans"):
        synthetic_data.initialize_from_kmeans(
            embedding_weight=model.token_embedding.weight,
            confidence=config.distillation.synthetic.init_confidence,
        )
    elif init_mode == "random_norm":
        if not hasattr(synthetic_data, "initialize_random_norm"):
            raise ValueError(
                "distillation.synthetic.init_mode=random_norm requires a "
                "synthetic parameterization with initialize_random_norm()."
            )
        if bool(config.distillation.synthetic.get("init_norm_from_real", True)):
            synthetic_init_tokens = _collect_synthetic_init_tokens(
                config,
                dataloaders,
                datasets,
                device,
            )
        synthetic_data.initialize_random_norm(
            embedding_weight=model.token_embedding.weight,
            input_ids=synthetic_init_tokens,
            confidence=config.distillation.synthetic.init_confidence,
        )
    elif config.distillation.synthetic.init_from_real:
        synthetic_init_tokens = _collect_synthetic_init_tokens(
            config,
            dataloaders,
            datasets,
            device,
        )
        synthetic_data.initialize_from_token_ids(
            synthetic_init_tokens,
            confidence=config.distillation.synthetic.init_confidence,
            embedding_weight=model.token_embedding.weight,
        )

    conditional_matcher = None
    conditional_config = config.distillation.get("conditional_matching")
    if conditional_config is not None and bool(conditional_config.get("enabled", False)):
        conditional_matcher = TextGroupMatcher(
            real_dataset=datasets[config.distillation.real_split],
            synthetic_data=synthetic_data,
            config=conditional_config,
            synthetic_init_tokens=synthetic_init_tokens,
            embedding_weight=model.token_embedding.weight,
        )
        print(f"conditional_matching: {conditional_matcher.describe()}")

    synthetic_parameters = list(synthetic_data.parameters())
    if config.distillation.learn_inner_lr:
        log_inner_lr = torch.nn.Parameter(
            torch.log(torch.tensor(config.distillation.inner_lr, device=device))
        )
        optimizer_params = synthetic_parameters + [log_inner_lr]
    else:
        log_inner_lr = torch.log(
            torch.tensor(config.distillation.inner_lr, device=device)
        )
        optimizer_params = synthetic_parameters

    optimizer = instantiate(config.distillation.optimizer, params=optimizer_params)
    objective = config.distillation.get("objective", "one_step")
    expert_checkpoints = _load_expert_checkpoints(config, objective)

    best_tracking_loss = math.inf
    best_state = None
    outer_batches = int(config.distillation.get("outer_batches", 1))
    save_step_checkpoints = bool(
        config.distillation.get("save_step_checkpoints", False)
    )
    progress = trange(1, config.distillation.n_steps + 1, desc="distill")
    for step in progress:
        optimizer.zero_grad()
        model.load_state_dict(initial_state)
        current_inner_lr = resolve_inner_lr(log_inner_lr, config)

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        embedding_weight = params["token_embedding.weight"]

        if objective == "feature_matching":
            if conditional_matcher is None:
                synth_batch_data = synthetic_batch(
                    synthetic_data=synthetic_data,
                    model_params=params,
                    batch_size=config.distillation.synthetic.batch_size,
                    device=device,
                )
                inner_loss = feature_matching_loss(
                    model=model,
                    real_loader=real_loader,
                    synth_batch=synth_batch_data,
                    outer_batches=outer_batches,
                    device=device,
                )
            else:
                inner_loss = grouped_feature_matching_loss(
                    model=model,
                    group_matcher=conditional_matcher,
                    synthetic_data=synthetic_data,
                    model_params=params,
                    outer_batches=outer_batches,
                    device=device,
                )
            backward_loss, entropy_value, target_entropy_value = _regularized_loss(
                inner_loss,
                synthetic_data,
                embedding_weight,
                config,
            )
            backward_loss.backward()
            with torch.no_grad():
                outer_loss = outer_loss_on_real_batches(
                    model=model,
                    params=params,
                    buffers=buffers,
                    real_loader=real_loader,
                    device=device,
                    outer_batches=outer_batches,
                )
        elif objective == "gradient_matching":
            if conditional_matcher is None:
                synth_batch_data = synthetic_batch(
                    synthetic_data=synthetic_data,
                    model_params=params,
                    batch_size=config.distillation.synthetic.batch_size,
                    device=device,
                )
                inner_loss, _ = gradient_matching_loss(
                    model=model,
                    params=params,
                    buffers=buffers,
                    real_loader=real_loader,
                    synth_batch=synth_batch_data,
                    outer_batches=outer_batches,
                    device=device,
                    parameter_names=config.distillation.get(
                        "gradient_match_parameter_names"
                    ),
                )
            else:
                inner_loss, _ = grouped_gradient_matching_loss(
                    model=model,
                    params=params,
                    buffers=buffers,
                    group_matcher=conditional_matcher,
                    synthetic_data=synthetic_data,
                    outer_batches=outer_batches,
                    device=device,
                    parameter_names=config.distillation.get(
                        "gradient_match_parameter_names"
                    ),
                )
            backward_loss, entropy_value, target_entropy_value = _regularized_loss(
                inner_loss,
                synthetic_data,
                embedding_weight,
                config,
            )
            backward_loss.backward()
            with torch.no_grad():
                outer_loss = outer_loss_on_real_batches(
                    model=model,
                    params=params,
                    buffers=buffers,
                    real_loader=real_loader,
                    device=device,
                    outer_batches=outer_batches,
                )
        elif objective == "trajectory_matching":
            inner_loss = trajectory_matching_loss(
                model=model,
                buffers=buffers,
                synthetic_data=synthetic_data,
                expert_checkpoints=expert_checkpoints,
                n_inner_steps=int(config.distillation.get("n_inner_steps", 5)),
                inner_lr=current_inner_lr,
                config=config,
                device=device,
            )
            backward_loss, entropy_value, target_entropy_value = _regularized_loss(
                inner_loss,
                synthetic_data,
                embedding_weight,
                config,
            )
            backward_loss.backward()
            with torch.no_grad():
                outer_loss = outer_loss_on_real_batches(
                    model=model,
                    params=params,
                    buffers=buffers,
                    real_loader=real_loader,
                    device=device,
                    outer_batches=outer_batches,
                )
        elif objective == "one_step":
            updated_params, inner_loss = one_step_parameters(
                model=model,
                params=params,
                buffers=buffers,
                synthetic_data=synthetic_data,
                config=config,
                device=device,
                inner_lr=current_inner_lr,
            )
            outer_loss = outer_loss_on_real_batches(
                model=model,
                params=updated_params,
                buffers=buffers,
                real_loader=real_loader,
                device=device,
                outer_batches=outer_batches,
            )
            backward_loss, entropy_value, target_entropy_value = _regularized_loss(
                outer_loss,
                synthetic_data,
                embedding_weight,
                config,
            )
            backward_loss.backward()
        elif objective == "multi_step":
            updated_params, inner_loss = multi_step_parameters(
                model=model,
                params=params,
                buffers=buffers,
                synthetic_data=synthetic_data,
                config=config,
                device=device,
                inner_lr=current_inner_lr,
            )
            outer_loss = outer_loss_on_real_batches(
                model=model,
                params=updated_params,
                buffers=buffers,
                real_loader=real_loader,
                device=device,
                outer_batches=outer_batches,
            )
            backward_loss, entropy_value, target_entropy_value = _regularized_loss(
                outer_loss,
                synthetic_data,
                embedding_weight,
                config,
            )
            backward_loss.backward()
        elif objective == "multi_step_adamw":
            updated_params, inner_loss = multi_step_adamw_parameters(
                model=model,
                params=params,
                buffers=buffers,
                synthetic_data=synthetic_data,
                config=config,
                device=device,
                inner_lr=current_inner_lr,
            )
            outer_loss = outer_loss_on_real_batches(
                model=model,
                params=updated_params,
                buffers=buffers,
                real_loader=real_loader,
                device=device,
                outer_batches=outer_batches,
            )
            backward_loss, entropy_value, target_entropy_value = _regularized_loss(
                outer_loss,
                synthetic_data,
                embedding_weight,
                config,
            )
            backward_loss.backward()
        else:
            raise ValueError(
                f"Unknown distillation objective {objective!r}. "
                "Expected one of: one_step, feature_matching, "
                "gradient_matching, trajectory_matching, multi_step, "
                "multi_step_adamw."
            )

        if config.distillation.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                optimizer_params,
                config.distillation.max_grad_norm,
            )
        optimizer.step()

        current_outer_loss = outer_loss.item()
        current_outer_ppl = math.exp(min(current_outer_loss, 20.0))
        tracking_loss = (
            inner_loss.item()
            if objective
            in (
                "feature_matching",
                "gradient_matching",
                "trajectory_matching",
            )
            else current_outer_loss
        )
        support_metrics = {}
        if objective in ("one_step", "multi_step_adamw") and should_run_support_search(
            config,
            step,
            synthetic_data,
        ):
            model.load_state_dict(initial_state)
            search_params = dict(model.named_parameters())
            search_buffers = dict(model.named_buffers())
            search_embedding_weight = search_params["token_embedding.weight"]
            support_metrics = hard_em_topk_step(
                model=model,
                params=search_params,
                buffers=search_buffers,
                synthetic_data=synthetic_data,
                real_loader=real_loader,
                config=config,
                device=device,
                inner_lr=current_inner_lr.detach(),
                embedding_weight=search_embedding_weight,
            )
            optimizer.zero_grad(set_to_none=True)

        if tracking_loss < best_tracking_loss:
            best_tracking_loss = tracking_loss
            best_state = {
                "step": step,
                "inner_lr": _scalar(current_inner_lr),
            }
            best_state.update(snapshot_synthetic_data(synthetic_data, embedding_weight))

        progress.set_postfix(
            {
                "outer": f"{current_outer_loss:.4f}",
                "inner": f"{inner_loss.item():.4f}",
                "ppl": f"{current_outer_ppl:.2f}",
            }
        )
        if step % config.distillation.log_step == 0 or step == 1:
            target_entropy_suffix = (
                ""
                if target_entropy_value is None
                else f" target_entropy={target_entropy_value.item():.4f}"
            )
            print(
                f"step={step} "
                f"outer_loss={current_outer_loss:.4f} "
                f"inner_loss={inner_loss.item():.4f} "
                f"inner_lr={_scalar(current_inner_lr):.6f} "
                f"outer_ppl={current_outer_ppl:.2f} "
                f"entropy={entropy_value.item():.4f}"
                f"{target_entropy_suffix}"
            )
        if support_metrics:
            proxy_suffix = ""
            if "support_proxy_before" in support_metrics:
                proxy_suffix = (
                    f" proxy={support_metrics['support_proxy_before']:.4f}"
                    f"->{support_metrics['support_proxy_after']:.4f}"
                )
            print(
                f"support_search step={step} "
                f"replacements={support_metrics['support_replacements']} "
                f"proposals={support_metrics.get('support_proposals', 0)} "
                f"rejected={support_metrics.get('support_rejected', 0)} "
                f"mean_gain={support_metrics['support_mean_gain']:.6f} "
                f"max_gain={support_metrics['support_max_gain']:.6f} "
                f"proxy_outer={support_metrics['support_outer_loss']:.4f}"
                f"{proxy_suffix}"
            )

        metrics_payload = {
            "step": step,
            "objective": objective,
            "outer_loss": current_outer_loss,
            "inner_loss": inner_loss.item(),
            "inner_lr": _scalar(current_inner_lr),
            "outer_ppl": current_outer_ppl,
            "entropy": entropy_value.item(),
            "best_outer_loss": best_tracking_loss,
            "outer_batches": outer_batches,
        }
        if target_entropy_value is not None:
            metrics_payload["target_entropy"] = target_entropy_value.item()
        metrics_payload.update(support_metrics)
        append_metrics(save_dir, metrics_payload)

        if step % config.distillation.save_period == 0:
            save_checkpoint(
                save_dir,
                step,
                synthetic_data,
                config,
                best_tracking_loss,
                current_inner_lr,
                tokenizer=tokenizer,
                embedding_weight=embedding_weight,
            )
            if save_step_checkpoints:
                save_checkpoint(
                    save_dir,
                    step,
                    synthetic_data,
                    config,
                    best_tracking_loss,
                    current_inner_lr,
                    checkpoint_name=f"full_soft_tokens_step{step}.pth",
                    decoded_name=f"decoded_samples_step{step}.txt",
                    plain_name=f"decoded_samples_plain_step{step}.txt",
                    tokenizer=tokenizer,
                    embedding_weight=embedding_weight,
                )

    final_inner_lr = resolve_inner_lr(log_inner_lr, config)
    final_embedding_weight = model.token_embedding.weight
    save_checkpoint(
        save_dir,
        config.distillation.n_steps,
        synthetic_data,
        config,
        best_tracking_loss,
        final_inner_lr,
        tokenizer=tokenizer,
        embedding_weight=final_embedding_weight,
    )
    if best_state is not None:
        save_checkpoint(
            save_dir,
            best_state["step"],
            synthetic_data,
            config,
            best_tracking_loss,
            best_state["inner_lr"],
            checkpoint_name="full_soft_tokens_best.pth",
            decoded_name="decoded_samples_best.txt",
            plain_name="decoded_samples_plain_best.txt",
            synthetic_logits=best_state.get("synthetic_logits"),
            input_probs=best_state.get("input_probs"),
            target_probs=best_state.get("target_probs"),
            hard_tokens=best_state["hard_tokens"],
            synthetic_state_dict=best_state["synthetic_state_dict"],
            tokenizer=tokenizer,
            embedding_weight=final_embedding_weight,
        )
    print(
        "Saved distilled "
        f"{config.distillation.synthetic.get('parameterization', 'full')} "
        f"synthetic-token dataset to: {save_dir}"
    )

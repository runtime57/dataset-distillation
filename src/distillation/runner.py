import math
from copy import deepcopy

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
    collect_grouped_initial_token_probs,
    collect_initial_tokens,
    synthetic_batch,
)
from src.distillation.factory import build_synthetic_data
from src.distillation.losses import entropy
from src.distillation.objectives import (
    feature_matching_loss,
    gradient_matching_loss,
    one_step_parameters,
    outer_loss_on_real_batches,
    trajectory_matching_loss,
)
from src.distillation.utils import resolve_inner_lr
from src.tokenizers import build_tokenizer_from_dataset_config
from src.utils.init_utils import resolve_device, set_random_seed
from src.utils.io_utils import ROOT_PATH


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _maybe_compile_model(model, config):
    compile_enabled = bool(config.distillation.get("compile_enabled", False))
    if not compile_enabled:
        return model

    if not hasattr(torch, "compile"):
        print("torch.compile is unavailable in this PyTorch build; skipping compile.")
        return model

    compile_mode = config.distillation.get("compile_mode")
    compile_dynamic = bool(config.distillation.get("compile_dynamic", False))
    print(
        "Compiling distillation model with torch.compile "
        f"(mode={compile_mode!r}, dynamic={compile_dynamic})"
    )
    return torch.compile(
        model,
        mode=compile_mode,
        dynamic=compile_dynamic,
    )


def _plain_config(config_node):
    return OmegaConf.to_container(config_node, resolve=True)


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
        raise ValueError(
            "Expert trajectory must contain at least two checkpoints."
        )

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

    momentum = float(
        metadata.get(
            "momentum",
            metadata.get("optimizer", {}).get("momentum", 0.0),
        )
    )
    if momentum != 0.0:
        raise ValueError(
            "TM currently supports only momentum-free expert trajectories. "
            f"Got expert momentum={momentum}."
        )

    if not bool(config.distillation.get("learn_inner_lr", False)):
        expert_lr = float(metadata.get("lr", metadata.get("optimizer", {}).get("lr")))
        inner_lr = float(config.distillation.inner_lr)
        if abs(expert_lr - inner_lr) > 1e-12:
            raise ValueError(
                "Expert lr must match distillation.inner_lr when "
                "learn_inner_lr=false. "
                f"Got expert lr={expert_lr}, inner_lr={inner_lr}."
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
        f"(step {metadata['save_period']}, lr {metadata['lr']}) from {expert_path}"
    )
    return expert_checkpoints


def _alternating_config(config, synthetic_data):
    alternating = config.distillation.synthetic.get("alternating")
    if not alternating or not bool(alternating.get("enabled", False)):
        return None
    if not hasattr(synthetic_data, "mixture_logits"):
        raise ValueError(
            "Alternating optimization requires synthetic_data to expose "
            "'mixture_logits'."
        )
    component_attr = None
    component_phase = None
    if hasattr(synthetic_data, "anchor_logits"):
        component_attr = "anchor_logits"
        component_phase = "anchors"
    elif hasattr(synthetic_data, "concept_vectors"):
        component_attr = "concept_vectors"
        component_phase = "concepts"
    if component_attr is None:
        raise ValueError(
            "Alternating optimization currently supports datasets exposing "
            "'anchor_logits' or 'concept_vectors'."
        )
    mixture_steps = int(alternating.get("mixture_steps", 1))
    component_steps = int(
        alternating.get(
            "component_steps",
            alternating.get("anchor_steps", alternating.get("concept_steps", 1)),
        )
    )
    mode = str(alternating.get("mode", "cyclic"))
    if mixture_steps < 1 or component_steps < 1:
        raise ValueError(
            "Alternating optimization expects positive mixture_steps and "
            f"component_steps, got {mixture_steps} and {component_steps}."
        )
    if mode not in {"cyclic", "staged"}:
        raise ValueError(
            f"Unknown alternating optimization mode {mode!r}. "
            "Expected 'cyclic' or 'staged'."
        )
    return {
        "mode": mode,
        "mixture_steps": mixture_steps,
        "component_steps": component_steps,
        "component_attr": component_attr,
        "component_phase": component_phase,
    }


def _alternating_phase(alternating, step):
    if alternating is None:
        return "joint"
    if alternating["mode"] == "staged":
        if step <= alternating["mixture_steps"]:
            return "mixture"
        return alternating["component_phase"]
    cycle = alternating["mixture_steps"] + alternating["component_steps"]
    cycle_step = (step - 1) % cycle
    if cycle_step < alternating["mixture_steps"]:
        return "mixture"
    return alternating["component_phase"]


def _set_trainable_component_phase(synthetic_data, alternating, phase):
    for parameter in synthetic_data.parameters():
        parameter.requires_grad_(True)
    if phase == "joint":
        return
    synthetic_data.mixture_logits.requires_grad_(phase == "mixture")
    component_attr = alternating["component_attr"] if alternating is not None else None
    if component_attr is None:
        return
    getattr(synthetic_data, component_attr).requires_grad_(
        phase == alternating["component_phase"]
    )


def _trainable_parameters(module):
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _maybe_initialize_from_checkpoint(config, synthetic_data):
    checkpoint_path = config.distillation.synthetic.get("init_checkpoint_path")
    if not checkpoint_path:
        return
    payload = torch.load(ROOT_PATH / checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "synthetic_state_dict" not in payload:
        raise ValueError(
            "Synthetic init checkpoint must be a distillation checkpoint with "
            "'synthetic_state_dict'."
        )
    synthetic_data.load_state_dict(payload["synthetic_state_dict"])
    print(f"Initialized synthetic data from checkpoint: {ROOT_PATH / checkpoint_path}")


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
    model = _maybe_compile_model(model, config)
    model.train()
    initial_state = deepcopy(model.state_dict())

    synthetic_data = build_synthetic_data(config).to(device)
    init_mode = config.distillation.synthetic.get("init_mode", "real")
    if init_mode == "kmeans" and hasattr(synthetic_data, "initialize_from_kmeans"):
        synthetic_data.initialize_from_kmeans(
            embedding_weight=model.token_embedding.weight,
            confidence=config.distillation.synthetic.init_confidence,
        )
    elif init_mode in {"grouped_real", "real_grouped"}:
        grouped_probs = collect_grouped_initial_token_probs(
            dataloader=dataloaders["train"],
            num_sequences=config.distillation.synthetic.num_sequences,
            group_size=int(config.distillation.synthetic.get("init_group_size", 1)),
            vocab_size=config.distillation.synthetic.vocab_size,
            device=device,
            probability_floor=float(
                config.distillation.synthetic.get("init_probability_floor", 0.01)
            ),
        )
        synthetic_data.initialize_from_token_probs(
            grouped_probs,
            confidence=config.distillation.synthetic.init_confidence,
            embedding_weight=model.token_embedding.weight,
        )
    elif config.distillation.synthetic.init_from_real:
        init_tokens = collect_initial_tokens(
            dataloaders["train"],
            num_sequences=config.distillation.synthetic.num_sequences,
            device=device,
        )
        synthetic_data.initialize_from_token_ids(
            init_tokens,
            confidence=config.distillation.synthetic.init_confidence,
            embedding_weight=model.token_embedding.weight,
        )
    _maybe_initialize_from_checkpoint(config, synthetic_data)

    if config.distillation.learn_inner_lr:
        log_inner_lr = torch.nn.Parameter(
            torch.log(torch.tensor(config.distillation.inner_lr, device=device))
        )
        optimizer_params = list(synthetic_data.parameters()) + [log_inner_lr]
    else:
        log_inner_lr = torch.log(
            torch.tensor(config.distillation.inner_lr, device=device)
        )
        optimizer_params = synthetic_data.parameters()

    optimizer = instantiate(config.distillation.optimizer, params=optimizer_params)
    objective = config.distillation.get("objective", "one_step")
    expert_checkpoints = _load_expert_checkpoints(config, objective)
    alternating = _alternating_config(config, synthetic_data)

    best_tracking_loss = math.inf
    best_state = None
    best_tracking_step = None
    outer_batches = int(config.distillation.get("outer_batches", 1))
    save_step_checkpoints = bool(
        config.distillation.get("save_step_checkpoints", False)
    )
    progress = trange(1, config.distillation.n_steps + 1, desc="distill")
    for step in progress:
        phase = _alternating_phase(alternating, step)
        _set_trainable_component_phase(synthetic_data, alternating, phase)
        optimizer.zero_grad()
        model.load_state_dict(initial_state)
        current_inner_lr = resolve_inner_lr(log_inner_lr, config)

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        embedding_weight = params["token_embedding.weight"]

        if objective == "feature_matching":
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
            synth_probs = synthetic_data.token_probs(embedding_weight=embedding_weight)
            entropy_value = entropy(synth_probs)
            (inner_loss - config.distillation.entropy_weight * entropy_value).backward()
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
            )
            synth_probs = synthetic_data.token_probs(embedding_weight=embedding_weight)
            entropy_value = entropy(synth_probs)
            (inner_loss - config.distillation.entropy_weight * entropy_value).backward()
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
                inner_lr=_scalar(current_inner_lr),
                config=config,
                device=device,
            )
            synth_probs = synthetic_data.token_probs(embedding_weight=embedding_weight)
            entropy_value = entropy(synth_probs)
            (inner_loss - config.distillation.entropy_weight * entropy_value).backward()
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
            synth_probs = synthetic_data.token_probs(embedding_weight=embedding_weight)
            entropy_value = entropy(synth_probs)
            (outer_loss - config.distillation.entropy_weight * entropy_value).backward()
        else:
            raise ValueError(
                f"Unknown distillation objective {objective!r}. "
                "Expected one of: one_step, feature_matching, "
                "gradient_matching, trajectory_matching."
            )

        if config.distillation.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                _trainable_parameters(synthetic_data),
                config.distillation.max_grad_norm,
            )
        optimizer.step()

        current_outer_loss = outer_loss.item()
        current_outer_ppl = math.exp(min(current_outer_loss, 20.0))
        tracking_loss = (
            inner_loss.item()
            if objective in (
                "feature_matching",
                "gradient_matching",
                "trajectory_matching",
            )
            else current_outer_loss
        )
        tracking_metric_name = (
            "inner_loss"
            if objective in (
                "feature_matching",
                "gradient_matching",
                "trajectory_matching",
            )
            else "outer_loss"
        )
        if tracking_loss < best_tracking_loss:
            best_tracking_loss = tracking_loss
            best_tracking_step = step
            best_state = {
                "step": step,
                "inner_lr": _scalar(current_inner_lr),
            }
            best_state.update(snapshot_synthetic_data(synthetic_data, embedding_weight))

        progress.set_postfix(
            {
                "phase": phase,
                "outer": f"{current_outer_loss:.4f}",
                tracking_metric_name: f"{tracking_loss:.4f}",
                "ppl": f"{current_outer_ppl:.2f}",
            }
        )
        if step % config.distillation.log_step == 0 or step == 1:
            print(
                f"step={step} "
                f"phase={phase} "
                f"outer_loss={current_outer_loss:.4f} "
                f"inner_loss={inner_loss.item():.4f} "
                f"tracking_metric={tracking_metric_name} "
                f"tracking_value={tracking_loss:.4f} "
                f"best_tracking_loss={best_tracking_loss:.4f} "
                f"best_tracking_step={best_tracking_step} "
                f"inner_lr={_scalar(current_inner_lr):.6f} "
                f"outer_ppl={current_outer_ppl:.2f} "
                f"entropy={entropy_value.item():.4f}"
            )

        append_metrics(
            save_dir,
            {
                "step": step,
                "phase": phase,
                "objective": objective,
                "outer_loss": current_outer_loss,
                "inner_loss": inner_loss.item(),
                "tracking_metric_name": tracking_metric_name,
                "tracking_value": tracking_loss,
                "best_tracking_loss": best_tracking_loss,
                "best_tracking_step": best_tracking_step,
                "inner_lr": _scalar(current_inner_lr),
                "outer_ppl": current_outer_ppl,
                "entropy": entropy_value.item(),
                "best_outer_loss": current_outer_loss,
                "outer_batches": outer_batches,
            },
        )

        if step % config.distillation.save_period == 0:
            save_checkpoint(
                save_dir,
                step,
                synthetic_data,
                config,
                current_outer_loss,
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
                    current_outer_loss,
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
        current_outer_loss,
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

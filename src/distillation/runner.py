import math
from copy import deepcopy

import torch
from hydra.utils import instantiate
from tqdm.auto import trange

from src.datasets.data_utils import build_dataloaders, build_datasets, inf_loop
from src.distillation.checkpoints import (
    append_metrics,
    prepare_save_dir,
    save_checkpoint,
    snapshot_synthetic_data,
)
from src.distillation.data import collect_initial_tokens, synthetic_batch
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


def _load_expert_checkpoints(config, objective):
    if objective != "trajectory_matching":
        return None
    expert_path = ROOT_PATH / config.distillation.expert_trajectory_path
    expert_checkpoints = torch.load(expert_path, map_location="cpu")
    print(f"Loaded {len(expert_checkpoints)} expert checkpoints from {expert_path}")
    return expert_checkpoints


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
    if init_mode == "kmeans" and hasattr(synthetic_data, "initialize_from_kmeans"):
        synthetic_data.initialize_from_kmeans(
            embedding_weight=model.token_embedding.weight,
            confidence=config.distillation.synthetic.init_confidence,
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
                synthetic_data.parameters(),
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
            print(
                f"step={step} "
                f"outer_loss={current_outer_loss:.4f} "
                f"inner_loss={inner_loss.item():.4f} "
                f"inner_lr={_scalar(current_inner_lr):.6f} "
                f"outer_ppl={current_outer_ppl:.2f} "
                f"entropy={entropy_value.item():.4f}"
            )

        append_metrics(
            save_dir,
            {
                "step": step,
                "objective": objective,
                "outer_loss": current_outer_loss,
                "inner_loss": inner_loss.item(),
                "inner_lr": _scalar(current_inner_lr),
                "outer_ppl": current_outer_ppl,
                "entropy": entropy_value.item(),
                "best_outer_loss": best_tracking_loss,
                "outer_batches": outer_batches,
            },
        )

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

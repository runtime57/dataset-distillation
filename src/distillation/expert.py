import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import trange

from src.datasets.data_utils import (
    build_dataloaders,
    build_datasets,
    inf_loop,
)
from src.distillation.losses import hard_lm_loss
from src.distillation.utils import move_batch_to_device
from src.utils.init_utils import resolve_device, set_random_seed
from src.utils.io_utils import ROOT_PATH


def _state_dict_cpu(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _trajectory_metadata(config, momentum):
    return {
        "format_version": 2,
        "model": OmegaConf.to_container(config.model, resolve=True),
        "datasets": OmegaConf.to_container(config.datasets, resolve=True),
        "dataloader": OmegaConf.to_container(config.dataloader, resolve=True),
        "optimizer": {
            "name": "SGD",
            "lr": float(config.expert.lr),
            "momentum": float(momentum),
        },
        "lr": float(config.expert.lr),
        "momentum": float(momentum),
        "save_period": int(config.expert.save_period),
        "n_steps": int(config.expert.n_steps),
        "seed": int(config.expert.seed),
    }


def run_expert_trajectory(config):
    set_random_seed(config.expert.seed)
    device = resolve_device(config.expert.device)
    save_period = int(config.expert.save_period)
    n_steps = int(config.expert.n_steps)
    if save_period <= 0:
        raise ValueError(f"expert.save_period must be positive, got {save_period}.")
    if n_steps % save_period != 0:
        raise ValueError(
            "expert.n_steps must be divisible by expert.save_period for "
            f"trajectory matching, got n_steps={n_steps}, "
            f"save_period={save_period}."
        )

    datasets = build_datasets(config, partitions=("train",))
    dataloaders = build_dataloaders(config, datasets)
    real_loader = inf_loop(dataloaders["train"])

    model = instantiate(config.model).to(device)
    momentum = float(config.expert.get("momentum", 0.0))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.expert.lr,
        momentum=momentum,
    )

    save_path = ROOT_PATH / config.expert.save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoints = [{"step": 0, "state_dict": _state_dict_cpu(model)}]
    model.train()
    for step in trange(1, n_steps + 1, desc="expert trajectory"):
        batch = move_batch_to_device(next(real_loader), device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        loss = hard_lm_loss(
            outputs["logits"],
            batch["labels"],
            attention_mask=batch.get("attention_mask"),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % save_period == 0:
            checkpoints.append({"step": step, "state_dict": _state_dict_cpu(model)})

    payload = {
        "metadata": _trajectory_metadata(config, momentum),
        "checkpoints": checkpoints,
    }
    output_path = save_path / "expert_checkpoints.pth"
    torch.save(payload, output_path)
    print(
        f"Saved {len(checkpoints)} expert checkpoints "
        f"(0..{n_steps} step {save_period}) -> {output_path}"
    )

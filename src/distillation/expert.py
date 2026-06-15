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
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _optimizer_state_cpu(optimizer, model):
    parameter_names = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    state = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = parameter_names.get(id(parameter))
            if name is None:
                continue
            parameter_state = optimizer.state.get(parameter)
            if not parameter_state:
                continue
            state[name] = {}
            for key, value in parameter_state.items():
                if torch.is_tensor(value):
                    state[name][key] = value.detach().cpu().clone()
                else:
                    state[name][key] = value
    return state


def _expert_optimizer_settings(config):
    name = str(config.expert.get("optimizer", "sgd")).lower()
    lr = float(config.expert.lr)
    if name == "sgd":
        momentum = float(config.expert.get("momentum", 0.0))
        return {
            "name": "sgd",
            "lr": lr,
            "momentum": momentum,
        }
    if name == "adamw":
        beta1 = float(config.expert.get("beta1", 0.9))
        beta2 = float(config.expert.get("beta2", 0.999))
        eps = float(config.expert.get("eps", 1e-8))
        weight_decay = float(config.expert.get("weight_decay", 0.0))
        return {
            "name": "adamw",
            "lr": lr,
            "betas": (beta1, beta2),
            "eps": eps,
            "weight_decay": weight_decay,
        }
    raise ValueError(f"Unsupported expert.optimizer={name!r}.")


def _build_optimizer(model, settings):
    if settings["name"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=settings["lr"],
            momentum=settings["momentum"],
        )
    if settings["name"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=settings["lr"],
            betas=tuple(settings["betas"]),
            eps=settings["eps"],
            weight_decay=settings["weight_decay"],
        )
    raise ValueError(f"Unsupported expert optimizer={settings['name']!r}.")


def _trajectory_metadata(config, optimizer_settings):
    return {
        "format_version": 2,
        "model": OmegaConf.to_container(config.model, resolve=True),
        "datasets": OmegaConf.to_container(config.datasets, resolve=True),
        "dataloader": OmegaConf.to_container(config.dataloader, resolve=True),
        "optimizer": optimizer_settings,
        "lr": float(config.expert.lr),
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
    optimizer_settings = _expert_optimizer_settings(config)
    optimizer = _build_optimizer(model, optimizer_settings)

    save_path = ROOT_PATH / config.expert.save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoints = [
        {
            "step": 0,
            "state_dict": _state_dict_cpu(model),
            "optimizer_state": _optimizer_state_cpu(optimizer, model),
        }
    ]
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
            checkpoints.append(
                {
                    "step": step,
                    "state_dict": _state_dict_cpu(model),
                    "optimizer_state": _optimizer_state_cpu(optimizer, model),
                }
            )

    payload = {
        "metadata": _trajectory_metadata(config, optimizer_settings),
        "checkpoints": checkpoints,
    }
    output_path = save_path / "expert_checkpoints.pth"
    torch.save(payload, output_path)
    print(
        f"Saved {len(checkpoints)} expert checkpoints "
        f"(0..{n_steps} step {save_period}) -> {output_path}"
    )

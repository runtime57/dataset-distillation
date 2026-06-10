import json
import shutil

from omegaconf import OmegaConf
import torch

from src.distillation.data import decode_token_ids
from src.distillation.parameterizations import FullSoftTokenDataset
from src.utils.io_utils import ROOT_PATH


def save_checkpoint(
    save_dir,
    step,
    synthetic_data,
    config,
    best_outer_loss=None,
    best_tracking_loss=None,
    inner_lr=None,
    checkpoint_name="full_soft_tokens.pth",
    decoded_name="decoded_samples.txt",
    plain_name="decoded_samples_plain.txt",
    synthetic_logits=None,
    target_probs=None,
    hard_tokens=None,
    synthetic_state_dict=None,
    tokenizer=None,
    embedding_weight=None,
):
    with torch.no_grad():
        if (
            synthetic_logits is None
            and target_probs is None
            and isinstance(synthetic_data, FullSoftTokenDataset)
        ):
            synthetic_logits = synthetic_data.logits.detach().cpu()
        if synthetic_logits is None and target_probs is None:
            target_probs = synthetic_data.token_probs(
                embedding_weight=embedding_weight,
            ).detach().cpu()
        if hard_tokens is None:
            hard_tokens = synthetic_data.hard_tokens(embedding_weight).detach().cpu()
        if synthetic_state_dict is None:
            synthetic_state_dict = {
                key: value.detach().cpu()
                for key, value in synthetic_data.state_dict().items()
            }

    checkpoint = {
        "step": step,
        "hard_tokens": hard_tokens,
        "synthetic_parameter_count": synthetic_data.parameter_count(),
        "synthetic_state_dict": synthetic_state_dict,
        "best_outer_loss": best_outer_loss,
        "best_tracking_loss": best_tracking_loss,
        "inner_lr": (
            None
            if inner_lr is None
            else float(inner_lr.detach().cpu())
            if torch.is_tensor(inner_lr)
            else float(inner_lr)
        ),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    if synthetic_logits is not None:
        checkpoint["synthetic_logits"] = synthetic_logits
    if target_probs is not None:
        checkpoint["target_probs"] = target_probs
    torch.save(checkpoint, save_dir / checkpoint_name)

    decoded_samples = decode_token_ids(hard_tokens, tokenizer)
    decoded_path = save_dir / decoded_name
    with decoded_path.open("w", encoding="utf-8") as file:
        for index, text in enumerate(decoded_samples):
            file.write(f"===== sample {index} =====\n")
            file.write(text)
            file.write("\n\n")

    plain_path = save_dir / plain_name
    with plain_path.open("w", encoding="utf-8") as file:
        for text in decoded_samples:
            file.write(text.replace("\n", " ").strip())
            file.write("\n")


def append_metrics(save_dir, metrics):
    with (save_dir / "distill_metrics.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics, sort_keys=True) + "\n")


def prepare_save_dir(config):
    save_dir = ROOT_PATH / config.distillation.save_dir / config.distillation.run_name
    if save_dir.exists():
        if config.distillation.override:
            shutil.rmtree(save_dir)
        else:
            raise ValueError(
                f"Save directory already exists: {save_dir}. "
                "Set distillation.override=true to overwrite it."
            )
    save_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(config, save_dir / "config.yaml")
    return save_dir


def snapshot_synthetic_data(synthetic_data, embedding_weight):
    with torch.no_grad():
        snapshot = {
            "hard_tokens": (
                synthetic_data.hard_tokens(embedding_weight).detach().cpu().clone()
            ),
            "synthetic_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in synthetic_data.state_dict().items()
            },
        }
        if isinstance(synthetic_data, FullSoftTokenDataset):
            snapshot["synthetic_logits"] = synthetic_data.logits.detach().cpu().clone()
        else:
            snapshot["target_probs"] = (
                synthetic_data.token_probs(embedding_weight=embedding_weight)
                .detach()
                .cpu()
                .clone()
            )
    return snapshot

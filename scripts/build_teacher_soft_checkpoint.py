import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.datasets import TinyStoriesLocalBPEDataset
from src.utils.init_utils import resolve_device, set_random_seed
from src.utils.io_utils import ROOT_PATH


def _resolve(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT_PATH / path
    return path


def _load_teacher(checkpoint_path, device):
    checkpoint = torch.load(
        _resolve(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    model = instantiate(config.model).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, OmegaConf.to_container(config, resolve=True)


def _build_dataset(args, sequence_length):
    return TinyStoriesLocalBPEDataset(
        tokenizer_path=args.tokenizer_path,
        sequence_length=sequence_length,
        skip_texts=args.skip_texts,
        max_texts=args.max_texts,
        max_sequences=args.candidate_sequences,
        local_text_path=args.local_text_path,
        data_dir=args.data_dir,
        use_cache=True,
    )


@torch.no_grad()
def _score_sequences(model, input_ids, batch_size, device, mode):
    if mode in ("first", "random"):
        return None

    scores = []
    for start in tqdm(range(0, input_ids.shape[0], batch_size), desc="score"):
        batch = input_ids[start : start + batch_size].to(device)
        attention_mask = torch.ones_like(batch)
        logits = model(input_ids=batch, attention_mask=attention_mask)["logits"]
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        if mode == "loss_top":
            per_token = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="none",
            ).view_as(shift_labels)
            score = per_token.mean(dim=1)
        elif mode == "entropy_top":
            probs = torch.softmax(shift_logits, dim=-1)
            score = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=1)
        else:
            raise ValueError(f"Unknown selection mode: {mode}")
        scores.append(score.cpu())
    return torch.cat(scores)


def _select_input_ids(args, model, input_ids, device):
    if input_ids.shape[0] < args.num_sequences:
        raise ValueError(
            f"Need at least {args.num_sequences} candidate sequences, "
            f"got {input_ids.shape[0]}."
        )
    if args.selection == "first":
        indices = torch.arange(args.num_sequences)
    elif args.selection == "random":
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(input_ids.shape[0], generator=generator)[
            : args.num_sequences
        ]
    else:
        scores = _score_sequences(
            model=model,
            input_ids=input_ids,
            batch_size=args.batch_size,
            device=device,
            mode=args.selection,
        )
        indices = scores.topk(args.num_sequences).indices
    return input_ids[indices].contiguous(), indices


@torch.no_grad()
def _teacher_target_probs(
    model,
    input_ids,
    batch_size,
    device,
    temperature,
    hard_label_weight,
    dtype,
):
    chunks = []
    for start in tqdm(range(0, input_ids.shape[0], batch_size), desc="teacher"):
        batch = input_ids[start : start + batch_size].to(device)
        attention_mask = torch.ones_like(batch)
        logits = model(input_ids=batch, attention_mask=attention_mask)["logits"]
        shifted_probs = torch.softmax(logits[:, :-1] / temperature, dim=-1)
        if hard_label_weight > 0:
            shifted_probs = shifted_probs * (1.0 - hard_label_weight)
            hard_weight = torch.full(
                (*batch[:, 1:].shape, 1),
                hard_label_weight,
                device=device,
                dtype=shifted_probs.dtype,
            )
            shifted_probs.scatter_add_(2, batch[:, 1:].unsqueeze(-1), hard_weight)
        target_probs = torch.zeros(
            batch.shape[0],
            batch.shape[1],
            logits.shape[-1],
            device=device,
            dtype=shifted_probs.dtype,
        )
        target_probs[:, 0].scatter_(1, batch[:, :1], 1.0)
        target_probs[:, 1:] = shifted_probs
        chunks.append(target_probs.to("cpu", dtype=dtype))
    return torch.cat(chunks, dim=0).contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-sequences", type=int, default=256)
    parser.add_argument("--candidate-sequences", type=int, default=256)
    parser.add_argument(
        "--selection",
        choices=("first", "random", "loss_top", "entropy_top"),
        default="first",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--hard-label-weight",
        type=float,
        default=0.0,
        help="Blend weight for one-hot next-token labels in [0, 1].",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--local-text-path",
        default="data/tinystories_train_15k.txt",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="artifacts/tokenizers/tinystories_bpe_1024/tokenizer.json",
    )
    parser.add_argument("--data-dir", default="data/tinystories_local_bpe_1024")
    parser.add_argument("--skip-texts", type=int, default=0)
    parser.add_argument("--max-texts", type=int, default=10000)
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    args = parser.parse_args()
    if not 0.0 <= args.hard_label_weight <= 1.0:
        raise ValueError("--hard-label-weight must be in [0, 1].")

    set_random_seed(args.seed)
    device = resolve_device(args.device)
    teacher, teacher_config = _load_teacher(args.teacher_checkpoint, device)
    sequence_length = int(teacher_config["model"]["max_seq_len"])
    dataset = _build_dataset(args, sequence_length)
    selected_ids, selected_indices = _select_input_ids(
        args,
        teacher,
        dataset.input_ids,
        device,
    )
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    target_probs = _teacher_target_probs(
        model=teacher,
        input_ids=selected_ids,
        batch_size=args.batch_size,
        device=device,
        temperature=args.temperature,
        hard_label_weight=args.hard_label_weight,
        dtype=dtype,
    )

    output_path = _resolve(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": 0,
        "input_ids": selected_ids.cpu(),
        "hard_tokens": selected_ids.cpu(),
        "target_probs": target_probs,
        "teacher_checkpoint": str(_resolve(args.teacher_checkpoint)),
        "teacher_temperature": args.temperature,
        "hard_label_weight": args.hard_label_weight,
        "selection": args.selection,
        "selected_indices": selected_indices.cpu(),
        "synthetic_parameter_count": int(target_probs.numel()),
        "config": teacher_config,
    }
    torch.save(checkpoint, output_path)
    print(f"Saved teacher-soft checkpoint to {output_path}")
    print(f"target_probs={tuple(target_probs.shape)} dtype={target_probs.dtype}")
    print(f"hard_label_weight={args.hard_label_weight}")


if __name__ == "__main__":
    main()

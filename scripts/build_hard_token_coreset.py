import argparse
import sys
from pathlib import Path

import torch
from tqdm.auto import trange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import TinyStoriesLocalBPEDataset
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH


def _resolve(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT_PATH / path
    return path


def _build_dataset(args):
    return TinyStoriesLocalBPEDataset(
        tokenizer_path=args.tokenizer_path,
        sequence_length=args.sequence_length,
        skip_texts=args.skip_texts,
        max_texts=args.max_texts,
        max_sequences=args.candidate_sequences,
        local_text_path=args.local_text_path,
        data_dir=args.data_dir,
        use_cache=True,
    )


def _scatter_counts(indices, num_bins):
    num_sequences, width = indices.shape
    counts = torch.zeros(num_sequences, num_bins, dtype=torch.float32)
    rows = torch.arange(num_sequences).unsqueeze(1).expand_as(indices)
    counts.index_put_(
        (rows.reshape(-1), indices.reshape(-1)),
        torch.ones(indices.numel(), dtype=torch.float32),
        accumulate=True,
    )
    return counts / float(width)


def _features(
    input_ids,
    vocab_size,
    bigram_bins,
    trigram_bins,
    unigram_weight,
    bigram_weight,
    trigram_weight,
    position_bins,
    position_weight,
    position_bigram_bins,
    position_bigram_weight,
):
    unigram = _scatter_counts(input_ids, vocab_size) * float(unigram_weight)
    parts = [unigram]

    if bigram_bins > 0 and bigram_weight != 0.0:
        left = input_ids[:, :-1].long()
        right = input_ids[:, 1:].long()
        bigram_ids = (left * 1009 + right) % int(bigram_bins)
        bigram = _scatter_counts(bigram_ids, int(bigram_bins)) * float(bigram_weight)
        parts.append(bigram)

    if trigram_bins > 0 and trigram_weight != 0.0:
        first = input_ids[:, :-2].long()
        second = input_ids[:, 1:-1].long()
        third = input_ids[:, 2:].long()
        trigram_ids = (first * 1_000_003 + second * 1009 + third) % int(
            trigram_bins
        )
        trigram = _scatter_counts(trigram_ids, int(trigram_bins)) * float(
            trigram_weight
        )
        parts.append(trigram)

    if position_bins > 0 and position_weight != 0.0:
        positions = torch.arange(input_ids.shape[1], dtype=torch.long)
        position_ids = (positions.unsqueeze(0) * 1009 + input_ids.long()) % int(
            position_bins
        )
        position_counts = _scatter_counts(position_ids, int(position_bins))
        parts.append(position_counts * float(position_weight))

    if position_bigram_bins > 0 and position_bigram_weight != 0.0:
        positions = torch.arange(input_ids.shape[1] - 1, dtype=torch.long)
        left = input_ids[:, :-1].long()
        right = input_ids[:, 1:].long()
        position_bigram_ids = (
            positions.unsqueeze(0) * 1_000_003 + left * 1009 + right
        ) % int(position_bigram_bins)
        position_bigram = _scatter_counts(
            position_bigram_ids,
            int(position_bigram_bins),
        )
        parts.append(position_bigram * float(position_bigram_weight))

    return torch.cat(parts, dim=1).contiguous()


def _first(input_ids, num_sequences, args):
    return torch.arange(num_sequences)


def _random(input_ids, num_sequences, args):
    generator = torch.Generator().manual_seed(args.seed)
    return torch.randperm(input_ids.shape[0], generator=generator)[:num_sequences]


def _rare_top(input_ids, num_sequences, args):
    counts = torch.bincount(input_ids.reshape(-1), minlength=args.vocab_size).float()
    inv_freq = 1.0 / counts.clamp_min(1.0).sqrt()
    scores = inv_freq[input_ids].mean(dim=1)
    return scores.topk(num_sequences).indices


def _herding(input_ids, num_sequences, args):
    feats = _features(
        input_ids=input_ids,
        vocab_size=args.vocab_size,
        bigram_bins=args.bigram_bins,
        trigram_bins=args.trigram_bins,
        unigram_weight=args.unigram_weight,
        bigram_weight=args.bigram_weight,
        trigram_weight=args.trigram_weight,
        position_bins=args.position_bins,
        position_weight=args.position_weight,
        position_bigram_bins=args.position_bigram_bins,
        position_bigram_weight=args.position_bigram_weight,
    )
    target = feats.mean(dim=0)
    norms = (feats * feats).sum(dim=1)
    selected = []
    selected_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool)
    selected_sum = torch.zeros_like(target)

    for step in trange(num_sequences, desc="herding"):
        residual = target * float(step + 1) - selected_sum
        scores = 2.0 * (feats @ residual) - norms
        scores[selected_mask] = -torch.inf
        index = int(scores.argmax().item())
        selected.append(index)
        selected_mask[index] = True
        selected_sum += feats[index]

    selected_indices = torch.tensor(selected, dtype=torch.long)
    if args.local_search_swaps > 0:
        selected_indices = _local_search_swaps(
            feats=feats,
            target=target,
            norms=norms,
            selected_indices=selected_indices,
            num_swaps=args.local_search_swaps,
            min_improvement=args.local_search_min_improvement,
        )

    return selected_indices


def _local_search_swaps(
    feats,
    target,
    norms,
    selected_indices,
    num_swaps,
    min_improvement,
):
    selected_indices = selected_indices.clone()
    selected_mask = torch.zeros(feats.shape[0], dtype=torch.bool)
    selected_mask[selected_indices] = True
    target_sum = target * float(selected_indices.numel())

    for _ in trange(num_swaps, desc="local-search"):
        selected_feats = feats[selected_indices]
        selected_norms = norms[selected_indices]
        residual = selected_feats.sum(dim=0) - target_sum
        feats_residual = feats @ residual
        selected_residual = selected_feats @ residual
        cross = feats @ selected_feats.T
        deltas = (
            norms[:, None]
            + selected_norms[None, :]
            - 2.0 * cross
            + 2.0 * feats_residual[:, None]
            - 2.0 * selected_residual[None, :]
        )
        deltas[selected_mask] = torch.inf
        best_delta, flat_index = deltas.reshape(-1).min(dim=0)
        improvement = -float(best_delta.item())
        if improvement <= float(min_improvement):
            break

        candidate_index = int(flat_index.item() // selected_indices.numel())
        selected_position = int(flat_index.item() % selected_indices.numel())
        selected_mask[int(selected_indices[selected_position].item())] = False
        selected_indices[selected_position] = candidate_index
        selected_mask[candidate_index] = True

    return selected_indices


def _select(input_ids, args):
    if input_ids.shape[0] < args.num_sequences:
        raise ValueError(
            f"Need at least {args.num_sequences} candidates, got {input_ids.shape[0]}."
        )

    selectors = {
        "first": _first,
        "random": _random,
        "rare_top": _rare_top,
        "ngram_herding": _herding,
    }
    return selectors[args.selection](input_ids, args.num_sequences, args)


def _one_hot_probs(input_ids, vocab_size, dtype):
    probs = torch.zeros(
        input_ids.shape[0],
        input_ids.shape[1],
        vocab_size,
        dtype=dtype,
    )
    probs.scatter_(2, input_ids.unsqueeze(-1), 1.0)
    return probs.contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--selection",
        choices=("first", "random", "rare_top", "ngram_herding"),
        default="ngram_herding",
    )
    parser.add_argument("--num-sequences", type=int, default=256)
    parser.add_argument("--candidate-sequences", type=int, default=10000)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--bigram-bins", type=int, default=4096)
    parser.add_argument("--trigram-bins", type=int, default=0)
    parser.add_argument("--unigram-weight", type=float, default=1.0)
    parser.add_argument("--bigram-weight", type=float, default=1.0)
    parser.add_argument("--trigram-weight", type=float, default=0.0)
    parser.add_argument("--position-bins", type=int, default=0)
    parser.add_argument("--position-weight", type=float, default=0.0)
    parser.add_argument("--position-bigram-bins", type=int, default=0)
    parser.add_argument("--position-bigram-weight", type=float, default=0.0)
    parser.add_argument("--local-search-swaps", type=int, default=0)
    parser.add_argument("--local-search-min-improvement", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-texts", type=int, default=0)
    parser.add_argument("--max-texts", type=int, default=10000)
    parser.add_argument(
        "--local-text-path",
        default="data/tinystories_train_15k.txt",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="artifacts/tokenizers/tinystories_bpe_1024/tokenizer.json",
    )
    parser.add_argument("--data-dir", default="data/tinystories_local_bpe_1024")
    parser.add_argument(
        "--save-one-hot-probs",
        action="store_true",
        help="Also store one-hot input_probs/target_probs for soft-checkpoint eval.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    args = parser.parse_args()

    set_random_seed(args.seed)
    dataset = _build_dataset(args)
    input_ids = dataset.input_ids.long()
    selected_indices = _select(input_ids, args)
    selected_ids = input_ids[selected_indices].contiguous()

    checkpoint = {
        "step": 0,
        "input_ids": selected_ids,
        "hard_tokens": selected_ids,
        "selected_indices": selected_indices.cpu(),
        "selection": args.selection,
        "selection_config": vars(args),
        "config": {
            "model": {
                "_target_": "src.model.TinyTransformerLM",
                "vocab_size": args.vocab_size,
                "max_seq_len": args.sequence_length,
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 4,
                "dim_feedforward": 512,
                "dropout": 0.0,
                "pad_token_id": 0,
            }
        },
    }
    if args.save_one_hot_probs:
        dtype = torch.float16 if args.dtype == "float16" else torch.float32
        probs = _one_hot_probs(selected_ids, args.vocab_size, dtype=dtype)
        checkpoint["input_probs"] = probs
        checkpoint["target_probs"] = probs
        checkpoint["synthetic_parameter_count"] = int(probs.numel())

    output_path = _resolve(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Saved {args.selection} hard-token coreset to {output_path}")
    print(f"input_ids={tuple(selected_ids.shape)}")
    print(f"selected_indices[:16]={selected_indices[:16].tolist()}")


if __name__ == "__main__":
    main()

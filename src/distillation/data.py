import torch

from src.datasets.data_utils import inf_loop


def collect_initial_tokens(dataloader, num_sequences, device):
    chunks = []
    for batch in inf_loop(dataloader):
        chunks.append(batch["input_ids"])
        input_ids = torch.cat(chunks, dim=0)
        if input_ids.shape[0] >= num_sequences:
            return input_ids[:num_sequences].to(device)
    raise RuntimeError("Could not collect enough real tokens for initialization.")


def collect_initial_tokens_from_dataset(dataset, num_sequences, offset, device):
    if not hasattr(dataset, "input_ids"):
        raise ValueError(
            "init_sequence_offset requires a dataset with an input_ids tensor."
        )
    offset = int(offset)
    if offset < 0:
        raise ValueError(f"init_sequence_offset must be non-negative, got {offset}.")
    end = offset + int(num_sequences)
    if end > len(dataset):
        raise ValueError(
            "Not enough dataset sequences for requested synthetic init slice. "
            f"Need [{offset}:{end}], dataset length is {len(dataset)}."
        )
    return dataset.input_ids[offset:end].to(device)


def collect_token_mixture_logits_from_dataset(
    dataset,
    num_sequences,
    sequence_length,
    vocab_size,
    eps,
    device,
    offset=0,
    max_source_sequences=None,
):
    if not hasattr(dataset, "input_ids"):
        raise ValueError(
            "init_mode=real_mixture requires a dataset with an input_ids tensor."
        )

    source_ids = dataset.input_ids
    if source_ids.dim() != 2:
        raise ValueError(f"Expected 2D input_ids, got {tuple(source_ids.shape)}.")
    if int(source_ids.shape[1]) != int(sequence_length):
        raise ValueError(
            "Source dataset sequence length does not match synthetic length. "
            f"dataset={source_ids.shape[1]}, synthetic={sequence_length}."
        )

    offset = int(offset or 0)
    if offset < 0:
        raise ValueError(f"init_mixture_offset must be non-negative, got {offset}.")

    max_source_sequences = (
        None if max_source_sequences is None else int(max_source_sequences)
    )
    end = None if max_source_sequences is None else offset + max_source_sequences
    source_ids = source_ids[offset:end]
    if source_ids.numel() == 0:
        raise ValueError("No source sequences left for real_mixture init.")

    if source_ids.min().item() < 0 or source_ids.max().item() >= int(vocab_size):
        raise ValueError("Source token ids are outside synthetic vocab range.")

    source_ids = source_ids.long().cpu()
    num_source = int(source_ids.shape[0])
    group_ids = torch.arange(num_source, dtype=torch.long)
    group_ids = torch.div(
        group_ids * int(num_sequences),
        num_source,
        rounding_mode="floor",
    )
    positions = torch.arange(int(sequence_length), dtype=torch.long)
    group_index = group_ids[:, None].expand(num_source, int(sequence_length))
    position_index = positions[None, :].expand(num_source, int(sequence_length))

    counts = torch.full(
        (int(num_sequences), int(sequence_length), int(vocab_size)),
        float(eps),
        dtype=torch.float32,
    )
    counts.index_put_(
        (group_index, position_index, source_ids),
        torch.ones_like(source_ids, dtype=counts.dtype),
        accumulate=True,
    )
    probs = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs.log().to(device)


def synthetic_batch(synthetic_data, model_params, batch_size, device, indices=None):
    if indices is None:
        indices = synthetic_data.sample_indices(batch_size, device=device)
    else:
        indices = torch.as_tensor(indices, dtype=torch.long, device=device)
    embedding_weight = model_params["token_embedding.weight"]
    token_probs = synthetic_data.token_probs(
        indices=indices,
        embedding_weight=embedding_weight,
    )
    if getattr(synthetic_data, "uses_decoupled_targets", False):
        target_probs = synthetic_data.target_probs(
            indices=indices,
            embedding_weight=embedding_weight,
        )
    else:
        target_probs = token_probs
    input_embeds = synthetic_data.input_embeds(indices, embedding_weight)
    attention_mask = torch.ones(
        token_probs.shape[:2],
        dtype=torch.long,
        device=device,
    )
    return {
        "indices": indices,
        "input_embeds": input_embeds,
        "attention_mask": attention_mask,
        "target_probs": target_probs,
    }


def decode_synthetic_texts(synthetic_data, tokenizer, embedding_weight=None):
    return decode_token_ids(synthetic_data.hard_tokens(embedding_weight), tokenizer)


def decode_token_ids(token_ids, tokenizer):
    return [tokenizer.decode(tokens) for tokens in token_ids.detach().cpu().tolist()]

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


def collect_grouped_initial_token_probs(
    dataloader,
    num_sequences,
    group_size,
    vocab_size,
    device,
    probability_floor=0.01,
):
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    if probability_floor <= 0:
        raise ValueError(
            f"probability_floor must be positive, got {probability_floor}."
        )

    total_sequences = num_sequences * group_size
    input_ids = collect_initial_tokens(
        dataloader=dataloader,
        num_sequences=total_sequences,
        device=device,
    )
    grouped_ids = input_ids.view(num_sequences, group_size, -1).transpose(1, 2).contiguous()
    token_probs = torch.full(
        (num_sequences, grouped_ids.shape[1], vocab_size),
        fill_value=float(probability_floor),
        device=device,
        dtype=torch.float32,
    )
    token_probs.scatter_add_(
        dim=-1,
        index=grouped_ids,
        src=torch.ones_like(grouped_ids, dtype=token_probs.dtype),
    )
    token_probs /= token_probs.sum(dim=-1, keepdim=True)
    return token_probs


def synthetic_batch(synthetic_data, model_params, batch_size, device):
    indices = synthetic_data.sample_indices(batch_size, device=device)
    embedding_weight = model_params["token_embedding.weight"]
    token_probs = synthetic_data.token_probs(
        indices=indices,
        embedding_weight=embedding_weight,
    )
    input_embeds = synthetic_data.input_embeds(indices, embedding_weight)
    attention_mask = torch.ones(
        token_probs.shape[:2],
        dtype=torch.long,
        device=device,
    )
    return {
        "input_embeds": input_embeds,
        "attention_mask": attention_mask,
        "target_probs": token_probs,
    }


def decode_synthetic_texts(synthetic_data, tokenizer, embedding_weight=None):
    return decode_token_ids(synthetic_data.hard_tokens(embedding_weight), tokenizer)


def decode_token_ids(token_ids, tokenizer):
    return [tokenizer.decode(tokens) for tokens in token_ids.detach().cpu().tolist()]

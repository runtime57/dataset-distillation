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

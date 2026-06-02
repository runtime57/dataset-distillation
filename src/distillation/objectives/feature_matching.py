import torch

from src.distillation.utils import move_batch_to_device


def _hidden_position_sum_and_count(hidden, attention_mask=None):
    if attention_mask is None:
        count = hidden.new_full((hidden.shape[1],), hidden.shape[0])
        return hidden.sum(dim=0), count

    mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    count = mask.squeeze(-1).sum(dim=0)
    return (hidden * mask).sum(dim=0), count


def _masked_position_mean(hidden, attention_mask=None):
    hidden_sum, count = _hidden_position_sum_and_count(hidden, attention_mask)
    return hidden_sum / count.clamp_min(1.0).unsqueeze(-1), count > 0


def _position_mse(real_mean, synth_mean, valid_positions):
    squared_error = (real_mean - synth_mean) ** 2
    valid = valid_positions.to(device=squared_error.device).unsqueeze(-1)
    return (squared_error * valid).sum() / (
        valid.sum().clamp_min(1.0) * squared_error.shape[-1]
    )


def feature_matching_loss(model, real_loader, synth_batch, outer_batches, device):
    n_layers = None
    real_sums = None
    real_counts = None

    with torch.no_grad():
        for _ in range(outer_batches):
            real_batch = move_batch_to_device(next(real_loader), device)
            real_attention_mask = real_batch.get("attention_mask")
            outputs = model(
                input_ids=real_batch["input_ids"],
                attention_mask=real_attention_mask,
                return_hidden_states=True,
            )
            hidden_states = outputs["hidden_states"]

            if real_sums is None:
                n_layers = len(hidden_states)
                real_sums = []
                real_counts = []
                for hidden in hidden_states:
                    hidden_sum, count = _hidden_position_sum_and_count(
                        hidden,
                        real_attention_mask,
                    )
                    real_sums.append(hidden_sum)
                    real_counts.append(count)
            else:
                for layer_index in range(n_layers):
                    hidden_sum, count = _hidden_position_sum_and_count(
                        hidden_states[layer_index],
                        real_attention_mask,
                    )
                    real_sums[layer_index] = real_sums[layer_index] + hidden_sum
                    real_counts[layer_index] = real_counts[layer_index] + count

    real_means = [
        hidden_sum / count.clamp_min(1.0).unsqueeze(-1)
        for hidden_sum, count in zip(real_sums, real_counts)
    ]
    real_valid_positions = [count > 0 for count in real_counts]

    synth_outputs = model(
        input_embeds=synth_batch["input_embeds"],
        attention_mask=synth_batch["attention_mask"],
        return_hidden_states=True,
    )
    synth_hidden_states = synth_outputs["hidden_states"]

    return sum(
        _position_mse(
            real_means[layer_index],
            _masked_position_mean(
                synth_hidden_states[layer_index],
                synth_batch.get("attention_mask"),
            )[0],
            real_valid_positions[layer_index],
        )
        for layer_index in range(n_layers)
    )

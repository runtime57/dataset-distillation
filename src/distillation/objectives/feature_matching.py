import torch

from src.distillation.utils import move_batch_to_device

def feature_matching_loss(model, real_loader, synth_batch, outer_batches, device):
    n_layers = None
    real_sums = None
    real_count = 0

    with torch.no_grad():
        for _ in range(outer_batches):
            real_batch = move_batch_to_device(next(real_loader), device)
            outputs = model(
                input_ids=real_batch["input_ids"],
                attention_mask=real_batch.get("attention_mask"),
                return_hidden_states=True,
            )
            hidden_states = outputs["hidden_states"]

            if real_sums is None:
                n_layers = len(hidden_states)
                real_sums = [hidden.mean(dim=[0, 1]) for hidden in hidden_states]
            
            else:
                for layer_index in range(n_layers):
                    real_sums[layer_index] = real_sums[layer_index] + hidden_states[layer_index].mean(dim=[0, 1])
            
            real_count += 1
    real_means = [hidden_sum / real_count for hidden_sum in real_sums]

    synth_outputs = model(
        input_embeds=synth_batch["input_embeds"],
        attention_mask=synth_batch["attention_mask"],
        return_hidden_states=True,
    )
    synth_hidden_states = synth_outputs["hidden_states"]

    return sum(
        ((real_means[layer_index] - synth_hidden_states[layer_index].mean(dim=[0, 1])) ** 2).mean()
        for layer_index in range(n_layers)
    )
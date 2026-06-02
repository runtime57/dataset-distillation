import torch
from torch.func import functional_call

from src.distillation.data import synthetic_batch
from src.distillation.losses import soft_lm_loss


def one_step_parameters(
    model,
    params,
    buffers,
    synthetic_data,
    config,
    device,
    inner_lr,
):
    synth_batch = synthetic_batch(
        synthetic_data=synthetic_data,
        model_params=params,
        batch_size=config.distillation.synthetic.batch_size,
        device=device,
    )
    outputs = functional_call(
        model,
        (params, buffers),
        args=(),
        kwargs={
            "input_embeds": synth_batch["input_embeds"],
            "attention_mask": synth_batch["attention_mask"],
        },
    )
    inner_loss = soft_lm_loss(
        outputs["logits"],
        synth_batch["target_probs"],
        attention_mask=synth_batch["attention_mask"],
    )
    gradients = torch.autograd.grad(
        inner_loss,
        tuple(params.values()),
        create_graph=True,
        allow_unused=True,
    )
    updated_params = {}
    for (name, parameter), gradient in zip(params.items(), gradients):
        updated_params[name] = (
            parameter if gradient is None else parameter - inner_lr * gradient
        )
    return updated_params, inner_loss

import torch
from hydra.utils import instantiate
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


def run_expert_trajectory(config):
    set_random_seed(config.expert.seed)
    device = resolve_device(config.expert.device)

    datasets = build_datasets(config, partitions=("train",))
    dataloaders = build_dataloaders(config, datasets)
    real_loader = inf_loop(dataloaders["train"])

    model = instantiate(config.model).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.expert.lr,
        # momentum=0.9,
    )

    save_path = ROOT_PATH / config.expert.save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoints = []
    model.train()
    for step in trange(config.expert.n_steps, desc="expert trajectory"):
        batch = move_batch_to_device(next(real_loader), device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        loss = hard_lm_loss(outputs["logits"], batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.expert.save_period == 0:
            checkpoints.append(
                {key: value.cpu().clone() for key, value in model.state_dict().items()}
            )

    checkpoints.append(
        {key: value.cpu().clone() for key, value in model.state_dict().items()}
    )
    output_path = save_path / "expert_checkpoints.pth"
    torch.save(checkpoints, output_path)
    print(f"Saved {len(checkpoints)} expert checkpoints -> {output_path}")
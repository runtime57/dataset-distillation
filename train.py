import warnings
from pprint import pformat

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import (
    resolve_device,
    set_random_seed,
    setup_saving_and_logging,
)

warnings.filterwarnings("ignore", category=UserWarning)


def _validate_checkpoint_dataset_matches_model(config, dataloaders):
    train_loader = dataloaders.get("train")
    train_dataset = (
        None if train_loader is None else getattr(train_loader, "dataset", None)
    )
    if train_dataset is None:
        return

    sequence_length = getattr(train_dataset, "sequence_length", None)
    max_seq_len = config.model.get("max_seq_len")
    if sequence_length is not None and max_seq_len is not None:
        if int(sequence_length) > int(max_seq_len):
            raise ValueError(
                "Train dataset sequence length exceeds model.max_seq_len. "
                f"dataset={sequence_length}, model.max_seq_len={max_seq_len}. "
                "Use a matched model config for this checkpoint."
            )

    dataset_vocab_size = getattr(train_dataset, "vocab_size", None)
    model_vocab_size = config.model.get("vocab_size")
    if dataset_vocab_size is not None and model_vocab_size is not None:
        if int(dataset_vocab_size) != int(model_vocab_size):
            raise ValueError(
                "Train dataset vocab_size does not match model.vocab_size. "
                f"dataset={dataset_vocab_size}, model.vocab_size={model_vocab_size}."
            )

    checkpoint_model = getattr(train_dataset, "checkpoint_model_config", None)
    if checkpoint_model is None:
        return

    current_model = OmegaConf.to_container(config.model, resolve=True)
    if checkpoint_model != current_model:
        checkpoint_path = getattr(train_dataset, "checkpoint_path", "<unknown>")
        raise ValueError(
            "Distilled checkpoint model config does not match current train model. "
            f"checkpoint_path={checkpoint_path}\n"
            f"checkpoint model:\n{pformat(checkpoint_model)}\n"
            f"current model:\n{pformat(current_model)}"
        )


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    device = resolve_device(config.trainer.device)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)
    _validate_checkpoint_dataset_matches_model(config, dataloaders)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()

from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def _resolve_partitions(config, partitions=None):
    """
    Resolve dataset partitions requested by the caller.

    Args:
        config (DictConfig): hydra experiment config.
        partitions (Iterable[str] | None): requested partitions. If None,
            use all dataset partitions defined in the config.
    Returns:
        list[str]: partitions in a stable order.
    """
    available_partitions = list(config.datasets.keys())
    if partitions is None:
        return available_partitions

    resolved = []
    for partition in partitions:
        if partition not in config.datasets:
            raise KeyError(
                f"Unknown dataset partition {partition!r}. "
                f"Available partitions: {available_partitions}."
            )
        if partition not in resolved:
            resolved.append(partition)
    return resolved


def build_datasets(config, partitions=None):
    """
    Instantiate only the requested dataset partitions.

    Args:
        config (DictConfig): hydra experiment config.
        partitions (Iterable[str] | None): requested dataset partitions.
    Returns:
        dict[str, Dataset]: instantiated datasets keyed by partition.
    """
    resolved_partitions = _resolve_partitions(config, partitions)
    return {
        partition: instantiate(config.datasets[partition])
        for partition in resolved_partitions
    }


def build_dataloaders(config, datasets):
    """
    Build dataloaders for already instantiated datasets.

    Args:
        config (DictConfig): hydra experiment config.
        datasets (dict[str, Dataset]): dataset instances keyed by partition.
    Returns:
        dict[str, DataLoader]: dataloaders keyed by partition.
    """
    dataloaders = {}
    for dataset_partition, dataset in datasets.items():
        assert config.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader
    return dataloaders


def get_dataloaders(config, device, partitions=None):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
        partitions (Iterable[str] | None): requested dataset partitions.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    datasets = build_datasets(config, partitions=partitions)
    dataloaders = build_dataloaders(config, datasets)
    return dataloaders, batch_transforms

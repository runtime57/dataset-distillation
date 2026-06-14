import argparse
import csv
import gzip
import json
import math
import struct
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


class LeNet(nn.Module):
    """LeNet-style MNIST classifier without dropout, matching the paper setup."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, return_features=False):
        x = F.pad(x, (2, 2, 2, 2))
        h1 = F.relu(self.conv1(x))
        p1 = F.avg_pool2d(h1, 2)
        h2 = F.relu(self.conv2(p1))
        p2 = F.avg_pool2d(h2, 2)
        flat = torch.flatten(p2, 1)
        h3 = F.relu(self.fc1(flat))
        h4 = F.relu(self.fc2(h3))
        logits = self.fc3(h4)
        if return_features:
            return logits, (h1, h2, h3, h4)
        return logits


class SyntheticMNIST(nn.Module):
    def __init__(
        self,
        n_steps,
        images_per_step,
        init_std=0.1,
        lr_init=0.02,
        lr_steps=None,
        image_param="sigmoid",
    ):
        super().__init__()
        if image_param not in {"sigmoid", "raw"}:
            raise ValueError(f"Unsupported image_param: {image_param}")
        self.n_steps = n_steps
        self.images_per_step = images_per_step
        self.image_param = image_param
        self.images = nn.Parameter(
            init_std * torch.randn(n_steps, images_per_step, 1, 28, 28)
        )
        lr_steps = n_steps if lr_steps is None else lr_steps
        raw_lr = math.log(math.exp(lr_init) - 1.0)
        self.raw_lrs = nn.Parameter(torch.full((lr_steps,), raw_lr))
        labels = torch.arange(images_per_step) % 10
        self.register_buffer("labels", labels.long())

    def batch(self, step):
        return self.training_images()[step], self.labels

    def training_images(self):
        if self.image_param == "sigmoid":
            return self.images.sigmoid()
        return self.images

    def visualization_images(self):
        images = self.training_images().detach().cpu()
        if self.image_param == "sigmoid":
            return images.clamp(0.0, 1.0)
        flat = images.flatten(2)
        min_values = flat.min(dim=2).values[:, :, None, None, None]
        max_values = flat.max(dim=2).values[:, :, None, None, None]
        return (images - min_values) / (max_values - min_values).clamp(min=1e-8)

    def lrs(self):
        return F.softplus(self.raw_lrs)

    @torch.no_grad()
    def initialize_from_images(self, images):
        if self.image_param == "sigmoid":
            clamped = images.clamp(1e-4, 1.0 - 1e-4)
            self.images.copy_(torch.logit(clamped))
        else:
            self.images.copy_(images)

    @torch.no_grad()
    def initialize_uniform_pixels(self, low=1e-3, high=1.0 - 1e-3):
        pixels = torch.empty_like(self.images).uniform_(low, high)
        if self.image_param == "sigmoid":
            self.images.copy_(torch.logit(pixels))
        else:
            self.images.copy_(pixels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective",
        choices=["one_step", "gradient_matching", "trajectory_matching", "feature_matching"],
        required=True,
    )
    parser.add_argument("--output-root", default="results/mnist_smoke")
    parser.add_argument("--data-root", default="data/mnist")
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--distill-steps", type=int, default=10)
    parser.add_argument("--distill-epochs", type=int, default=3)
    parser.add_argument(
        "--lr-schedule",
        choices=["per_step", "per_update"],
        default="per_step",
        help=(
            "Use one learned LR per synthetic step, re-used across epochs, or "
            "one LR per inner update."
        ),
    )
    parser.add_argument("--eval-seeds", type=int, default=1)
    parser.add_argument("--init-batch", type=int, default=2)
    parser.add_argument("--synthetic-lr", type=float, default=0.001)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--image-param",
        choices=["sigmoid", "raw"],
        default="sigmoid",
        help="Optimize pixels through a sigmoid or pass unconstrained synthetic tensors to the model.",
    )
    parser.add_argument(
        "--synthetic-lr-scheduler",
        choices=["none", "reduce_on_plateau", "step"],
        default="none",
    )
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=100)
    parser.add_argument("--plateau-threshold", type=float, default=1e-3)
    parser.add_argument("--plateau-cooldown", type=int, default=0)
    parser.add_argument("--min-synthetic-lr", type=float, default=1e-5)
    parser.add_argument("--plateau-ema", type=float, default=0.95)
    parser.add_argument("--step-lr-size", type=int, default=400)
    parser.add_argument("--step-lr-gamma", type=float, default=0.5)
    parser.add_argument("--distilled-lr-init", type=float, default=0.02)
    parser.add_argument("--init-std", type=float, default=0.1)
    parser.add_argument("--init-mode", choices=["noise", "uniform", "real"], default="noise")
    parser.add_argument("--expert-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def resolve_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def download_file(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def read_idx_images(path):
    with gzip.open(path, "rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic in {path}: {magic}")
        data = np.frombuffer(file.read(), dtype=np.uint8)
    images = data.reshape(count, rows, cols).astype("float32") / 255.0
    return torch.from_numpy(images).unsqueeze(1)


def read_idx_labels(path):
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic in {path}: {magic}")
        data = np.frombuffer(file.read(), dtype=np.uint8)
    return torch.from_numpy(data.astype("int64"))


def load_mnist(data_root):
    data_root = Path(data_root)
    files = {name: data_root / f"{name}.gz" for name in MNIST_URLS}
    for name, url in MNIST_URLS.items():
        download_file(url, files[name])
    train_x = read_idx_images(files["train_images"])
    train_y = read_idx_labels(files["train_labels"])
    test_x = read_idx_images(files["test_images"])
    test_y = read_idx_labels(files["test_labels"])
    return train_x, train_y, test_x, test_y


def subset_by_seed(images, labels, size, seed):
    if size is None or size >= len(labels):
        return images, labels
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(labels), generator=generator)[:size]
    return images[indices], labels[indices]


def balanced_real_init(images, labels, n_steps):
    selected = []
    for _step in range(n_steps):
        step_images = []
        for class_index in range(10):
            class_indices = (labels == class_index).nonzero(as_tuple=False).flatten()
            source_index = class_indices[_step % len(class_indices)]
            step_images.append(images[source_index])
        selected.append(torch.stack(step_images))
    return torch.stack(selected)


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def make_model(seed, device):
    torch.manual_seed(seed)
    return LeNet().to(device)


def named_params(model):
    return OrderedDict((name, value) for name, value in model.named_parameters())


def model_buffers(model):
    return OrderedDict((name, value) for name, value in model.named_buffers())


def ce_with_params(model, params, buffers, x, y):
    logits = functional_call(model, (params, buffers), (x,))
    return F.cross_entropy(logits, y)


def update_params(model, params, buffers, x, y, lr, create_graph=True):
    loss = ce_with_params(model, params, buffers, x, y)
    grads = torch.autograd.grad(
        loss,
        tuple(params.values()),
        create_graph=create_graph,
        allow_unused=True,
    )
    updated = OrderedDict()
    for (name, value), grad in zip(params.items(), grads):
        updated[name] = value if grad is None else value - lr * grad
    return updated, loss


def apply_synthetic_training(
    model,
    params,
    buffers,
    synthetic,
    epochs,
    device,
    create_graph=True,
):
    inner_loss = None
    lrs = synthetic.lrs()
    expected_lrs = epochs * synthetic.n_steps
    if lrs.numel() not in {synthetic.n_steps, expected_lrs}:
        raise ValueError(
            "Synthetic LR count must equal n_steps or epochs * n_steps, got "
            f"{lrs.numel()} for epochs={epochs}, n_steps={synthetic.n_steps}."
        )
    for epoch in range(epochs):
        for step in range(synthetic.n_steps):
            lr_index = epoch * synthetic.n_steps + step
            lr = lrs[lr_index] if lrs.numel() == expected_lrs else lrs[step]
            x_s, y_s = synthetic.batch(step)
            x_s = x_s.to(device)
            y_s = y_s.to(device)
            params, inner_loss = update_params(
                model,
                params,
                buffers,
                x_s,
                y_s,
                lr,
                create_graph=create_graph,
            )
    return params, inner_loss


def one_step_loss(model, synthetic, real_batch, epochs, device):
    x_real, y_real = (tensor.to(device) for tensor in real_batch)
    params = named_params(model)
    buffers = model_buffers(model)
    updated_params, inner_loss = apply_synthetic_training(
        model, params, buffers, synthetic, epochs, device
    )
    outer_loss = ce_with_params(model, updated_params, buffers, x_real, y_real)
    return outer_loss, inner_loss.detach()


def gradient_matching_loss(model, synthetic, real_batch, synth_step, device):
    x_real, y_real = (tensor.to(device) for tensor in real_batch)
    params = named_params(model)
    buffers = model_buffers(model)
    x_s, y_s = synthetic.batch(synth_step)
    x_s = x_s.to(device)
    y_s = y_s.to(device)

    total = x_real.new_tensor(0.0)
    weight = 0
    matched_classes = 0
    for class_index in y_s.unique(sorted=True):
        real_mask = y_real == class_index
        synth_mask = y_s == class_index
        if not real_mask.any() or not synth_mask.any():
            continue
        matched_classes += 1
        real_loss = ce_with_params(
            model,
            params,
            buffers,
            x_real[real_mask],
            y_real[real_mask],
        )
        real_grads = torch.autograd.grad(
            real_loss,
            tuple(params.values()),
            create_graph=False,
            allow_unused=True,
        )
        synth_loss = ce_with_params(
            model,
            params,
            buffers,
            x_s[synth_mask],
            y_s[synth_mask],
        )
        synth_grads = torch.autograd.grad(
            synth_loss,
            tuple(params.values()),
            create_graph=True,
            allow_unused=True,
        )
        for real_grad, synth_grad in zip(real_grads, synth_grads):
            if real_grad is None or synth_grad is None:
                continue
            real_flat = real_grad.detach().flatten()
            synth_flat = synth_grad.flatten()
            real_norm = real_flat.norm()
            synth_norm = synth_flat.norm()
            if real_norm > 1e-8 and synth_norm > 1e-8:
                total = total + real_flat.numel() * (
                    1.0 - torch.dot(real_flat, synth_flat) / (real_norm * synth_norm)
                )
                weight += real_flat.numel()
    if matched_classes == 0:
        raise RuntimeError("Gradient matching batch has no overlapping classes.")
    synth_loss = ce_with_params(model, params, buffers, x_s, y_s)
    return total / max(weight, 1), synth_loss.detach()


def feature_matching_loss(model, synthetic, real_batch, synth_step, device):
    x_real, y_real = (tensor.to(device) for tensor in real_batch)
    with torch.no_grad():
        _logits, real_features = model(x_real, return_features=True)
    x_s, y_s = synthetic.batch(synth_step)
    x_s = x_s.to(device)
    y_s = y_s.to(device)
    logits, synth_features = model(x_s, return_features=True)

    feature_loss = x_real.new_tensor(0.0)
    matched_classes = 0
    for class_index in y_s.unique(sorted=True):
        real_mask = y_real == class_index
        synth_mask = y_s == class_index
        if not real_mask.any() or not synth_mask.any():
            continue
        matched_classes += 1
        for real_feature, synth_feature in zip(real_features, synth_features):
            real_mean = real_feature[real_mask].mean(dim=0)
            synth_mean = synth_feature[synth_mask].mean(dim=0)
            feature_loss = feature_loss + F.mse_loss(synth_mean, real_mean)
    if matched_classes == 0:
        raise RuntimeError("Feature matching batch has no overlapping classes.")
    feature_loss = feature_loss / matched_classes
    return feature_loss, F.cross_entropy(logits, y_s).detach()


def train_expert_trajectory(loader, args, device):
    model = make_model(args.seed + 10_000, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    iterator = cycle_loader(loader)
    checkpoint_interval = args.distill_epochs * args.distill_steps
    trajectory = [clone_state(model)]
    for step in range(1, args.expert_steps + 1):
        x, y = (tensor.to(device) for tensor in next(iterator))
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        if step % checkpoint_interval == 0:
            trajectory.append(clone_state(model))
    return trajectory


def clone_state(model):
    return OrderedDict(
        (name, value.detach().clone()) for name, value in model.named_parameters()
    )


def trajectory_matching_loss(model, synthetic, expert_trajectory, epochs, device):
    index = torch.randint(0, len(expert_trajectory) - 1, (1,)).item()
    start = expert_trajectory[index]
    target = expert_trajectory[index + 1]
    params = OrderedDict(
        (name, value.detach().clone().to(device).requires_grad_(True))
        for name, value in start.items()
    )
    buffers = model_buffers(model)
    params, inner_loss = apply_synthetic_training(
        model,
        params,
        buffers,
        synthetic,
        epochs,
        device,
        create_graph=True,
    )
    names = list(params.keys())
    student_flat = torch.cat([params[name].flatten() for name in names])
    target_flat = torch.cat([target[name].to(device).flatten() for name in names])
    start_flat = torch.cat([start[name].to(device).flatten() for name in names])
    numerator = ((student_flat - target_flat) ** 2).sum()
    denominator = ((start_flat - target_flat) ** 2).sum().clamp(min=1e-8)
    return numerator / denominator, inner_loss.detach()


@torch.no_grad()
def evaluate_model(model, loader, device, max_batches=None):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for batch_index, (x, y) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()
    return {"loss": loss_sum / total, "accuracy": correct / total}


def evaluate_synthetic(synthetic, test_loader, args, device):
    accuracies = []
    losses = []
    for seed_index in range(args.eval_seeds):
        model = make_model(args.seed + 20_000 + seed_index, device)
        params = named_params(model)
        buffers = model_buffers(model)
        with torch.enable_grad():
            params, _ = apply_synthetic_training(
                model,
                params,
                buffers,
                synthetic,
                args.distill_epochs,
                device,
                create_graph=False,
            )
        final_state = model.state_dict()
        final_state.update({name: value.detach() for name, value in params.items()})
        model.load_state_dict(final_state, strict=False)
        metrics = evaluate_model(model, test_loader, device)
        accuracies.append(metrics["accuracy"])
        losses.append(metrics["loss"])
    return {
        "eval_accuracy_mean": float(np.mean(accuracies)),
        "eval_accuracy_std": float(np.std(accuracies)),
        "eval_loss_mean": float(np.mean(losses)),
    }


def save_image_grid(synthetic, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    images = synthetic.visualization_images().numpy()
    n_steps = images.shape[0]
    cell = 56
    gap = 4
    top = 14
    left = 22
    width = left + 10 * cell + 9 * gap
    height = top + n_steps * cell + max(n_steps - 1, 0) * gap
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)
    for label in range(10):
        x = left + label * (cell + gap) + cell // 2 - 3
        draw.text((x, 1), str(label), fill=0)
    for step in range(n_steps):
        draw.text((2, top + step * (cell + gap) + cell // 2 - 5), f"s{step}", fill=0)
        for label in range(10):
            tile = (images[step, label, 0] * 255.0).clip(0, 255).astype(np.uint8)
            tile_image = Image.fromarray(tile).resize(
                (cell, cell),
                resample=Image.Resampling.NEAREST,
            )
            x = left + label * (cell + gap)
            y = top + step * (cell + gap)
            canvas.paste(tile_image, (x, y))
    canvas.save(path)


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_markdown_table(csv_path, md_path):
    with csv_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    headers = [
        "objective",
        "best_step",
        "best_train_loss",
        "eval_accuracy_mean",
        "eval_accuracy_std",
        "eval_loss_mean",
        "image_grid",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if header.startswith("eval_") or header == "best_train_loss":
                value = f"{float(value):.4f}"
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    output_root = Path(args.output_root)
    run_dir = output_root / args.objective
    run_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, test_x, test_y = load_mnist(args.data_root)
    train_x, train_y = subset_by_seed(train_x, train_y, args.train_size, args.seed)
    test_x, test_y = subset_by_seed(test_x, test_y, args.test_size, args.seed + 1)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=args.batch_size,
        shuffle=False,
    )
    real_iter = cycle_loader(train_loader)

    synthetic = SyntheticMNIST(
        n_steps=args.distill_steps,
        images_per_step=10,
        init_std=args.init_std,
        lr_init=args.distilled_lr_init,
        lr_steps=(
            args.distill_steps * args.distill_epochs
            if args.lr_schedule == "per_update"
            else args.distill_steps
        ),
        image_param=args.image_param,
    ).to(device)
    if args.init_mode == "uniform":
        synthetic.initialize_uniform_pixels()
    elif args.init_mode == "real":
        init_images = balanced_real_init(train_x, train_y, args.distill_steps).to(device)
        synthetic.initialize_from_images(init_images)
    optimizer = torch.optim.Adam(
        synthetic.parameters(),
        lr=args.synthetic_lr,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    scheduler = None
    if args.synthetic_lr_scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            threshold=args.plateau_threshold,
            threshold_mode="abs",
            cooldown=args.plateau_cooldown,
            min_lr=args.min_synthetic_lr,
        )
    elif args.synthetic_lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_lr_size,
            gamma=args.step_lr_gamma,
        )
    expert_trajectory = None
    if args.objective == "trajectory_matching":
        expert_trajectory = train_expert_trajectory(train_loader, args, device)

    best_loss = math.inf
    best_step = 0
    best_state = None
    ema_loss = None
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    progress = trange(
        1,
        args.iterations + 1,
        desc=args.objective,
        disable=args.disable_progress,
    )
    for step in progress:
        optimizer.zero_grad()
        losses = []
        inner_losses = []
        for init_index in range(
            args.init_batch if args.objective in {"one_step", "gradient_matching"} else 1
        ):
            model = make_model(args.seed + step * 100 + init_index, device)
            real_batch = next(real_iter)
            synth_step = (step - 1) % synthetic.n_steps
            if args.objective == "one_step":
                loss, inner_loss = one_step_loss(
                    model, synthetic, real_batch, args.distill_epochs, device
                )
            elif args.objective == "gradient_matching":
                loss, inner_loss = gradient_matching_loss(
                    model, synthetic, real_batch, synth_step, device
                )
            elif args.objective == "feature_matching":
                loss, inner_loss = feature_matching_loss(
                    model, synthetic, real_batch, synth_step, device
                )
            elif args.objective == "trajectory_matching":
                loss, inner_loss = trajectory_matching_loss(
                    model, synthetic, expert_trajectory, args.distill_epochs, device
                )
            else:
                raise ValueError(args.objective)
            losses.append(loss)
            inner_losses.append(inner_loss)
        train_loss = torch.stack(losses).mean()
        train_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(synthetic.parameters(), args.grad_clip)
        optimizer.step()

        train_loss_value = float(train_loss.detach().cpu())
        if ema_loss is None:
            ema_loss = train_loss_value
        else:
            ema_loss = args.plateau_ema * ema_loss + (1.0 - args.plateau_ema) * train_loss_value
        if args.synthetic_lr_scheduler == "reduce_on_plateau":
            scheduler.step(ema_loss)
        elif scheduler is not None:
            scheduler.step()
        optimizer_lr = optimizer.param_groups[0]["lr"]
        inner_loss_value = float(torch.stack(inner_losses).mean().detach().cpu())
        if train_loss_value < best_loss:
            best_loss = train_loss_value
            best_step = step
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in synthetic.state_dict().items()
            }
        row = {
            "step": step,
            "objective": args.objective,
            "train_loss": train_loss_value,
            "ema_loss": ema_loss,
            "inner_loss": inner_loss_value,
            "best_loss": best_loss,
            "optimizer_lr": optimizer_lr,
            "distilled_lr_mean": float(synthetic.lrs().mean().detach().cpu()),
            "synthetic_pixel_std": float(
                synthetic.training_images().std().detach().cpu()
            ),
        }
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, sort_keys=True) + "\n")
        progress.set_postfix(
            loss=f"{train_loss_value:.4f}",
            best=f"{best_loss:.4f}",
            opt_lr=f"{optimizer_lr:.2e}",
        )

    if best_state is not None:
        synthetic.load_state_dict(best_state)
    torch.save(
        {
            "objective": args.objective,
            "synthetic_state_dict": synthetic.state_dict(),
            "args": vars(args),
            "best_step": best_step,
            "best_train_loss": best_loss,
        },
        run_dir / "synthetic_mnist.pt",
    )
    image_path = run_dir / "synthetic_grid.png"
    save_image_grid(synthetic, image_path)
    eval_metrics = evaluate_synthetic(synthetic, test_loader, args, device)

    row = {
        "objective": args.objective,
        "best_step": best_step,
        "best_train_loss": best_loss,
        **eval_metrics,
        "image_grid": str(image_path),
    }
    summary_csv = output_root / "mnist_smoke_results.csv"
    append_csv(summary_csv, row)
    write_markdown_table(summary_csv, output_root / "mnist_smoke_results.md")
    (run_dir / "summary.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm


class ConvNet(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, width * 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(width * 2 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, return_features=False):
        features = []

        x = F.relu(self.conv1(x))
        features.append(x)
        x = F.avg_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        features.append(x)
        x = F.avg_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        features.append(x)
        logits = self.fc2(x)

        if return_features:
            return logits, features
        return logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="MNIST feature/gradient matching dataset distillation."
    )
    parser.add_argument("--method", choices=["fm", "gm"], required=True)
    parser.add_argument(
        "--init",
        choices=["random", "real1", "real_many"],
        default="random",
    )
    parser.add_argument("--ipc", type=int, default=1, help="Synthetic images per class.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--batch-real", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--eval-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-width", type=int, default=64)
    parser.add_argument("--reinit-period", type=int, default=1)
    parser.add_argument(
        "--gm-classwise",
        action="store_true",
        help="Use slower per-class GM instead of one balanced gradient match.",
    )
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--no-aug", dest="aug", action="store_false")
    parser.set_defaults(aug=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="saved/mnist_fm_gm")
    parser.add_argument("--save-every", type=int, default=100)
    return parser.parse_args()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def logit(x, eps=1e-6):
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def load_mnist(data_dir):
    train = datasets.MNIST(data_dir, train=True, download=True, transform=ToTensor())
    test = datasets.MNIST(data_dir, train=False, download=True, transform=ToTensor())
    train_images = train.data.float().unsqueeze(1) / 255.0
    train_labels = train.targets.long()
    return train_images, train_labels, test


def build_class_index(labels):
    return [torch.where(labels == cls)[0] for cls in range(10)]


def sample_real(images, labels, class_index, cls, batch_size, device):
    indices = class_index[cls]
    picks = indices[torch.randint(len(indices), (batch_size,))]
    return images[picks].to(device), labels[picks].to(device)


def sample_balanced_real(images, labels, class_index, batch_size_per_class, device):
    xs = []
    ys = []
    for cls in range(10):
        x, y = sample_real(
            images,
            labels,
            class_index,
            cls,
            batch_size_per_class,
            device,
        )
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def init_synthetic(images, labels, class_index, init_mode, ipc, device):
    syn_labels = torch.arange(10, device=device).repeat_interleave(ipc)
    chunks = []
    for cls in range(10):
        if init_mode == "random":
            chunk = torch.rand(ipc, 1, 28, 28)
        elif init_mode == "real1":
            first = images[class_index[cls][0]].repeat(ipc, 1, 1, 1)
            noise = 0.01 * torch.randn_like(first)
            chunk = (first + noise).clamp(0.0, 1.0)
        elif init_mode == "real_many":
            indices = class_index[cls]
            picks = indices[torch.randperm(len(indices))[:ipc]]
            chunk = images[picks]
        else:
            raise ValueError(f"Unknown init: {init_mode}")
        chunks.append(chunk)

    syn_images = torch.cat(chunks, dim=0).to(device)
    syn_param = nn.Parameter(logit(syn_images))
    return syn_param, syn_labels


def decode_synthetic(syn_param):
    return syn_param.sigmoid()


def augment(x, padding=2):
    if padding <= 0:
        return x
    x_pad = F.pad(x, (padding, padding, padding, padding))
    out = torch.empty_like(x)
    max_offset = padding * 2
    shifts = torch.randint(0, max_offset + 1, (x.shape[0], 2), device=x.device)
    for idx, (dy, dx) in enumerate(shifts.tolist()):
        out[idx] = x_pad[idx, :, dy : dy + 28, dx : dx + 28]
    return out


def feature_mean(feature):
    if feature.ndim == 4:
        return feature.mean(dim=(0, 2, 3))
    return feature.mean(dim=0)


def class_feature_means(feature, labels):
    means = []
    for cls in range(10):
        means.append(feature_mean(feature[labels == cls]))
    return means


def feature_matching_step(
    model,
    syn_param,
    syn_labels,
    train_images,
    train_labels,
    class_index,
    batch_real,
    device,
    use_aug,
):
    syn_images = decode_synthetic(syn_param)
    total = torch.tensor(0.0, device=device)
    real_x, real_y = sample_balanced_real(
        train_images,
        train_labels,
        class_index,
        batch_real,
        device,
    )
    if use_aug:
        real_x = augment(real_x)
        syn_images = augment(syn_images)

    with torch.no_grad():
        _, real_features = model(real_x, return_features=True)
    _, syn_features = model(syn_images, return_features=True)

    for real_feature, syn_feature in zip(real_features, syn_features):
        real_means = class_feature_means(real_feature, real_y)
        syn_means = class_feature_means(syn_feature, syn_labels)
        for real_mean, syn_mean in zip(real_means, syn_means):
            total = total + F.mse_loss(syn_mean, real_mean.detach())

    return total / 10.0


def gradient_distance(real_grads, syn_grads):
    total = syn_grads[0].new_tensor(0.0)
    total_weight = 0
    for real_grad, syn_grad in zip(real_grads, syn_grads):
        real_flat = real_grad.detach().flatten()
        syn_flat = syn_grad.flatten()
        real_norm = real_flat.norm()
        syn_norm = syn_flat.norm()
        if real_norm > 1e-8 and syn_norm > 1e-8:
            cos = (real_flat * syn_flat).sum() / (real_norm * syn_norm)
            weight = real_flat.numel()
            total = total + weight * (1.0 - cos)
            total_weight += weight
    return total / max(total_weight, 1)


def gradient_matching_step(
    model,
    syn_param,
    syn_labels,
    train_images,
    train_labels,
    class_index,
    batch_real,
    device,
    use_aug,
    classwise,
):
    syn_images = decode_synthetic(syn_param)
    params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and name.endswith("weight")
    ]
    total = torch.tensor(0.0, device=device)

    if not classwise:
        real_x, real_y = sample_balanced_real(
            train_images,
            train_labels,
            class_index,
            batch_real,
            device,
        )
        if use_aug:
            real_x = augment(real_x)
            syn_images = augment(syn_images)

        real_loss = F.cross_entropy(model(real_x), real_y)
        real_grads = torch.autograd.grad(real_loss, params, create_graph=False)

        syn_loss = F.cross_entropy(model(syn_images), syn_labels)
        syn_grads = torch.autograd.grad(syn_loss, params, create_graph=True)
        return gradient_distance(real_grads, syn_grads)

    for cls in range(10):
        real_x, real_y = sample_real(
            train_images,
            train_labels,
            class_index,
            cls,
            batch_real,
            device,
        )
        syn_x = syn_images[syn_labels == cls]
        syn_y = syn_labels[syn_labels == cls]
        if use_aug:
            real_x = augment(real_x)
            syn_x = augment(syn_x)

        real_loss = F.cross_entropy(model(real_x), real_y)
        real_grads = torch.autograd.grad(real_loss, params, create_graph=False)

        syn_loss = F.cross_entropy(model(syn_x), syn_y)
        syn_grads = torch.autograd.grad(syn_loss, params, create_graph=True)
        total = total + gradient_distance(real_grads, syn_grads)

    return total / 10.0


def save_grid(syn_param, path, ipc):
    path.parent.mkdir(parents=True, exist_ok=True)
    images = decode_synthetic(syn_param).detach().cpu().clamp(0.0, 1.0)
    utils.save_image(images, path, nrow=max(10, ipc), padding=2)


def evaluate_synthetic(
    syn_param,
    syn_labels,
    test_dataset,
    device,
    seed,
    steps,
    lr,
    width,
    use_aug,
    num_workers,
):
    seed_everything(seed)
    model = ConvNet(width=width).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    syn_images = decode_synthetic(syn_param).detach()
    syn_labels = syn_labels.detach()
    batch_size = min(256, len(syn_labels))

    for _ in tqdm(range(steps), desc=f"eval_train_seed{seed}", leave=False):
        idx = torch.randint(len(syn_labels), (batch_size,), device=device)
        x = syn_images[idx]
        y = syn_labels[idx]
        if use_aug:
            x = augment(x)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{stamp}_{args.method}_{args.init}_ipc{args.ipc}"
        f"_steps{args.steps}_seed{args.seed}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_images, train_labels, test_dataset = load_mnist(args.data_dir)
    class_index = build_class_index(train_labels)
    syn_param, syn_labels = init_synthetic(
        train_images,
        train_labels,
        class_index,
        args.init,
        args.ipc,
        device,
    )

    save_grid(syn_param, out_dir / "synthetic_step0000.png", args.ipc)
    optimizer = torch.optim.Adam([syn_param], lr=args.lr)
    model = None
    losses = []

    pbar = tqdm(range(1, args.steps + 1), desc=f"{args.method}:{args.init}:ipc{args.ipc}")
    for step in pbar:
        if model is None or (args.reinit_period > 0 and (step - 1) % args.reinit_period == 0):
            model = ConvNet(width=args.model_width).to(device)
            model.train()

        if args.method == "fm":
            loss = feature_matching_step(
                model,
                syn_param,
                syn_labels,
                train_images,
                train_labels,
                class_index,
                args.batch_real,
                device,
                args.aug,
            )
        else:
            loss = gradient_matching_step(
                model,
                syn_param,
                syn_labels,
                train_images,
                train_labels,
                class_index,
                args.batch_real,
                device,
                args.aug,
                args.gm_classwise,
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_item = float(loss.detach().cpu())
        losses.append(loss_item)
        pbar.set_postfix(loss=f"{loss_item:.4f}")

        if args.save_every > 0 and step % args.save_every == 0:
            save_grid(syn_param, out_dir / f"synthetic_step{step:04d}.png", args.ipc)

    save_grid(syn_param, out_dir / "synthetic_final.png", args.ipc)

    accuracies = []
    for repeat in range(args.eval_repeats):
        acc = evaluate_synthetic(
            syn_param,
            syn_labels,
            test_dataset,
            device,
            seed=args.seed + 1000 + repeat,
            steps=args.eval_steps,
            lr=args.eval_lr,
            width=args.model_width,
            use_aug=args.aug,
            num_workers=args.num_workers,
        )
        accuracies.append(acc)

    summary = {
        "method": args.method,
        "init": args.init,
        "ipc": args.ipc,
        "steps": args.steps,
        "eval_steps": args.eval_steps,
        "eval_repeats": args.eval_repeats,
        "batch_real": args.batch_real,
        "lr": args.lr,
        "eval_lr": args.eval_lr,
        "seed": args.seed,
        "model_width": args.model_width,
        "reinit_period": args.reinit_period,
        "gm_classwise": args.gm_classwise,
        "aug": args.aug,
        "device": str(device),
        "final_loss": losses[-1] if losses else None,
        "test_accuracies": accuracies,
        "test_acc_mean": float(np.mean(accuracies)) if accuracies else None,
        "test_acc_std": float(np.std(accuracies)) if accuracies else None,
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "losses.json").write_text(json.dumps(losses), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

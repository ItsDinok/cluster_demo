"""
Train a ResNet-18 on MNIST/EMNIST — a small, self-contained GPU demo.

Uses the FastMNISTLoader: a RAM-resident dataset of pre-computed augmentation
variants (no disk I/O and no CPU augmentation during training). Adapts
torchvision's ResNet-18 for 1-channel 28x28 inputs and trains with mixed
precision. Reports per-epoch train/test loss and accuracy, timing, throughput.

Expects pre-generated arrays produced by generate_superfast_data.py:
    superfast_emnist_train.npz  ->  keys: data [N, V, (1,) 28, 28], targets [N]
    superfast_emnist_val.npz    ->  keys: data [N, V, (1,) 28, 28], targets [N]

Usage:
    python train_mnist_resnet.py --data-dir ./data
    python train_mnist_resnet.py --epochs 10 --batch-size 512 --lr 0.2
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18


class FastMNISTLoader(Dataset):
    """
    Zero disk I/O. Zero CPU augmentation during training. Pure memory feeding.

    Data is stored as [N, num_variants, (1,) 28, 28]; each epoch we pick one of
    the pre-computed variants per image, so __getitem__ is a plain memory lookup.
    """

    def __init__(self, root, train, normalize=(0.1307, 0.3081)):
        filename = f"fast_mnist_{'train' if train else 'val'}.npz"
        filepath = os.path.join(root, filename)

        if not os.path.exists(filepath):
            print(f"Cannot find {filepath}!")
            print("Run 'generate_fast_data.py' (with the right dataset dir) first.")
            sys.exit(1)

        print(f"--> [SUPERFAST] Loading pre-computed arrays from {filepath}...")
        npz_file = np.load(filepath)

        self.data = torch.from_numpy(npz_file["data"])
        self.targets = torch.from_numpy(npz_file["targets"]).long()
        self.train = train

        # Add a channel dim if the arrays are stored as [N, V, H, W].
        if self.data.dim() == 4:
            self.data = self.data.unsqueeze(2)  # -> [N, V, 1, H, W]

        # Normalize once, up front, so per-item lookups stay math-free. If the
        # generator already emitted normalized float data, pass normalize=None.
        if normalize is not None and not torch.is_floating_point(self.data):
            mean, std = normalize
            self.data = (self.data.float() / 255.0 - mean) / std
        elif normalize is None:
            self.data = self.data.float()

        self.num_variants = self.data.shape[1]
        self.num_classes = int(self.targets.max().item()) + 1

        # Which variant to use for each image (re-rolled each epoch).
        self.current_choices = torch.zeros(len(self.data), dtype=torch.long)
        print(f"--> [SUPERFAST] Loaded {self.num_variants} variant(s) per image "
              f"into RAM ({self.num_classes} classes).")

    def set_epoch(self):
        """Randomly pick 1 of the N pre-computed variants for this epoch."""
        if self.train:
            self.current_choices = torch.randint(0, self.num_variants, (len(self.data),))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # O(1) memory lookup: no math on the CPU here.
        chosen_variant_idx = self.current_choices[idx]
        return self.data[idx, chosen_variant_idx], self.targets[idx]


def build_model(num_classes: int) -> nn.Module:
    """ResNet-18 adapted for small, single-channel 28x28 images.

    conv1 accepts 1 channel via a 3x3 stride-1 stem and the initial maxpool is
    dropped — the ImageNet 7x7/stride-2/maxpool stem discards too much signal at
    this resolution.
    """
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_loaders(batch_size: int, data_dir: str):
    train_ds = FastMNISTLoader(data_dir, train=True)
    test_ds = FastMNISTLoader(data_dir, train=False)

    # num_workers=0 on purpose: the data already lives in RAM, so there is no
    # disk I/O or CPU augmentation to parallelize. Workers would only fork/copy
    # the big tensor AND break set_epoch (main-process choices wouldn't reach
    # persistent worker copies). Single-process indexing is both faster here and
    # keeps set_epoch correct.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x)
            loss = F.cross_entropy(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x)
            loss = F.cross_entropy(out, y, reduction="sum")
        loss_sum += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on MNIST/EMNIST (GPU demo).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device found — this demo is meant for a GPU cluster.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"Device        : {torch.cuda.get_device_name(device)}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Learning rate : {args.lr}\n")

    train_loader, test_loader = get_loaders(args.batch_size, args.data_dir)
    num_classes = train_loader.dataset.num_classes

    model = build_model(num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model         : ResNet-18 ({n_params/1e6:.1f}M params, {num_classes} classes)\n")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    n_train = len(train_loader.dataset)
    best_acc = 0.0
    header = f"{'epoch':>5} | {'train loss':>10} {'train acc':>9} | {'test loss':>9} {'test acc':>8} | {'sec':>6} {'img/s':>8}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        train_loader.dataset.set_epoch()  # re-roll augmentation variants
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        scheduler.step()
        dt = time.time() - t0
        best_acc = max(best_acc, te_acc)

        print(
            f"{epoch:>5} | {tr_loss:>10.4f} {tr_acc:>9.4f} | "
            f"{te_loss:>9.4f} {te_acc:>8.4f} | {dt:>6.1f} {n_train/dt:>8.0f}"
        )

    print("-" * len(header))
    print(f"\nBest test accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()

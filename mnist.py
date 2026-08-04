"""
Train a ResNet-18 on MNIST — a small, self-contained GPU demo.

Adapts torchvision's ResNet-18 for MNIST (1-channel, 28x28 inputs) and trains
with mixed precision. Reports per-epoch train/test loss and accuracy, epoch
timing, and throughput.

Usage:
    python train_mnist_resnet.py                 # sensible defaults
    python train_mnist_resnet.py --epochs 10 --batch-size 512 --lr 0.2
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18


def build_model() -> nn.Module:
    """ResNet-18 adapted for small, single-channel MNIST images.

    Two changes vs. the ImageNet default:
      - conv1 accepts 1 channel and uses a 3x3 stride-1 kernel (the 7x7 stride-2
        stem throws away too much signal on 28x28 inputs).
      - the initial maxpool is removed for the same reason.
    """
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_loaders(batch_size: int, data_dir: str):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
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
    parser = argparse.ArgumentParser(description="Train ResNet-18 on MNIST (GPU demo).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device found — this demo is meant for a GPU cluster.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # autotune convs for fixed input sizes

    print(f"Device        : {torch.cuda.get_device_name(device)}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Learning rate : {args.lr}")

    train_loader, test_loader = get_loaders(args.batch_size, args.data_dir)

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model         : ResNet-18 ({n_params/1e6:.1f}M params)\n")

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

"""
mnist.py — single-file ResNet-18 on MNIST/EMNIST for the cluster demo.

One `python mnist.py` does the whole job:
  1. If the pre-computed arrays are missing, generate them once: load the source
     dataset and pre-calculate several random augmentation variants per image
     (albumentations), then cache them to disk as .npz.
  2. Load those arrays fully into RAM via FastMNISTLoader (no disk I/O, no CPU
     augmentation in the training loop) and train ResNet-18 with mixed precision.
  3. Report per-epoch train/test loss, accuracy, timing, and throughput.

HPC nodes have huge GPU power but are easily bottlenecked by CPU-side data
loading and augmentation. Pre-computing the augmentation once (and caching it)
moves that cost out of every training run. Only the first run pays for it.

Requires (add to requirements.txt): torch, torchvision, numpy, albumentations.
Meant to be submitted with sbatch (see run.sh), never run raw on a login node.

Usage:
    python mnist.py                                  # MNIST, defaults
    python mnist.py --dataset emnist                 # EMNIST balanced (47 classes)
    python mnist.py --epochs 10 --batch-size 512 --lr 0.2
    python mnist.py --regenerate --variants 8        # force fresh augmentation
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Shared, pre-staged dataset location on the cluster (read-only mount). Used as
# the default source dir for EMNIST; override with --dataset-root if needed.
SHARED_EMNIST_DIR = "/home/shared/air/datasets/emnist"


# --------------------------------------------------------------------------- #
# Data generation (runs once, then cached to disk)
# --------------------------------------------------------------------------- #
def _emnist_transpose(image, **kwargs):
    # EMNIST ships rotated 90° and flipped; one transpose fixes the orientation.
    return np.transpose(image, (1, 0))


def build_transform(dataset, train, rotate, scale, translate, aug_p):
    """Albumentations pipeline. Affine is applied only to the training set;
    validation gets just the orientation fix + normalization.
    """
    steps = []
    if dataset == "emnist":
        steps.append(A.Lambda(image=_emnist_transpose))
    if train:
        steps.append(A.Affine(
            translate_percent=(-translate, translate),
            scale=(scale[0], scale[1]),
            rotate=(-rotate, rotate),
            p=aug_p,
        ))
    steps.append(A.Normalize(mean=(0.5,), std=(0.5,)))  # -> ~[-1, 1] float32
    steps.append(ToTensorV2())
    return A.Compose(steps)


def _load_source(dataset, root, split, train):
    """Returns (images_u8 [N,28,28], targets [N]) from torchvision."""
    if dataset == "emnist":
        ds = torchvision.datasets.EMNIST(root=root, split=split, train=train, download=True)
    else:
        ds = torchvision.datasets.MNIST(root=root, train=train, download=True)
    return ds.data.numpy(), ds.targets.numpy()


def generate_variants(dataset, root, split, train, num_variants, save_dir, name,
                      rotate, scale, translate, aug_p, compress):
    """Pre-compute variants for one split and write superfast_<name>_<split>.npz.

    Stored as float32 of shape (N, variants, 1, 28, 28), already normalized.
    Validation uses a single, un-augmented variant.
    """
    transform = build_transform(dataset, train, rotate, scale, translate, aug_p)

    print(f"--> [DATA] Loading {dataset} (train={train})...")
    raw_data, targets = _load_source(dataset, root, split, train)

    actual_variants = num_variants if train else 1
    n = len(raw_data)
    processed = np.zeros((n, actual_variants, 1, 28, 28), dtype=np.float32)

    # Periodic, newline-terminated progress — no carriage-return bars, which
    # turn into log-file garbage under slurm. Prints roughly every 5s.
    print(f"    applying {actual_variants} variant(s) to {n} images...")
    t0 = time.time()
    last = t0
    for i in range(n):
        img = raw_data[i]
        for v in range(actual_variants):
            processed[i, v] = transform(image=img)["image"].numpy()
        now = time.time()
        if now - last >= 5.0 or i == n - 1:
            done = i + 1
            rate = done / max(now - t0, 1e-9)
            print(f"      {done}/{n} ({100 * done / n:5.1f}%)  {rate:6.0f} img/s")
            last = now

    split_tag = "train" if train else "val"
    filename = os.path.join(save_dir, f"superfast_{name}_{split_tag}.npz")
    save = np.savez_compressed if compress else np.savez
    print(f"    saving {filename} "
          f"({processed.shape} {processed.dtype})...")
    save(filename, data=processed, targets=targets)
    print("    done.\n")


def ensure_data(args):
    """Generate + cache superfast_<name>_{train,val}.npz if not already present."""
    train_path = os.path.join(args.data_dir, f"superfast_{args.name}_train.npz")
    val_path = os.path.join(args.data_dir, f"superfast_{args.name}_val.npz")

    if not args.regenerate and os.path.exists(train_path) and os.path.exists(val_path):
        print(f"--> [DATA] Found cached arrays in {args.data_dir}; skipping generation.\n")
        return

    os.makedirs(args.data_dir, exist_ok=True)
    scale = (args.scale_min, args.scale_max)
    for train in (True, False):
        generate_variants(
            dataset=args.dataset, root=args.dataset_root, split=args.emnist_split,
            train=train, num_variants=args.variants, save_dir=args.data_dir,
            name=args.name, rotate=args.rotate, scale=scale,
            translate=args.translate, aug_p=args.aug_p, compress=args.compress,
        )


# --------------------------------------------------------------------------- #
# Dataset: RAM-resident, pre-augmented, zero CPU math per item
# --------------------------------------------------------------------------- #
class FastMNISTLoader(Dataset):
    """Zero disk I/O. Zero CPU augmentation during training. Pure memory feeding.

    Data is stored as [N, num_variants, 1, 28, 28] (already normalized float32);
    each epoch we pick one pre-computed variant per image, so __getitem__ is a
    plain memory lookup.
    """

    def __init__(self, root, name, train, normalize=None):
        filename = f"superfast_{name}_{'train' if train else 'val'}.npz"
        filepath = os.path.join(root, filename)

        if not os.path.exists(filepath):
            print(f"Cannot find {filepath}! (generation should have produced it)")
            sys.exit(1)

        print(f"--> [SUPERFAST] Loading pre-computed arrays from {filepath}...")
        npz_file = np.load(filepath)

        self.data = torch.from_numpy(npz_file["data"])
        self.targets = torch.from_numpy(npz_file["targets"]).long()
        self.train = train

        # Add a channel dim only if arrays were stored as [N, V, H, W].
        if self.data.dim() == 4:
            self.data = self.data.unsqueeze(2)

        # The generator already normalized to float32, so no work here by
        # default. normalize=(mean, std) is only used if data is integer-typed.
        if normalize is not None and not torch.is_floating_point(self.data):
            mean, std = normalize
            self.data = (self.data.float() / 255.0 - mean) / std
        else:
            self.data = self.data.float()

        self.num_variants = self.data.shape[1]
        self.num_classes = int(self.targets.max().item()) + 1
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


def get_loaders(batch_size, data_dir, name):
    train_ds = FastMNISTLoader(data_dir, name, train=True)
    test_ds = FastMNISTLoader(data_dir, name, train=False)

    # num_workers=0 on purpose: the data is already in RAM, so there's no disk
    # I/O or CPU augmentation to parallelize. Workers would only fork/copy the
    # big tensor AND break set_epoch (main-process choices wouldn't reach
    # persistent worker copies). Single-process indexing is faster here and
    # keeps set_epoch correct.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
# Model + train / eval
# --------------------------------------------------------------------------- #
def build_model(num_classes):
    """ResNet-18 adapted for small, single-channel 28x28 images.

    conv1 accepts 1 channel via a 3x3 stride-1 stem and the initial maxpool is
    dropped — the ImageNet 7x7/stride-2/maxpool stem discards too much signal at
    this resolution.
    """
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    # Slurm redirects stdout to a file, which Python block-buffers by default.
    # Line-buffer it so `tail -f logs/*.log` shows progress as it happens.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="ResNet-18 on MNIST/EMNIST (single-file cluster demo).")
    # data / dataset
    parser.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    parser.add_argument("--emnist-split", type=str, default="balanced",
                        help="EMNIST split (balanced, byclass, letters, digits, ...).")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Writable dir for the .npz cache.")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="Source dir torchvision reads/downloads from. "
                             "Defaults to the shared EMNIST dir for --dataset emnist, "
                             "else --data-dir.")
    parser.add_argument("--name", type=str, default=None,
                        help="Filename stem superfast_<name>_*.npz. Defaults to --dataset.")
    # augmentation / generation
    parser.add_argument("--variants", type=int, default=6)
    parser.add_argument("--rotate", type=float, default=15.0, help="Max rotation degrees (±).")
    parser.add_argument("--translate", type=float, default=0.08, help="Max translate fraction (±).")
    parser.add_argument("--scale-min", type=float, default=0.92)
    parser.add_argument("--scale-max", type=float, default=1.08)
    parser.add_argument("--aug-p", type=float, default=0.8, help="Probability of applying Affine.")
    parser.add_argument("--compress", action="store_true", default=True,
                        help="Write compressed .npz (default on).")
    parser.add_argument("--no-compress", dest="compress", action="store_false")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regeneration even if cached arrays exist.")
    # training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Resolve defaults that depend on other args.
    if args.name is None:
        args.name = args.dataset
    if args.dataset_root is None:
        args.dataset_root = SHARED_EMNIST_DIR if args.dataset == "emnist" else args.data_dir

    # HPC guardrail: training runs on the GPU. If CUDA is missing you almost
    # certainly submitted without requesting a GPU (or are on a login node).
    if not torch.cuda.is_available():
        print("No CUDA device visible — you likely submitted without a GPU "
              "(or are on a login node). Exiting.")
        sys.exit(1)

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"Device        : {torch.cuda.get_device_name(device)}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Dataset       : {args.dataset}"
          + (f" ({args.emnist_split})" if args.dataset == "emnist" else ""))
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Learning rate : {args.lr}\n")

    # Step 1: make sure the cached arrays exist (generate once if not).
    ensure_data(args)

    # Step 2: load into RAM and build the model.
    train_loader, test_loader = get_loaders(args.batch_size, args.data_dir, args.name)
    num_classes = train_loader.dataset.num_classes
    model = build_model(num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model         : ResNet-18 ({n_params/1e6:.1f}M params, {num_classes} classes)\n")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    # Step 3: train.
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
        print(f"{epoch:>5} | {tr_loss:>10.4f} {tr_acc:>9.4f} | "
              f"{te_loss:>9.4f} {te_acc:>8.4f} | {dt:>6.1f} {n_train/dt:>8.0f}")

    print("-" * len(header))
    print(f"\nBest test accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()

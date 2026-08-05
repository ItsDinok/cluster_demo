"""
generate_fast_data.py

Pre-computes MNIST augmentation variants and writes them to .npz files that
FastMNISTLoader consumes. Each training image gets several pre-augmented
variants; at train time the loader just picks one per epoch, so there is no CPU
augmentation and no disk I/O inside the training loop.

Augmentation is done as a single batched grid_sample per variant (on GPU if one
is visible, otherwise CPU), with independent random affine parameters per image.

Output (uint8, values 0-255; FastMNISTLoader normalizes at load time):
    <out-dir>/fast_mnist_train.npz   data [N, V, 28, 28], targets [N]
    <out-dir>/fast_mnist_val.npz     data [M, 1, 28, 28], targets [M]

Variant 0 is always the clean, un-augmented image (so validation, which only
ever uses variant 0, sees undistorted digits).

This is a one-time preprocessing step. It's light, but per cluster etiquette run
it as a job (sbatch) or interactively via `erun`/`epy` rather than raw on a
login node.

Usage:
    python generate_fast_data.py --out-dir ./data --variants 6
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


def random_affine_theta(n, rot_deg, max_trans, scale_lo, scale_hi, device):
    """Per-image affine matrices [n, 2, 3] for F.affine_grid.

    Combines a random rotation, isotropic scale, and translation. Ranges are
    deliberately gentle — MNIST is sensitive and there are no flips (a flipped
    2 is not a 2, and 6/9 would collide).
    """
    angles = (torch.rand(n, device=device) * 2 - 1) * (rot_deg * math.pi / 180.0)
    scales = torch.empty(n, device=device).uniform_(scale_lo, scale_hi)
    tx = (torch.rand(n, device=device) * 2 - 1) * max_trans
    ty = (torch.rand(n, device=device) * 2 - 1) * max_trans

    cos = torch.cos(angles) * scales
    sin = torch.sin(angles) * scales
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    return theta


@torch.no_grad()
def make_variants(images_u8, num_variants, device, rot_deg, max_trans, scale_lo, scale_hi):
    """[N, H, W] uint8  ->  [N, num_variants, H, W] uint8.

    Variant 0 is the clean original (copied losslessly). Variants 1..V-1 are
    independent random affine augmentations, each produced with one batched
    grid_sample over all N images.
    """
    n, h, w = images_u8.shape
    variants = torch.empty((n, num_variants, h, w), dtype=torch.uint8)
    variants[:, 0] = images_u8  # clean, exact

    if num_variants > 1:
        imgs = images_u8.to(device).float().div_(255.0).unsqueeze(1)  # [N,1,H,W] in [0,1]
        for v in range(1, num_variants):
            theta = random_affine_theta(n, rot_deg, max_trans, scale_lo, scale_hi, device)
            grid = F.affine_grid(theta, imgs.shape, align_corners=False)
            aug = F.grid_sample(
                imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            aug = aug.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8).squeeze(1)
            variants[:, v] = aug.cpu()
    return variants


def load_mnist_split(root, train):
    """Returns (images_u8 [N,28,28], targets_int64 [N]) for one MNIST split."""
    from torchvision import datasets  # imported lazily so the rest is testable
    ds = datasets.MNIST(root, train=train, download=True)
    return ds.data.clone(), ds.targets.clone().long()


def main():
    parser = argparse.ArgumentParser(description="Pre-compute MNIST augmentation variants.")
    parser.add_argument("--out-dir", type=str, default="./data",
                        help="Where MNIST is downloaded and .npz files are written.")
    parser.add_argument("--name", type=str, default="mnist",
                        help="Filename stem: fast_<name>_{train,val}.npz. "
                             "Must match FastMNISTLoader's filename.")
    parser.add_argument("--variants", type=int, default=6,
                        help="Number of training variants per image (incl. the clean one).")
    parser.add_argument("--rot-deg", type=float, default=12.0)
    parser.add_argument("--max-trans", type=float, default=0.12,
                        help="Max translation as a fraction of half-width (~0.12 ≈ 1.7px).")
    parser.add_argument("--scale-min", type=float, default=0.9)
    parser.add_argument("--scale-max", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compress", action="store_true",
                        help="Write compressed .npz (smaller on disk, slower to load).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device : {device}")
    print(f"Output : {args.out_dir}")
    print(f"Augment: rot ±{args.rot_deg}°, trans ±{args.max_trans}, "
          f"scale [{args.scale_min}, {args.scale_max}]\n")

    save = np.savez_compressed if args.compress else np.savez

    for split, train, nv in [("train", True, args.variants), ("val", False, 1)]:
        print(f"--> {split}: loading MNIST...")
        images, targets = load_mnist_split(args.out_dir, train)
        print(f"    {len(images)} images; building {nv} variant(s) each...")

        variants = make_variants(
            images, nv, device,
            args.rot_deg, args.max_trans, args.scale_min, args.scale_max,
        )

        path = os.path.join(args.out_dir, f"fast_{args.name}_{split}.npz")
        save(path, data=variants.numpy(), targets=targets.numpy())

        size_mb = os.path.getsize(path) / 1e6
        print(f"    wrote {path}")
        print(f"    data {tuple(variants.shape)} {variants.dtype}, "
              f"targets {tuple(targets.shape)}, {size_mb:.1f} MB on disk\n")

    print("Done. Ensure FastMNISTLoader loads "
          f"'fast_{args.name}_{{train,val}}.npz'.")


if __name__ == "__main__":
    main()

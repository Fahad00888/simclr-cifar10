"""
Task 2: Understanding Augmentations + Two-View Transform
Assignment 5 - From Supervised Learning to Self-Supervised Learning
Student: Fahad Khalid | Roll: MSDS25025
"""

import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset
from utils.visualization import save_augmentation_grid

SEED       = 2026
DATA_ROOT  = "./data"
SPLITS_DIR = "./splits"
RESULTS_DIR = "./results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)

simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)),
])

class TwoViewTransform:
    """
    Wraps a single transform and applies it TWICE independently to
    the same image, producing two different augmented views.

    This is the core of SimCLR's self-supervised signal:
    both views come from the same image but look different,
    so the model must learn invariant features to bring them together.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2

def visualize_augmentations():
    """
    Show 10 examples in format: Original | View 1 | View 2
    Uses the plain (no-normalize) dataset for 'original' column,
    and the SimCLR transform dataset for the two views.
    """

    plain_transform = T.ToTensor()
    plain_ds = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=f"{SPLITS_DIR}/train_labeled_10percent.txt",
        train=True,
        transform=plain_transform,
        download=False,
    )

    two_view = TwoViewTransform(simclr_transform)
    aug_ds = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=f"{SPLITS_DIR}/train_labeled_10percent.txt",
        train=True,
        transform=two_view,
        download=False,
    )

    set_seed(SEED)

    originals, view1s, view2s = [], [], []
    indices = list(range(10))

    for idx in indices:
        orig_img, _ = plain_ds[idx]
        (v1, v2), _ = aug_ds[idx]
        originals.append(orig_img)
        view1s.append(v1)
        view2s.append(v2)

    save_augmentation_grid(
        originals=originals,
        view1s=view1s,
        view2s=view2s,
        out_path=f"{RESULTS_DIR}/augmentation_examples.png",
        max_rows=10,
    )
    print(f"Augmentation grid saved → {RESULTS_DIR}/augmentation_examples.png")

    print("\n── Task 2 Observations ──────────────────────────────────────────")
    print("Q1: Are the two augmented views identical?")
    print("    No. Each view is independently sampled from the same transform,")
    print("    so crop regions, color jitter values, and grayscale decisions differ.")
    print()
    print("Q2: Do they still represent the same object?")
    print("    Yes. Both views come from the same original image, so the")
    print("    underlying object identity is preserved despite appearance changes.")
    print()
    print("Q3: Why should SimCLR treat them as a positive pair?")
    print("    Because they share the same semantic content. The model is trained")
    print("    to pull their representations together, forcing it to learn features")
    print("    that are invariant to augmentation (the useful, generalizable ones).")
    print()
    print("Q4: What if augmentations are too weak?")
    print("    The two views look almost identical. The model can match them using")
    print("    trivial shortcuts (e.g., pixel-level similarity) without learning")
    print("    any meaningful features — it becomes an easy task with no real learning.")
    print()
    print("Q5: What if augmentations are too strong?")
    print("    The two views look so different that they no longer reliably represent")
    print("    the same object. The model is asked to match views that genuinely look")
    print("    like different images, making the task impossible and training unstable.")

def main():
    print(f"Device : {DEVICE}")
    print(f"Seed   : {SEED}")

    print("\n── SimCLR Augmentation Pipeline ─────────────────────────────────")
    print(simclr_transform)

    print("\n── TwoViewTransform ─────────────────────────────────────────────")
    print("Applies the same stochastic transform TWICE to get two different views.")
    two_view = TwoViewTransform(simclr_transform)
    print(f"Transform: {two_view.transform}")

    print("\n── Generating Augmentation Examples ─────────────────────────────")
    visualize_augmentations()

if __name__ == "__main__":
    main()

"""
Task 3: Feature Similarity Before Training
Assignment 5 - From Supervised Learning to Self-Supervised Learning
Student: Fahad Khalid | Roll: MSDS25025

Purpose: Pass images through a RANDOM (untrained) ResNet-18 encoder and
compute cosine similarity between:
  - two augmented views of the SAME image (should be high after SimCLR)
  - views from DIFFERENT images (should be low after SimCLR)

This shows that BEFORE training, the encoder has no idea that two views
of the same image should be similar.
"""

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 2026
DATA_ROOT   = "/kaggle/input/datasets/fahadkhalid08/cifar10-assignment5/data"
SPLITS_DIR  = "./splits"
RESULTS_DIR = "./results"

BATCH_SIZE  = 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Device : {DEVICE}")
print(f"Seed   : {SEED}")


# ── SimCLR Augmentation (same as Task 2) ─────────────────────────────────────
simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)),
])


# ── Two-View Transform ────────────────────────────────────────────────────────
class TwoViewTransform:
    """Applies the same stochastic transform twice to get two different views."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


# ── Random Encoder (untrained ResNet-18 modified for CIFAR-10) ───────────────
def get_random_encoder():
    """
    ResNet-18 modified for CIFAR-10:
    - conv1: 3x3, stride 1, padding 1 (instead of 7x7 stride 2)
    - maxpool removed (replaced with Identity)
    - fc removed (we want 512-d features, not class logits)
    """
    model = torchvision.models.resnet18(weights=None)
    model.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc     = nn.Identity()   # output: 512-d feature vector
    return model.to(DEVICE)


# ── Extract Features ──────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(encoder, loader):
    """
    Returns:
        feats1: (N, 512) features for view 1
        feats2: (N, 512) features for view 2
    """
    encoder.eval()
    all_f1, all_f2 = [], []
    for (v1, v2), _ in loader:
        v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
        f1 = encoder(v1)
        f2 = encoder(v2)
        all_f1.append(f1.cpu())
        all_f2.append(f2.cpu())
    return torch.cat(all_f1), torch.cat(all_f2)


# ── Cosine Similarity ─────────────────────────────────────────────────────────
def cosine_similarity_rowwise(a, b):
    """
    Compute cosine similarity between corresponding rows of a and b.
    a, b: (N, D) tensors
    Returns: (N,) tensor of similarities
    """
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    return (a * b).sum(dim=1)


# ── Similarity Matrix Visualization ──────────────────────────────────────────
def plot_similarity_matrix(feats1, feats2, out_path, title):
    """
    Build and visualize a 2N x 2N cosine similarity matrix.
    Rows/cols 0..N-1  = view 1 of each image
    Rows/cols N..2N-1 = view 2 of each image

    Positive pairs: (i, N+i) and (N+i, i)
    Everything else: negatives
    """
    # Take first 32 samples for visualization
    n = min(32, feats1.shape[0])
    f1 = feats1[:n]
    f2 = feats2[:n]

    # Stack: [view1_0, ..., view1_n, view2_0, ..., view2_n]
    all_feats = torch.cat([f1, f2], dim=0)         # (2n, 512)
    all_feats = torch.nn.functional.normalize(all_feats, dim=1)

    # Compute 2n x 2n similarity matrix
    sim_matrix = torch.mm(all_feats, all_feats.T).numpy()  # (2n, 2n)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)

    # Mark the diagonal line separating view1 and view2 blocks
    ax.axhline(y=n - 0.5, color='blue', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.axvline(x=n - 0.5, color='blue', linewidth=1.5, linestyle='--', alpha=0.7)

    ax.set_title(title, fontsize=12, pad=15)
    ax.set_xlabel("Image Index (0-31: View1, 32-63: View2)")
    ax.set_ylabel("Image Index (0-31: View1, 32-63: View2)")

    # Add text annotations for quadrants
    ax.text(n/2, -2.5, "View 1", ha='center', fontsize=9, color='blue')
    ax.text(n + n/2, -2.5, "View 2", ha='center', fontsize=9, color='blue')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Similarity matrix saved → {out_path}")

    return sim_matrix


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load dataset with TwoViewTransform
    two_view = TwoViewTransform(simclr_transform)
    dataset = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=f"{SPLITS_DIR}/train_labeled_10percent.txt",
        train=True,
        transform=two_view,
        download=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # Get random untrained encoder
    encoder = get_random_encoder()
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters : {total_params:,}")
    print(f"Dataset size       : {len(dataset)}")

    # Extract features
    print("\nExtracting features from RANDOM encoder...")
    feats1, feats2 = extract_features(encoder, loader)
    print(f"Feature shape: {feats1.shape}")

    # ── Same image similarity ─────────────────────────────────────────────────
    same_sim = cosine_similarity_rowwise(feats1, feats2)
    avg_same = same_sim.mean().item()

    # ── Different image similarity ────────────────────────────────────────────
    # Shift feats2 by 1 to get different image pairs
    feats2_shifted = torch.roll(feats2, shifts=1, dims=0)
    diff_sim = cosine_similarity_rowwise(feats1, feats2_shifted)
    avg_diff = diff_sim.mean().item()

    print("\n── Similarity Results (Before SimCLR Training) ──────────────────")
    print(f"Same image, two augmented views : {avg_same:.4f}")
    print(f"Different images                : {avg_diff:.4f}")
    print(f"Gap (same - diff)               : {avg_same - avg_diff:.4f}")
    print("\nObservation: Before training, same-image similarity is barely")
    print("higher than different-image similarity. The encoder has learned")
    print("nothing yet — both values are close to random.")

    # ── Pair table (for report — Task 4.2) ───────────────────────────────────
    print("\n── Positive Pair Table (batch of 4 images) ──────────────────────")
    print(f"{'Original Image':<20} {'View 1 Index':<15} {'View 2 Index':<15} {'Positive Pair'}")
    print("-" * 65)
    for i in range(4):
        print(f"{'image ' + str(i):<20} {i:<15} {BATCH_SIZE + i:<15} {'yes'}")
    print("\nNote: For batch size N, views 0..N-1 are View1, N..2N-1 are View2")

    # ── Similarity Matrix ─────────────────────────────────────────────────────
    sim_matrix = plot_similarity_matrix(
        feats1, feats2,
        out_path=f"{RESULTS_DIR}/similarity_matrix_before_training.png",
        title="Cosine Similarity Matrix — Random Encoder (Before SimCLR Training)\n"
              "Blue dashed line separates View1 (left/top) from View2 (right/bottom)"
    )

    # ── Answer assignment questions ───────────────────────────────────────────
    print("\n── Task 4.3 Questions ───────────────────────────────────────────")
    print("Q1: Why is the diagonal ignored?")
    print("    The diagonal is sim(zi, zi) = 1.0 always (a vector with itself).")
    print("    Including it would trivially dominate the loss — it tells us nothing.")
    print()
    print("Q2: Where are the positive pairs located?")
    print("    At positions (i, N+i) and (N+i, i) — off-diagonal blocks.")
    print("    i.e. top-right and bottom-left quadrants of the matrix.")
    print()
    print("Q3: Why are all other entries treated as negatives?")
    print("    They come from different original images, so their representations")
    print("    should be pushed apart. More negatives = stronger learning signal.")

    # Save results for metrics.json
    import json
    results = {
        "same_view_similarity_before": round(avg_same, 4),
        "different_image_similarity_before": round(avg_diff, 4),
    }
    with open(f"{RESULTS_DIR}/task3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS_DIR}/task3_results.json")

    # Add to notes
    print("\n── ADD TO YOUR NOTES ────────────────────────────────────────────")
    print(f"Task 3 — Similarity BEFORE SimCLR:")
    print(f"  Same image, two views : {avg_same:.4f}")
    print(f"  Different images      : {avg_diff:.4f}")


if __name__ == "__main__":
    main()

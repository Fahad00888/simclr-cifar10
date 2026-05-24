"""
Task 4 + Task 5: SimCLR Implementation and Pretraining
Assignment 5 - From Supervised Learning to Self-Supervised Learning
Student: Fahad Khalid | Roll: MSDS25025

Implements from scratch:
  - ResNet-18 encoder modified for CIFAR-10
  - Projection head: Linear(512->256) -> ReLU -> Linear(256->128)
  - Positive and negative pair construction
  - Cosine similarity matrix (2N x 2N)
  - NT-Xent contrastive loss
  - SimCLR training loop (no labels used)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset

SEED           = 2026
DATA_ROOT      = "/kaggle/input/datasets/fahadkhalid08/cifar10-assignment5/data"
SPLITS_DIR     = "./splits"
RESULTS_DIR    = "./results"
GRAPHS_DIR     = "./graphs"
MODELS_DIR     = "./models"

BATCH_SIZE     = 64
EPOCHS         = 50
LR             = 3e-4
TEMPERATURE    = 0.5
PROJ_DIM       = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

print(f"Device      : {DEVICE}")
print(f"Batch size  : {BATCH_SIZE}")
print(f"Epochs      : {EPOCHS}")
print(f"Temperature : {TEMPERATURE}")
print(f"Projection  : {PROJ_DIM}")

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
    """Returns two independently augmented views of the same image."""
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return self.transform(x), self.transform(x)

class CIFARResNet18Encoder(nn.Module):
    """
    ResNet-18 modified for CIFAR-10 small images (32x32):
      - conv1: 3x3 kernel, stride 1, padding 1 (not 7x7 stride 2)
      - maxpool: removed (Identity)
      - fc: removed (Identity) → outputs 512-d feature vector h
    """
    def __init__(self):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        base.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc      = nn.Identity()
        self.encoder = base

    def forward(self, x):
        return self.encoder(x)

class ProjectionHead(nn.Module):
    """
    MLP projection head used ONLY during SimCLR pretraining.
    Maps 512-d encoder output h → 128-d projection z.

    Architecture: Linear(512->256) -> ReLU -> Linear(256->128)

    Why throw it away after training?
    The projection head maps features to a space optimized for contrastive loss.
    This space loses some information (color, orientation) that is useful for
    downstream tasks. The encoder output h (512-d) retains more useful info.
    """
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h):
        return self.net(h)

class SimCLR(nn.Module):
    """Combines encoder + projection head for contrastive pretraining."""
    def __init__(self):
        super().__init__()
        self.encoder    = CIFARResNet18Encoder()
        self.projector  = ProjectionHead(512, 256, PROJ_DIM)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    From: Chen et al., SimCLR, ICML 2020.

    For a batch of N images producing 2N views:
      - Positive pair (i, j): two views of the same image
      - Negatives: all other 2N-2 views in the batch

    Formula for positive pair (i, j):
      loss(i,j) = -log [ exp(sim(zi,zj)/tau) / sum_{k!=i} exp(sim(zi,zk)/tau) ]

    Final loss = mean over all 2N positive pairs (both directions)
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1: (N, D) projections for view 1
        z2: (N, D) projections for view 2
        """
        N = z1.shape[0]

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z  = torch.cat([z1, z2], dim=0)

        sim = torch.mm(z, z.T) / self.temperature

        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim  = sim.masked_fill(mask, float('-inf'))

        labels = torch.cat([
            torch.arange(N, 2*N, device=z.device),
            torch.arange(0, N,   device=z.device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss

@torch.no_grad()
def visualize_similarity_matrix(model, loader, out_path, title, n_samples=32):
    """Visualize the 2N x 2N similarity matrix for one small batch."""
    model.eval()
    (v1, v2), _ = next(iter(loader))
    v1 = v1[:n_samples].to(DEVICE)
    v2 = v2[:n_samples].to(DEVICE)

    _, z1 = model(v1)
    _, z2 = model(v2)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z  = torch.cat([z1, z2], dim=0).cpu()

    sim = torch.mm(z, z.T).numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)

    n = n_samples
    ax.axhline(y=n - 0.5, color='blue', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.axvline(x=n - 0.5, color='blue', linewidth=1.5, linestyle='--', alpha=0.8)

    ax.set_title(title, fontsize=11, pad=15)
    ax.set_xlabel("Index (0-31: View1, 32-63: View2)")
    ax.set_ylabel("Index (0-31: View1, 32-63: View2)")
    ax.text(n/2, -2.5, "View 1", ha='center', fontsize=9, color='blue')
    ax.text(n + n/2, -2.5, "View 2", ha='center', fontsize=9, color='blue')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Similarity matrix saved → {out_path}")
    model.train()

@torch.no_grad()
def measure_similarity(model, loader, n_batches=5):
    """
    Measure average cosine similarity between:
    - same image, two augmented views
    - different images
    """
    model.eval()
    same_sims, diff_sims = [], []

    for i, ((v1, v2), _) in enumerate(loader):
        if i >= n_batches:
            break
        v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
        h1, _ = model(v1)
        h2, _ = model(v2)

        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)

        same = (h1 * h2).sum(dim=1)
        same_sims.extend(same.cpu().tolist())

        h2_shifted = torch.roll(h2, shifts=1, dims=0)
        diff = (h1 * h2_shifted).sum(dim=1)
        diff_sims.extend(diff.cpu().tolist())

    model.train()
    return np.mean(same_sims), np.mean(diff_sims)

def train_simclr(model, loader, optimizer, criterion):
    train_losses = []

    print("\n── SimCLR Pretraining ───────────────────────────────────────────")
    print("Labels are NOT used during pretraining.")
    print(f"Training on {len(loader.dataset)} unlabeled images\n")

    for epoch in tqdm(range(1, EPOCHS + 1), desc="SimCLR Epochs"):
        model.train()
        total_loss = 0.0

        for (v1, v2), _ in loader:
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)

            _, z1 = model(v1)
            _, z2 = model(v2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * v1.size(0)

        avg_loss = total_loss / len(loader.dataset)
        train_losses.append(avg_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f}")

    return train_losses

def main():

    two_view = TwoViewTransform(simclr_transform)
    ssl_dataset = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=f"{SPLITS_DIR}/train_ssl_unlabeled.txt",
        train=True,
        transform=two_view,
        download=False,
    )
    loader = DataLoader(
        ssl_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    print(f"Unlabeled training samples: {len(ssl_dataset)}")

    model     = SimCLR().to(DEVICE)
    criterion = NTXentLoss(temperature=TEMPERATURE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    enc_params  = sum(p.numel() for p in model.encoder.parameters())
    proj_params = sum(p.numel() for p in model.projector.parameters())
    print(f"Encoder parameters   : {enc_params:,}")
    print(f"Projector parameters : {proj_params:,}")

    print("\n── Feature Similarity BEFORE SimCLR Training ────────────────────")
    same_before, diff_before = measure_similarity(model, loader)
    print(f"Same image, two views : {same_before:.4f}")
    print(f"Different images      : {diff_before:.4f}")

    visualize_similarity_matrix(
        model, loader,
        out_path=f"{RESULTS_DIR}/similarity_matrix_before_training.png",
        title="Similarity Matrix — Before SimCLR Training (Random Encoder)\n"
              "Positive pairs are at off-diagonal positions (top-right / bottom-left blocks)"
    )

    train_losses = train_simclr(model, loader, optimizer, criterion)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, EPOCHS + 1), train_losses, color='steelblue', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent Loss")
    ax.set_title("SimCLR Pretraining Loss\n(ResNet-18, CIFAR-10, 45k Unlabeled Images)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{GRAPHS_DIR}/simclr_pretraining_loss.png", dpi=200)
    plt.close(fig)
    print(f"\nLoss curve saved → {GRAPHS_DIR}/simclr_pretraining_loss.png")

    print("\n── Feature Similarity AFTER SimCLR Training ─────────────────────")
    same_after, diff_after = measure_similarity(model, loader)
    print(f"Same image, two views : {same_after:.4f}")
    print(f"Different images      : {diff_after:.4f}")

    visualize_similarity_matrix(
        model, loader,
        out_path=f"{RESULTS_DIR}/similarity_matrix_after_training.png",
        title="Similarity Matrix — After SimCLR Training\n"
              "Same-image views (off-diagonal blocks) should now be brighter"
    )

    torch.save(model.encoder.state_dict(), f"{MODELS_DIR}/simclr_encoder.pt")
    print(f"\nEncoder saved → {MODELS_DIR}/simclr_encoder.pt")

    print("\n── Similarity Comparison Table ──────────────────────────────────")
    print(f"{'Pair Type':<35} {'Before SimCLR':>15} {'After SimCLR':>15}")
    print("-" * 65)
    print(f"{'Same image, two augmented views':<35} {same_before:>15.4f} {same_after:>15.4f}")
    print(f"{'Different images':<35} {diff_before:>15.4f} {diff_after:>15.4f}")

    results = {
        "same_view_similarity_before": round(float(same_before), 4),
        "different_image_similarity_before": round(float(diff_before), 4),
        "same_view_similarity_after": round(float(same_after), 4),
        "different_image_similarity_after": round(float(diff_after), 4),
        "final_loss": round(train_losses[-1], 4),
    }
    with open(f"{RESULTS_DIR}/task4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS_DIR}/task4_results.json")

    print("\n── ADD TO YOUR NOTES ────────────────────────────────────────────")
    print(f"Task 5 — Similarity AFTER SimCLR:")
    print(f"  Same image, two views : {same_after:.4f}")
    print(f"  Different images      : {diff_after:.4f}")
    print(f"  Final training loss   : {train_losses[-1]:.4f}")

if __name__ == "__main__":
    main()

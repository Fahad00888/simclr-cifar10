
MSDS25025_05_allCode.py
All Tasks Combined - Assignment 5 SimCLR
Student: Fahad Khalid | Roll: MSDS25025
================================================================================
# MSDS25025_05_task1_supervised.py
================================================================================

"""
Task 1: Supervised Baseline with Limited Labels
Assignment 5 - From Supervised Learning to Self-Supervised Learning
Student: Fahad Khalid | Roll: MSDS25025
"""

import os
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset
from utils.metrics import (
    compute_accuracy,
    save_confusion_matrix,
    top1_accuracy_from_logits,
)

SEED        = 2026
DATA_ROOT   = "./data"
SPLITS_DIR  = "./splits"
RESULTS_DIR = "./results"
GRAPHS_DIR  = "./graphs"
MODELS_DIR  = "./models"

BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 3e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)),
])

eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)),
])

def get_dataloaders():
    train_ds = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=f"{SPLITS_DIR}/train_labeled_10percent.txt",
        train=True,
        transform=train_transform,
        download=False,
    )
    val_ds = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=f"{SPLITS_DIR}/val.txt",
        train=True,
        transform=eval_transform,
        download=False,
    )
    test_ds = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=f"{SPLITS_DIR}/test.txt",
        train=False,
        transform=eval_transform,
        download=False,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")
    print(f"Test  samples : {len(test_ds)}")
    return train_loader, val_loader, test_loader

def get_model():
    """ResNet-18 modified for CIFAR-10 with a classification head."""
    model = torchvision.models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(512, 10)
    return model.to(DEVICE)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    acc      = top1_accuracy_from_logits(all_logits, all_labels)
    return avg_loss, acc, all_logits, all_labels

def main():
    train_loader, val_loader, test_loader = get_dataloaders()

    model     = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    train_losses, val_losses, val_accs = [], [], []
    best_val_acc = 0.0

    print("\n── Training Supervised Baseline ──")
    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{MODELS_DIR}/supervised_best.pt")

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", color="steelblue")
    ax.plot(range(1, EPOCHS + 1), val_losses,   label="Val Loss",   color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Supervised Baseline — Training & Validation Loss\n(ResNet-18, 10% Labels, CIFAR-10)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{GRAPHS_DIR}/supervised_loss.png", dpi=200)
    plt.close(fig)
    print(f"\nLoss curve saved → {GRAPHS_DIR}/supervised_loss.png")

    model.load_state_dict(torch.load(f"{MODELS_DIR}/supervised_best.pt",
                                     map_location=DEVICE))
    test_loss, test_acc, test_logits, test_labels = evaluate(model, test_loader)

    preds = test_logits.argmax(dim=1).numpy()
    trues = test_labels.numpy()

    save_confusion_matrix(
        y_true=trues,
        y_pred=preds,
        out_path=f"{RESULTS_DIR}/supervised_confusion_matrix.png",
        title="Supervised Baseline — Confusion Matrix (Test Set)",
    )
    print(f"Confusion matrix saved → {RESULTS_DIR}/supervised_confusion_matrix.png")

    print("\n── Results ──────────────────────────────────────────────────")
    print(f"Best Val Accuracy  : {best_val_acc:.4f}  ({best_val_acc*100:.2f}%)")
    print(f"Final Test Accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"Final Test Loss    : {test_loss:.4f}")

    result = {
        "task": "supervised_baseline",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "best_val_acc": round(best_val_acc, 4),
        "test_acc": round(test_acc, 4),
    }
    with open(f"{RESULTS_DIR}/task1_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {RESULTS_DIR}/task1_results.json")

if __name__ == "__main__":
    main()


================================================================================
# MSDS25025_05_task2_augmentations.py
================================================================================

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


================================================================================
# MSDS25025_05_task3_similarity.py
================================================================================

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
    """Applies the same stochastic transform twice to get two different views."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2

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
    model.fc     = nn.Identity()
    return model.to(DEVICE)

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

def cosine_similarity_rowwise(a, b):
    """
    Compute cosine similarity between corresponding rows of a and b.
    a, b: (N, D) tensors
    Returns: (N,) tensor of similarities
    """
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    return (a * b).sum(dim=1)

def plot_similarity_matrix(feats1, feats2, out_path, title):
    """
    Build and visualize a 2N x 2N cosine similarity matrix.
    Rows/cols 0..N-1  = view 1 of each image
    Rows/cols N..2N-1 = view 2 of each image

    Positive pairs: (i, N+i) and (N+i, i)
    Everything else: negatives
    """

    n = min(32, feats1.shape[0])
    f1 = feats1[:n]
    f2 = feats2[:n]

    all_feats = torch.cat([f1, f2], dim=0)
    all_feats = torch.nn.functional.normalize(all_feats, dim=1)

    sim_matrix = torch.mm(all_feats, all_feats.T).numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.axhline(y=n - 0.5, color='blue', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.axvline(x=n - 0.5, color='blue', linewidth=1.5, linestyle='--', alpha=0.7)

    ax.set_title(title, fontsize=12, pad=15)
    ax.set_xlabel("Image Index (0-31: View1, 32-63: View2)")
    ax.set_ylabel("Image Index (0-31: View1, 32-63: View2)")

    ax.text(n/2, -2.5, "View 1", ha='center', fontsize=9, color='blue')
    ax.text(n + n/2, -2.5, "View 2", ha='center', fontsize=9, color='blue')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Similarity matrix saved → {out_path}")

    return sim_matrix

def main():

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

    encoder = get_random_encoder()
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters : {total_params:,}")
    print(f"Dataset size       : {len(dataset)}")

    print("\nExtracting features from RANDOM encoder...")
    feats1, feats2 = extract_features(encoder, loader)
    print(f"Feature shape: {feats1.shape}")

    same_sim = cosine_similarity_rowwise(feats1, feats2)
    avg_same = same_sim.mean().item()

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

    print("\n── Positive Pair Table (batch of 4 images) ──────────────────────")
    print(f"{'Original Image':<20} {'View 1 Index':<15} {'View 2 Index':<15} {'Positive Pair'}")
    print("-" * 65)
    for i in range(4):
        print(f"{'image ' + str(i):<20} {i:<15} {BATCH_SIZE + i:<15} {'yes'}")
    print("\nNote: For batch size N, views 0..N-1 are View1, N..2N-1 are View2")

    sim_matrix = plot_similarity_matrix(
        feats1, feats2,
        out_path=f"{RESULTS_DIR}/similarity_matrix_before_training.png",
        title="Cosine Similarity Matrix — Random Encoder (Before SimCLR Training)\n"
              "Blue dashed line separates View1 (left/top) from View2 (right/bottom)"
    )

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

    import json
    results = {
        "same_view_similarity_before": round(avg_same, 4),
        "different_image_similarity_before": round(avg_diff, 4),
    }
    with open(f"{RESULTS_DIR}/task3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS_DIR}/task3_results.json")

    print("\n── ADD TO YOUR NOTES ────────────────────────────────────────────")
    print(f"Task 3 — Similarity BEFORE SimCLR:")
    print(f"  Same image, two views : {avg_same:.4f}")
    print(f"  Different images      : {avg_diff:.4f}")

if __name__ == "__main__":
    main()


================================================================================
# MSDS25025_05_task4_simclr.py
================================================================================

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


================================================================================
# MSDS25025_05_task5_linear_probe.py
================================================================================

"""
Task 6: Linear Probe Evaluation
Assignment 5 - From Supervised Learning to Self-Supervised Learning
Student: Fahad Khalid | Roll: MSDS25025

Experiment A: Random Encoder Linear Probe
  - Random frozen ResNet-18 encoder
  - Train only Linear(512 -> 10)

Experiment B: SimCLR Encoder Linear Probe
  - SimCLR pretrained frozen encoder
  - Train only Linear(512 -> 10)

Key rule: Encoder is FROZEN — only the linear head is trained.
This tests whether SimCLR learned useful features WITHOUT any labels.
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
from utils.metrics import compute_accuracy, save_confusion_matrix

SEED          = 2026
DATA_ROOT     = "/kaggle/input/datasets/fahadkhalid08/cifar10-assignment5/data"
SPLITS_DIR    = "./splits"
RESULTS_DIR   = "./results"
GRAPHS_DIR    = "./graphs"
MODELS_DIR    = "./models"

BATCH_SIZE    = 64
EPOCHS        = 20
LR            = 3e-4
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIMCLR_ENCODER_PATH = "/kaggle/input/simclr-encoder-msds25025/simclr_encoder.pt"

set_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

print(f"Device : {DEVICE}")
print(f"Seed   : {SEED}")

eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)),
])

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)),
])

def get_encoder(pretrained_path=None):
    """
    Returns ResNet-18 encoder modified for CIFAR-10.
    If pretrained_path is given, loads SimCLR pretrained weights.
    Otherwise returns random untrained encoder.
    """
    model = torchvision.models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Identity()

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location=DEVICE)

        new_state = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                new_state[k[8:]] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=False)
        print(f"Loaded SimCLR weights from {pretrained_path}")
    else:
        print("Using random untrained encoder")

    return model.to(DEVICE)

class LinearProbe(nn.Module):
    """
    Frozen encoder + trainable linear classifier.
    Only the linear head learns — encoder weights are frozen.
    This tests if encoder features are linearly separable.
    """
    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(512, num_classes)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)
        return self.classifier(h)

def get_dataloaders():
    train_ds = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=f"{SPLITS_DIR}/train_labeled_10percent.txt",
        train=True,
        transform=train_transform,
        download=False,
    )
    val_ds = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=f"{SPLITS_DIR}/val.txt",
        train=True,
        transform=eval_transform,
        download=False,
    )
    test_ds = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=f"{SPLITS_DIR}/test.txt",
        train=False,
        transform=eval_transform,
        download=False,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds = all_logits.argmax(dim=1)
    acc   = (preds == all_labels).float().mean().item()
    return acc, all_logits, all_labels

def run_linear_probe(name, encoder, train_loader, val_loader, test_loader):
    print(f"\n── {name} ──────────────────────────────────────────")

    model     = LinearProbe(encoder).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params : {trainable:,} (linear head only)")
    print(f"Frozen params    : {frozen:,} (encoder)")

    val_accs  = []
    best_val  = 0.0
    train_losses = []

    for epoch in tqdm(range(1, EPOCHS + 1), desc=f"{name}"):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc, _, _ = evaluate(model, val_loader)
        train_losses.append(loss)
        val_accs.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), f"{MODELS_DIR}/linear_probe_{name.replace(' ', '_')}.pt")

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

    model.load_state_dict(torch.load(
        f"{MODELS_DIR}/linear_probe_{name.replace(' ', '_')}.pt",
        map_location=DEVICE))
    test_acc, test_logits, test_labels = evaluate(model, test_loader)

    print(f"\n  Best Val Acc  : {best_val:.4f} ({best_val*100:.2f}%)")
    print(f"  Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")

    return test_acc, val_accs, train_losses, test_logits, test_labels

def main():
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    random_encoder = get_encoder(pretrained_path=None)
    random_acc, random_val_accs, random_losses, _, _ = run_linear_probe(
        "Random Encoder", random_encoder, train_loader, val_loader, test_loader
    )

    simclr_encoder = get_encoder(pretrained_path=SIMCLR_ENCODER_PATH)
    simclr_acc, simclr_val_accs, simclr_losses, simclr_logits, simclr_labels = run_linear_probe(
        "SimCLR Encoder", simclr_encoder, train_loader, val_loader, test_loader
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, EPOCHS+1), random_val_accs, label="Random Encoder",
            color='tomato', linewidth=2)
    ax.plot(range(1, EPOCHS+1), simclr_val_accs, label="SimCLR Encoder",
            color='steelblue', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Linear Probe Validation Accuracy\n(Frozen Encoder + Linear Head, 10% Labels)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{GRAPHS_DIR}/linear_probe_accuracy.png", dpi=200)
    plt.close(fig)
    print(f"\nAccuracy plot saved → {GRAPHS_DIR}/linear_probe_accuracy.png")

    print("\n── Linear Probe Results ─────────────────────────────────────────")
    print(f"{'Model':<35} {'Encoder':<25} {'Trainable Part':<20} {'Test Acc':>10}")
    print("-" * 90)
    print(f"{'Random Linear Probe':<35} {'Random frozen':<25} {'Linear only':<20} {random_acc:>10.4f}")
    print(f"{'SimCLR Linear Probe':<35} {'SimCLR frozen':<25} {'Linear only':<20} {simclr_acc:>10.4f}")

    torch.save(
        torch.load(f"{MODELS_DIR}/linear_probe_SimCLR_Encoder.pt", map_location=DEVICE),
        f"{MODELS_DIR}/linear_probe.pt"
    )

    results = {
        "random_linear_probe_test_acc": round(random_acc, 4),
        "simclr_linear_probe_test_acc": round(simclr_acc, 4),
    }
    with open(f"{RESULTS_DIR}/task5_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS_DIR}/task5_results.json")

    print("\n── ADD TO YOUR NOTES ────────────────────────────────────────────")
    print(f"Task 6 — Linear Probe:")
    print(f"  Random encoder test acc : {random_acc:.4f} ({random_acc*100:.2f}%)")
    print(f"  SimCLR encoder test acc : {simclr_acc:.4f} ({simclr_acc*100:.2f}%)")

if __name__ == "__main__":
    main()


================================================================================
# MSDS25025_05_task6_finetune.py
================================================================================

"""
Task 7 + Task 8: Fine-tuning + PCA/t-SNE Visualization + Final Metrics
Assignment 5 - From Supervised Learning to Self-Supervised Learning
Student: Fahad Khalid | Roll: MSDS25025

Task 7: Fine-tune SimCLR encoder end-to-end on 10% labeled data
Task 8: PCA/t-SNE visualization of features from:
  1. Random untrained encoder
  2. SimCLR pretrained encoder
  3. Fine-tuned encoder
"""

import os
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset
from utils.metrics import compute_accuracy, save_confusion_matrix

SEED          = 2026
DATA_ROOT     = "/kaggle/input/datasets/fahadkhalid08/cifar10-assignment5/data"
SPLITS_DIR    = "./splits"
RESULTS_DIR   = "./results"
GRAPHS_DIR    = "./graphs"
MODELS_DIR    = "./models"

BATCH_SIZE    = 64
EPOCHS        = 20
LR            = 3e-4
VIZ_SAMPLES   = 1000
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIMCLR_ENCODER_PATH = "/kaggle/input/simclr-encoder-msds25025/simclr_encoder.pt"

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

set_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

print(f"Device : {DEVICE}")
print(f"Seed   : {SEED}")

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)),
])

eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)),
])

def get_encoder(pretrained_path=None):
    model = torchvision.models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Identity()

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location=DEVICE)
        new_state  = {}
        for k, v in state_dict.items():
            key = k[8:] if k.startswith('encoder.') else k
            new_state[key] = v
        model.load_state_dict(new_state, strict=False)
        print(f"Loaded weights from {pretrained_path}")
    else:
        print("Using random untrained encoder")

    return model.to(DEVICE)

class Classifier(nn.Module):
    """Encoder + classification head for supervised training / fine-tuning."""
    def __init__(self, encoder, num_classes=10, freeze_encoder=False):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(512, num_classes)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h)

def get_dataloaders():
    train_ds = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=f"{SPLITS_DIR}/train_labeled_10percent.txt",
        train=True, transform=train_transform, download=False,
    )
    val_ds = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=f"{SPLITS_DIR}/val.txt",
        train=True, transform=eval_transform, download=False,
    )
    test_ds = get_cifar10_subset(
        data_root=DATA_ROOT,
        split_file=f"{SPLITS_DIR}/test.txt",
        train=False, transform=eval_transform, download=False,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(DEVICE))
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    acc = (all_logits.argmax(1) == all_labels).float().mean().item()
    return acc, all_logits, all_labels

def run_finetuning(train_loader, val_loader, test_loader):
    print("\n── Fine-tuning SimCLR Encoder ───────────────────────────────────")
    encoder   = get_encoder(pretrained_path=SIMCLR_ENCODER_PATH)
    model     = Classifier(encoder, freeze_encoder=False).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} (full model — encoder + head)")

    val_accs = []
    best_val = 0.0

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Fine-tuning"):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc, _, _ = evaluate(model, val_loader)
        val_accs.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), f"{MODELS_DIR}/finetuned_model.pt")

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

    model.load_state_dict(torch.load(f"{MODELS_DIR}/finetuned_model.pt",
                                      map_location=DEVICE))
    test_acc, test_logits, test_labels = evaluate(model, test_loader)

    print(f"\n  Best Val Acc  : {best_val:.4f} ({best_val*100:.2f}%)")
    print(f"  Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")

    return test_acc, val_accs, test_logits, test_labels, model

def plot_finetuning_accuracy(finetune_val_accs, linear_probe_acc, supervised_acc):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, EPOCHS+1), finetune_val_accs,
            label="SimCLR Fine-tuning", color='steelblue', linewidth=2)
    ax.axhline(y=linear_probe_acc, color='tomato', linestyle='--',
               linewidth=2, label=f"SimCLR Linear Probe ({linear_probe_acc*100:.1f}%)")
    ax.axhline(y=supervised_acc, color='green', linestyle='--',
               linewidth=2, label=f"Supervised Baseline ({supervised_acc*100:.1f}%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Fine-tuning vs Baselines\n(SimCLR Pretrained Encoder, 10% Labels)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{GRAPHS_DIR}/finetuning_accuracy.png", dpi=200)
    plt.close(fig)
    print(f"Fine-tuning plot saved → {GRAPHS_DIR}/finetuning_accuracy.png")

@torch.no_grad()
def extract_features_for_viz(encoder, val_loader, n_samples=1000):
    """Extract 512-d features from encoder for PCA/t-SNE."""
    encoder.eval()
    all_feats, all_labels = [], []

    for images, labels in val_loader:
        feats = encoder(images.to(DEVICE))
        all_feats.append(feats.cpu())
        all_labels.append(labels)
        if sum(f.shape[0] for f in all_feats) >= n_samples:
            break

    feats  = torch.cat(all_feats)[:n_samples].numpy()
    labels = torch.cat(all_labels)[:n_samples].numpy()
    return feats, labels

def visualize_features(feats, labels, out_path, title, method='tsne'):
    """Reduce 512-d features to 2D and plot with class colors."""
    set_seed(SEED)

    if method == 'tsne':
        print(f"  Running t-SNE... (this takes ~1-2 min)")
        reducer = TSNE(n_components=2, random_state=SEED, perplexity=30,
                       n_iter=1000, verbose=0)
    else:
        reducer = PCA(n_components=2, random_state=SEED)

    feats_2d = reducer.fit_transform(feats)

    colors = plt.cm.get_cmap('tab10', 10)
    fig, ax = plt.subplots(figsize=(10, 8))

    for cls in range(10):
        mask = labels == cls
        ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1],
                   c=[colors(cls)], label=CIFAR10_CLASSES[cls],
                   alpha=0.6, s=15)

    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right', fontsize=8, markerscale=2)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")

def save_test_predictions(logits, labels, out_path):
    """
    Save test predictions in required format:
    image_index, true_label, predicted_label, prob_class_0, ..., prob_class_9
    """
    probs  = F.softmax(logits, dim=1).numpy()
    preds  = logits.argmax(dim=1).numpy()
    trues  = labels.numpy()

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['image_index', 'true_label', 'predicted_label'] + \
                 [f'prob_class_{i}' for i in range(10)]
        writer.writerow(header)
        for i in range(len(trues)):
            row = [i, trues[i], preds[i]] + [round(float(p), 6) for p in probs[i]]
            writer.writerow(row)
    print(f"Predictions saved → {out_path}")

def save_metrics(supervised_acc, random_probe_acc, simclr_probe_acc, finetune_acc):
    metrics = {
        "student_name": "Fahad Khalid",
        "roll_number": "MSDS25025",
        "seed": 2026,
        "batch_size": BATCH_SIZE,
        "simclr_epochs": 50,
        "linear_probe_epochs": 20,
        "finetuning_epochs": 20,
        "learning_rate": LR,
        "temperature": 0.5,
        "supervised_10percent_test_acc": round(supervised_acc, 4),
        "random_linear_probe_test_acc": round(random_probe_acc, 4),
        "simclr_linear_probe_test_acc": round(simclr_probe_acc, 4),
        "simclr_finetune_test_acc": round(finetune_acc, 4),
        "same_view_similarity_before": 0.9888,
        "different_image_similarity_before": 0.9854,
        "same_view_similarity_after": 0.9087,
        "different_image_similarity_after": 0.3363,
        "github_repo_url": "https://github.com/Fahad00888/DL_Assignment5_SimCLR_MSDS25025",
        "first_commit_date": "2026-05-20",
        "last_commit_before_deadline": "2026-05-31",
        "number_of_meaningful_commits": 10,
        "gpu_used": "Kaggle GPU T4 x2",
        "approximate_training_time": "~90 minutes total"
    }
    out_path = f"{RESULTS_DIR}/metrics.json"
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"metrics.json saved → {out_path}")
    return metrics

def main():
    train_loader, val_loader, test_loader = get_dataloaders()

    with open(f"{RESULTS_DIR}/task5_results.json") as f:
        probe_results = json.load(f)
    random_probe_acc = probe_results["random_linear_probe_test_acc"]
    simclr_probe_acc = probe_results["simclr_linear_probe_test_acc"]

    with open(f"{RESULTS_DIR}/task1_results.json") as f:
        task1_results = json.load(f)
    supervised_acc = task1_results["test_acc"]

    print(f"Loaded — Supervised: {supervised_acc:.4f} | "
          f"Random probe: {random_probe_acc:.4f} | "
          f"SimCLR probe: {simclr_probe_acc:.4f}")

    finetune_acc, finetune_val_accs, test_logits, test_labels, finetuned_model = \
        run_finetuning(train_loader, val_loader, test_loader)

    plot_finetuning_accuracy(finetune_val_accs, simclr_probe_acc, supervised_acc)

    preds = test_logits.argmax(dim=1).numpy()
    trues = test_labels.numpy()
    save_confusion_matrix(
        y_true=trues, y_pred=preds,
        out_path=f"{RESULTS_DIR}/finetuned_confusion_matrix.png",
        title="Fine-tuned Model — Confusion Matrix (Test Set)",
    )

    save_test_predictions(
        test_logits, test_labels,
        out_path=f"{RESULTS_DIR}/test_predictions.csv"
    )

    print("\n── Generating PCA/t-SNE Visualizations ──────────────────────────")

    print("\n1. Random encoder features...")
    random_encoder = get_encoder(pretrained_path=None)
    random_feats, viz_labels = extract_features_for_viz(random_encoder, val_loader, VIZ_SAMPLES)
    visualize_features(random_feats, viz_labels,
        out_path=f"{RESULTS_DIR}/random_encoder_pca_or_tsne.png",
        title="t-SNE: Random Encoder Features (1000 Val Samples)\nNo class structure expected",
        method='tsne')

    print("\n2. SimCLR encoder features...")
    simclr_encoder = get_encoder(pretrained_path=SIMCLR_ENCODER_PATH)
    simclr_feats, _ = extract_features_for_viz(simclr_encoder, val_loader, VIZ_SAMPLES)
    visualize_features(simclr_feats, viz_labels,
        out_path=f"{RESULTS_DIR}/simclr_encoder_pca_or_tsne.png",
        title="t-SNE: SimCLR Encoder Features (1000 Val Samples)\nSome class clustering expected",
        method='tsne')

    print("\n3. Fine-tuned encoder features...")
    finetuned_encoder = finetuned_model.encoder
    finetune_feats, _ = extract_features_for_viz(finetuned_encoder, val_loader, VIZ_SAMPLES)
    visualize_features(finetune_feats, viz_labels,
        out_path=f"{RESULTS_DIR}/finetuned_encoder_pca_or_tsne.png",
        title="t-SNE: Fine-tuned Encoder Features (1000 Val Samples)\nStrong class clustering expected",
        method='tsne')

    print("\n── Final Results Table ──────────────────────────────────────────")
    print(f"{'Model':<45} {'Labels in Pretraining?':<25} {'Frozen?':<10} {'Test Acc':>10}")
    print("-" * 95)
    print(f"{'Supervised ResNet-18, 10% labels':<45} {'Yes':<25} {'No':<10} {supervised_acc:>10.4f}")
    print(f"{'Random encoder + linear probe':<45} {'No':<25} {'Yes':<10} {random_probe_acc:>10.4f}")
    print(f"{'SimCLR encoder + linear probe':<45} {'No':<25} {'Yes':<10} {simclr_probe_acc:>10.4f}")
    print(f"{'SimCLR encoder + fine-tuning':<45} {'No then Yes':<25} {'No':<10} {finetune_acc:>10.4f}")

    metrics = save_metrics(supervised_acc, random_probe_acc, simclr_probe_acc, finetune_acc)

    print("\n── ADD TO YOUR NOTES ────────────────────────────────────────────")
    print(f"Task 7 — Fine-tuning:")
    print(f"  SimCLR + fine-tune test acc : {finetune_acc:.4f} ({finetune_acc*100:.2f}%)")
    print(f"\nFinal Summary:")
    print(f"  Supervised 10%    : {supervised_acc*100:.2f}%")
    print(f"  Random probe      : {random_probe_acc*100:.2f}%")
    print(f"  SimCLR probe      : {simclr_probe_acc*100:.2f}%")
    print(f"  SimCLR fine-tune  : {finetune_acc*100:.2f}%")

if __name__ == "__main__":
    main()



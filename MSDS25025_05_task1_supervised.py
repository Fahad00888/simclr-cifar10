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

# ── Config ──────────────────────────────────────────────────────────────────
SEED        = 2026
DATA_ROOT   = "./data"          # change to /kaggle/input/cifar-10-python on Kaggle
SPLITS_DIR  = "./splits"
RESULTS_DIR = "./results"
GRAPHS_DIR  = "./graphs"
MODELS_DIR  = "./models"

BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 3e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Setup ────────────────────────────────────────────────────────────────────
set_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")


# ── Transforms ───────────────────────────────────────────────────────────────
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


# ── Data Loading ─────────────────────────────────────────────────────────────
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


# ── Model ─────────────────────────────────────────────────────────────────────
def get_model():
    """ResNet-18 modified for CIFAR-10 with a classification head."""
    model = torchvision.models.resnet18(weights=None)

    # CIFAR-10 modification: replace 7x7 conv with 3x3, remove maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Replace final FC with 10-class head
    model.fc = nn.Linear(512, 10)
    return model.to(DEVICE)


# ── Training ──────────────────────────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────
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

    # ── Loss Curve ────────────────────────────────────────────────────────────
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

    # ── Final Test Evaluation ─────────────────────────────────────────────────
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

    # Save result for metrics.json
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

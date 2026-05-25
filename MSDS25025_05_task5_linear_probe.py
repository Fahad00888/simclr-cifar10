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
DATA_ROOT = "./data"
SPLITS_DIR    = "./splits"
RESULTS_DIR   = "./results"
GRAPHS_DIR    = "./graphs"
MODELS_DIR    = "./models"

BATCH_SIZE    = 64
EPOCHS        = 20
LR            = 3e-4
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIMCLR_ENCODER_PATH = "./models/simclr_encoder.pt"

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

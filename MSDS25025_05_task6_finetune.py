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
DATA_ROOT = "./data"
SPLITS_DIR    = "./splits"
RESULTS_DIR   = "./results"
GRAPHS_DIR    = "./graphs"
MODELS_DIR    = "./models"

BATCH_SIZE    = 64
EPOCHS        = 20
LR            = 3e-4
VIZ_SAMPLES   = 1000
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIMCLR_ENCODER_PATH = "./models/simclr_encoder.pt"

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

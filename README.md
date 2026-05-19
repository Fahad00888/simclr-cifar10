# DL Assignment 5 — SimCLR: From Supervised to Self-Supervised Learning

**Student:** Fahad Khalid  
**Roll Number:** MSDS25025  
**Course:** Deep Learning — Spring 2026  
**Institution:** Information Technology University, Lahore  

---

## Overview

Implementation of SimCLR (A Simple Framework for Contrastive Learning of Visual Representations) on CIFAR-10.  
Paper: https://arxiv.org/abs/2002.05709

---

## Project Structure

```
DL_Assignment5_SimCLR_MSDS25025/
├── splits/                          # Fixed dataset split files (provided by instructor)
│   ├── train_ssl_unlabeled.txt      # 45,000 indices — SimCLR pretraining
│   ├── train_labeled_10percent.txt  # 5,000 indices  — supervised / linear probe / fine-tuning
│   ├── val.txt                      # 5,000 indices  — model selection
│   ├── test.txt                     # 10,000 indices — final evaluation only
│   └── split_metadata.json
├── utils/                           # Provided helper utilities
│   ├── seed.py
│   ├── dataset_splits.py
│   ├── metrics.py
│   └── visualization.py
├── models/                          # Saved model weights (uploaded to Google Drive)
│   ├── supervised_best.pt
│   ├── simclr_encoder.pt
│   ├── linear_probe.pt
│   └── finetuned_model.pt
├── results/                         # Output images and metrics
│   ├── augmentation_examples.png
│   ├── similarity_matrix_before_training.png
│   ├── similarity_matrix_after_training.png
│   ├── supervised_confusion_matrix.png
│   ├── random_encoder_pca_or_tsne.png
│   ├── simclr_encoder_pca_or_tsne.png
│   ├── finetuned_encoder_pca_or_tsne.png
│   ├── metrics.json
│   └── test_predictions.csv
├── graphs/                          # Training curves
│   ├── supervised_loss.png
│   ├── simclr_pretraining_loss.png
│   ├── linear_probe_accuracy.png
│   └── finetuning_accuracy.png
├── MSDS25025_05_task1_supervised.py
├── MSDS25025_05_task2_augmentations.py
├── MSDS25025_05_task3_similarity.py
├── MSDS25025_05_task4_simclr.py
├── MSDS25025_05_task5_linear_probe.py
├── MSDS25025_05_task6_finetune.py
├── MSDS25025_05_allCode.py
├── requirements.txt
└── Report.pdf
```

---

## Setup

### On Kaggle (recommended — free GPU)

1. Create a new Kaggle notebook
2. Upload this repo or clone it
3. Add CIFAR-10 dataset from Kaggle datasets
4. Change `DATA_ROOT` in each script to `/kaggle/input/cifar-10-python`
5. Run scripts in order

### Local

```bash
pip install -r requirements.txt
```

Place CIFAR-10 data in `./data/` directory (torchvision format).

---

## Running Order

```bash
python MSDS25025_05_task1_supervised.py      # Supervised baseline
python MSDS25025_05_task2_augmentations.py   # Augmentation visualization
python MSDS25025_05_task3_similarity.py      # Feature similarity before training
python MSDS25025_05_task4_simclr.py          # SimCLR implementation + pretraining
python MSDS25025_05_task5_linear_probe.py    # Linear probe evaluation
python MSDS25025_05_task6_finetune.py        # Fine-tuning + final metrics
```

---

## Fixed Training Settings

| Setting | Value |
|---------|-------|
| Dataset | CIFAR-10 |
| Encoder | ResNet-18 (modified for CIFAR-10) |
| Image size | 32×32 |
| Batch size | 64 |
| SimCLR epochs | 50 |
| Linear probe epochs | 20 |
| Fine-tuning epochs | 20 |
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Temperature τ | 0.5 |
| Projection dim | 128 |
| Random seed | 2026 |

---

## Model Weights

Model weights are too large for GitHub. Download from Google Drive:  
🔗 **[Google Drive Link — add before submission]**

---

## Academic Integrity

- SimCLR-specific components (NT-Xent loss, similarity matrix, contrastive training loop, linear probing, fine-tuning) are implemented from scratch.
- No SimCLR libraries (lightly, solo-learn, VISSL, etc.) were used.
- Claude (claude.ai) was used for conceptual help and code assistance. Conversation transcript included in submission.
- Paper read: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations," ICML 2020.

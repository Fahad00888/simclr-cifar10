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
├── results/                         # Output images and metrics
├── graphs/                          # Training curves
├── MSDS25025_05_task1_supervised.py
├── MSDS25025_05_task2_augmentations.py
├── MSDS25025_05_task3_similarity.py
├── MSDS25025_05_task4_simclr.py
├── MSDS25025_05_task5_linear_probe.py
├── MSDS25025_05_task6_finetune.py
├── MSDS25025_05_allCode.py
├── requirements.txt
└── Report.pdf

---

## Kaggle Setup

1. Go to kaggle.com and create a New Notebook
2. Enable GPU T4 in Session options (right panel)
3. Upload data zip as Kaggle dataset (Input → Add Input → Upload)
4. In first notebook cell run:

```python
import zipfile, os, subprocess

# Copy code to working directory
subprocess.run(['cp', '-r',
    '/kaggle/input/datasets/fahadkhalid08/simclr-assignment5-code/DL_Assignment5_SimCLR_MSDS25025',
    '/kaggle/working/'])

os.chdir('/kaggle/working/DL_Assignment5_SimCLR_MSDS25025')

# Create output folders
for folder in ['graphs', 'results', 'models']:
    os.makedirs(folder, exist_ok=True)
```

5. DATA_ROOT in all scripts is set to:
   /kaggle/input/datasets/fahadkhalid08/cifar10-assignment5/data

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
| Image size | 32x32 |
| Batch size | 64 |
| SimCLR epochs | 50 |
| Linear probe epochs | 20 |
| Fine-tuning epochs | 20 |
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Temperature tau | 0.5 |
| Projection dim | 128 |
| Random seed | 2026 |

---

## Results

| Model | Test Accuracy |
|-------|--------------|
| Supervised ResNet-18 (10% labels) | 72.28% |
| Random encoder + linear probe | ___ |
| SimCLR encoder + linear probe | ___ |
| SimCLR encoder + fine-tuning | ___ |

---

## Model Weights

Model weights are too large for GitHub. Download from Google Drive:
Link: [add before submission]

---

## Academic Integrity

- SimCLR components implemented from scratch
- No SimCLR libraries used
- Claude (claude.ai) used for help — transcript included in submission
- Paper: Chen et al., ICML 2020

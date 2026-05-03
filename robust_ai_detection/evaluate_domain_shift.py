"""
evaluate_domain_shift.py
Tests CIFAKE-trained model on Stable Diffusion FACE images.
Same generator (Stable Diffusion), different domain (objects → faces).
Clean isolated domain shift.

Compare result against StyleGAN faces (AUC 0.489) to separate
domain shift contribution from generator shift contribution.
"""

import os, sys, torch
import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, confusion_matrix)
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, CHECKPOINT_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS
from models.resnet_detector import ResNetDetector

# ── Config ────────────────────────────────────────────────────────────────────
SD_FACES_DIR    = "/media/thinkpad/WINSETUP/zip/data/sd_faces/test"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_resnet18_baseline.pt")
NUM_SAMPLES     = 499   # per class — consistent with other experiments

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def load_model():
    model = ResNetDetector()
    ckpt  = torch.load(CHECKPOINT_PATH, map_location=DEVICE,
                       weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(DEVICE).eval()
    print(f"Model loaded from: {CHECKPOINT_PATH}")
    return model

def fix_label_mapping(dataset):
    """
    ImageFolder assigns labels alphabetically.
    We need to ensure fake=1, real=0 to match CIFAKE convention.
    Prints mapping so it can be verified before running inference.
    """
    print(f"Raw ImageFolder mapping: {dataset.class_to_idx}")

    # If alphabetical order gives fake=0, real=1 — flip it
    if dataset.class_to_idx.get('fake') == 0:
        dataset.class_to_idx = {'real': 0, 'fake': 1}
        dataset.targets       = [1 if t == 0 else 0
                                  for t in dataset.targets]
        print("Label mapping corrected: fake=1, real=0 ✓")
    else:
        print("Label mapping already correct: fake=1, real=0 ✓")

    n_fake = dataset.targets.count(1)
    n_real = dataset.targets.count(0)
    print(f"After remap — real={n_real}, fake={n_fake}")
    return dataset

def subsample(dataset, num_per_class):
    """Randomly subsample to num_per_class images per class."""
    import random

    # Handle both Dataset and Subset
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    else:
        targets = [dataset.dataset.targets[i] for i in dataset.indices]

    class_0 = [i for i, t in enumerate(targets) if t == 0]
    class_1 = [i for i, t in enumerate(targets) if t == 1]

    class_0 = random.sample(class_0, min(num_per_class, len(class_0)))
    class_1 = random.sample(class_1, min(num_per_class, len(class_1)))

    indices = class_0 + class_1
    print(f"Subsampled to {len(class_0)} real + {len(class_1)} fake "
          f"= {len(indices)} total")
    return Subset(dataset, indices)

def evaluate(model, loader):
    """Run inference and collect labels, predictions, and FAKE probabilities."""
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs   = imgs.to(DEVICE)
            logits = model(imgs)

            # Index 1 = FAKE probability (matches CIFAKE convention)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

            print(f"  Batch {i+1}/{len(loader)} done")

    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("DOMAIN SHIFT EVALUATION (ISOLATED)")
    print("Model trained on: CIFAKE (Stable Diffusion objects)")
    print("Tested on:        Stable Diffusion face images")
    print("Shift type:       Domain only (same generator)")
    print("="*60)

    model     = load_model()
    transform = get_transform()

    # Load dataset
    dataset = datasets.ImageFolder(SD_FACES_DIR, transform=transform)
    print(f"Total images before subsample: {len(dataset)}")

    # Fix label mapping
    dataset = fix_label_mapping(dataset)

    # Subsample to balanced NUM_SAMPLES per class
    dataset = subsample(dataset, NUM_SAMPLES)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)

    print(f"\nRunning inference...")
    labels, preds, probs = evaluate(model, loader)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc  = accuracy_score(labels, preds)
    auc  = roc_auc_score(labels, probs)
    f1   = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    cm   = confusion_matrix(labels, preds)

    baseline_auc        = 0.9771
    styleGAN_faces_auc  = 0.4890
    auc_drop            = baseline_auc - auc

    print("\n" + "="*60)
    print("DOMAIN SHIFT RESULTS (ISOLATED)")
    print("="*60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {cm}")
    print(f"\n--- Shift Decomposition Table ---")
    print(f"In-domain (CIFAKE, SD objects):       AUC {baseline_auc:.4f}")
    print(f"Domain shift only (SD faces):         AUC {auc:.4f}  "
          f"(drop={auc_drop:.4f})")
    print(f"Combined shift (StyleGAN faces):      AUC {styleGAN_faces_auc:.4f}  "
          f"(drop={baseline_auc - styleGAN_faces_auc:.4f})")
    print(f"\nDomain-only contribution:            {auc_drop:.4f}")
    print(f"Generator contribution (approx):     "
          f"{auc - styleGAN_faces_auc:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()

"""
evaluate_ensemble.py

Experiment 5: Ensemble of ResNet18 + CLIP ViT-L/14.
Averages softmax probabilities from both models across all shift conditions.

Hypothesis: Since ResNet18 is resize-fragile but compression-robust,
and CLIP is the opposite, their ensemble should be robust to both.

Usage:
    python evaluate_ensemble.py
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEVICE, CHECKPOINT_DIR, BATCH_SIZE, NUM_SAMPLES,
    JPEG_QUALITY_LEVELS, RESIZE_FACTORS
)
from data.dataset import get_dataloader

RESNET_CKPT = os.path.join(CHECKPOINT_DIR, "best_resnet18_baseline.pt")
CLIP_CKPT   = os.path.join(CHECKPOINT_DIR, "best_clip_mlp.pt")


def load_models():
    """Load both models from checkpoints."""
    # ResNet18
    from models.resnet_detector import ResNetDetector
    resnet = ResNetDetector()
    ckpt = torch.load(RESNET_CKPT, map_location=DEVICE, weights_only=False)
    resnet.load_state_dict(ckpt["state_dict"])
    resnet = resnet.to(DEVICE)
    resnet.eval()
    print(f"ResNet18 loaded  — epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")

    # CLIP MLP
    from models.clip_mlp import CLIPMLP
    clip_model = CLIPMLP()
    ckpt = torch.load(CLIP_CKPT, map_location=DEVICE, weights_only=False)
    clip_model.load_state_dict(ckpt["state_dict"])
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()
    print(f"CLIP MLP loaded  — epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")

    return resnet, clip_model


def evaluate_ensemble(resnet, clip_model, loader):
    """
    Run both models on loader, average softmax probabilities, return AUC + acc.
    """
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            # ResNet18 probabilities
            probs_resnet = torch.softmax(resnet(images), dim=1)[:, 1]

            # CLIP probabilities
            probs_clip = torch.softmax(clip_model(images), dim=1)[:, 1]

            # Average (equal weight ensemble)
            probs_ensemble = (probs_resnet + probs_clip) / 2.0

            all_labels.extend(labels.numpy())
            all_probs.extend(probs_ensemble.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    auc  = roc_auc_score(all_labels, all_probs)
    preds = (all_probs >= 0.5).astype(int)
    acc  = accuracy_score(all_labels, preds)

    return {"auc": auc, "accuracy": acc}


def main():
    print("\n" + "="*60)
    print("Experiment 5: Ensemble Evaluation (ResNet18 + CLIP)")
    print("="*60)

    resnet, clip_model = load_models()

    results = {}

    # ── Baseline ─────────────────────────────────────────────────────────────
    print("\nBaseline (no shift)...")
    loader = get_dataloader("test", num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
    results["Baseline"] = evaluate_ensemble(resnet, clip_model, loader)

    # ── JPEG Compression Shift ────────────────────────────────────────────────
    print("\nJPEG compression shift...")
    for quality in JPEG_QUALITY_LEVELS:
        print(f"  JPEG q={quality}...")
        loader = get_dataloader("test", jpeg_quality=quality,
                                num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
        results[f"JPEG q={quality}"] = evaluate_ensemble(resnet, clip_model, loader)

    # ── Resize Shift ──────────────────────────────────────────────────────────
    print("\nResize shift...")
    for factor in RESIZE_FACTORS:
        if factor == 1.0:
            continue
        print(f"  Resize x{factor}...")
        loader = get_dataloader("test", resize_factor=factor,
                                num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
        results[f"Resize x{factor}"] = evaluate_ensemble(resnet, clip_model, loader)

    # ── Results Table ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS SUMMARY")
    print("="*60)
    print(f"{'Condition':<20} {'AUC':>8} {'Accuracy':>10}")
    print("-" * 42)
    for name, m in results.items():
        print(f"{name:<20} {m['auc']:>8.4f} {m['accuracy']:>10.4f}")

    # ── Key finding ───────────────────────────────────────────────────────────
    baseline_auc = results["Baseline"]["auc"]
    print("\n" + "="*60)
    print("AUC DROPS FROM BASELINE")
    print("="*60)
    for name, m in results.items():
        if name == "Baseline":
            continue
        drop = baseline_auc - m["auc"]
        print(f"{name:<20} drop={drop:+.4f}")


if __name__ == "__main__":
    main()

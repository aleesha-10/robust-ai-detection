"""
utils/metrics.py

Evaluation metrics for binary classification.
Computes: Accuracy, Precision, Recall, F1, AUC, Confusion Matrix.

All metrics are logged to CSV so you have a full experiment record
for your paper's results section.
"""

import os
import sys
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_all_metrics(y_true: list, y_pred: list, y_prob: list) -> dict:
    """
    Compute the full metrics suite for a binary classification result.

    Args:
        y_true: Ground truth labels [0=real, 1=fake]
        y_pred: Predicted labels
        y_prob: Predicted probability of class 1 (fake), used for AUC

    Returns:
        dict with keys: accuracy, precision, recall, f1, auc
        and confusion matrix components: tp, fp, tn, fn
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    # AUC requires probabilities
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")   # Only one class present

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy":  round(acc, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "auc":       round(float(auc), 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_model(model, dataloader, device: str = "cpu") -> dict:
    """
    Run inference on a dataloader and return all metrics.

    Args:
        model:      PyTorch model (already trained)
        dataloader: DataLoader with (image, label) batches
        device:     "cpu" or "cuda"

    Returns:
        metrics dict (same format as compute_all_metrics)
    """
    model.eval()
    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())   # P(fake)

    return compute_all_metrics(all_labels, all_preds, all_probs)


def performance_drop(baseline_metrics: dict, shifted_metrics: dict,
                     metric: str = "auc") -> float:
    """
    Compute performance drop between in-distribution and shifted testing.

    This is the core robustness metric from the proposal:
        Performance Drop = Accuracy (original) – Accuracy (shifted)

    Args:
        baseline_metrics: Metrics on the original (unshifted) test set
        shifted_metrics:  Metrics on the shifted test set
        metric:           Which metric to compare ("accuracy", "auc", "f1")

    Returns:
        drop: Positive value means performance decreased under shift
    """
    base = baseline_metrics[metric]
    shifted = shifted_metrics[metric]
    drop = base - shifted
    return round(drop, 4)


def format_metrics_table(results_dict: dict) -> str:
    """
    Format a results dictionary as a neat ASCII table.
    For copying into your paper's results section.

    Args:
        results_dict: {experiment_name: metrics_dict}

    Returns:
        Formatted table string
    """
    header = f"{'Experiment':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}"
    sep    = "-" * len(header)
    rows   = [header, sep]

    for name, m in results_dict.items():
        row = (
            f"{name:<30} "
            f"{m['accuracy']:>6.4f} "
            f"{m['precision']:>6.4f} "
            f"{m['recall']:>6.4f} "
            f"{m['f1']:>6.4f} "
            f"{m['auc']:>6.4f}"
        )
        rows.append(row)

    return "\n".join(rows)


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate perfect predictions
    y_true = [0, 0, 1, 1, 0, 1, 1, 0]
    y_pred = [0, 0, 1, 1, 0, 1, 0, 0]   # one error
    y_prob = [0.1, 0.2, 0.9, 0.8, 0.3, 0.85, 0.45, 0.15]

    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    print("Metrics:", metrics)

    results = {
        "Baseline (no shift)": {"accuracy": 0.95, "precision": 0.94,
                                 "recall": 0.96, "f1": 0.95, "auc": 0.97},
        "JPEG q=50":           {"accuracy": 0.88, "precision": 0.87,
                                 "recall": 0.89, "f1": 0.88, "auc": 0.91},
        "JPEG q=10":           {"accuracy": 0.72, "precision": 0.71,
                                 "recall": 0.73, "f1": 0.72, "auc": 0.78},
    }
    print("\n" + format_metrics_table(results))

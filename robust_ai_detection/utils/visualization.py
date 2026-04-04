"""
utils/visualization.py

All visualization for the paper:
  1. Training curves (loss and accuracy over epochs)
  2. AUC/Accuracy vs. distribution shift (compression, resize)
  3. Confusion matrix heatmap
  4. Saliency maps (gradient-based: Vanilla Gradient)
  5. Softmax confidence distributions

These are the figures that belong in your paper's Results section.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PLOTS_DIR, JPEG_QUALITY_LEVELS, RESIZE_FACTORS, IMAGE_SIZE


def save_fig(fig, filename: str):
    """Save figure to plots directory."""
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)
    return path


# ─── 1. Training Curves ──────────────────────────────────────────────────────
def plot_training_curves(training_log: list, filename: str = "training_curves.png"):
    """
    Plot loss and accuracy over training epochs.
    training_log: list of dicts with keys train_loss, val_loss, train_acc, val_acc
    """
    epochs      = [r["epoch"] for r in training_log]
    train_loss  = [r["train_loss"] for r in training_log]
    val_loss    = [r["val_loss"] for r in training_log]
    train_acc   = [r["train_acc"] for r in training_log]
    val_acc     = [r["val_acc"] for r in training_log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss", color="steelblue", linewidth=2)
    ax1.plot(epochs, val_loss,   label="Val Loss",   color="tomato",    linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_acc, label="Train Acc", color="steelblue", linewidth=2)
    ax2.plot(epochs, val_acc,   label="Val Acc",   color="tomato",    linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return save_fig(fig, filename)


# ─── 2. Distribution Shift Results ──────────────────────────────────────────
def plot_compression_robustness(
    jpeg_quality_levels: list,
    results_by_quality: dict,
    filename: str = "compression_robustness.png"
):
    """
    Plot AUC and Accuracy vs. JPEG quality level.
    Shows the performance drop as compression increases.

    results_by_quality: {quality: metrics_dict}
    """
    qualities = sorted(jpeg_quality_levels, reverse=True)
    auc_vals  = [results_by_quality[q]["auc"]      for q in qualities]
    acc_vals  = [results_by_quality[q]["accuracy"]  for q in qualities]
    f1_vals   = [results_by_quality[q]["f1"]        for q in qualities]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qualities, auc_vals, "o-", label="AUC",      color="steelblue",  linewidth=2, markersize=7)
    ax.plot(qualities, acc_vals, "s-", label="Accuracy", color="darkorange", linewidth=2, markersize=7)
    ax.plot(qualities, f1_vals,  "^-", label="F1",       color="green",      linewidth=2, markersize=7)

    ax.set_xlabel("JPEG Quality (%)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Detection Performance under JPEG Compression Shift", fontsize=13)
    ax.set_ylim(0.4, 1.05)
    ax.invert_xaxis()   # High quality on left, heavy compression on right
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate drop
    drop = auc_vals[0] - auc_vals[-1]
    ax.annotate(
        f"AUC drop: {drop:.3f}",
        xy=(qualities[-1], auc_vals[-1]),
        xytext=(qualities[-1] + 5, auc_vals[-1] - 0.05),
        fontsize=9, color="steelblue",
        arrowprops=dict(arrowstyle="->", color="steelblue")
    )

    plt.tight_layout()
    return save_fig(fig, filename)


def plot_robustness_comparison(
    experiments: dict,
    baseline_key: str = "Baseline",
    metric: str = "auc",
    filename: str = "robustness_comparison.png"
):
    """
    Bar chart comparing performance across all distribution shifts.
    Clearly shows which shift causes the most degradation.

    experiments: {name: metrics_dict}
    """
    names  = list(experiments.keys())
    values = [experiments[n][metric] for n in names]
    colors = ["steelblue" if n == baseline_key else "tomato" for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"{metric.upper()} Across Distribution Shift Experiments", fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=values[0] if names[0] == baseline_key else max(values),
               color="grey", linestyle="--", alpha=0.5, label="Baseline")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    return save_fig(fig, filename)


# ─── 3. Confusion Matrix ─────────────────────────────────────────────────────
def plot_confusion_matrix(
    cm: list,
    title: str = "Confusion Matrix",
    filename: str = "confusion_matrix.png"
):
    """
    Heatmap of confusion matrix.
    cm: [[TN, FP], [FN, TP]]
    """
    cm_array = np.array(cm)
    labels = ["Real (0)", "Fake (1)"]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_array, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, cbar=True
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    return save_fig(fig, filename)


# ─── 4. Saliency Maps ────────────────────────────────────────────────────────
def compute_vanilla_saliency(model, image_tensor: torch.Tensor,
                              target_class: int = 1) -> np.ndarray:
    """
    Vanilla Gradient saliency map.

    Computes the gradient of the target class score with respect to the input
    image. High-gradient pixels are the ones the model "looks at" most.

    This is the standard first-order attribution method used in
    explainability literature (Simonyan et al., 2014).

    Args:
        model:        Trained PyTorch model
        image_tensor: [1, 3, H, W] — single image, normalized
        target_class: Class to explain (1 = fake)

    Returns:
        saliency: [H, W] — 2D saliency map (absolute gradient magnitude)
    """
    model.eval()

    # Enable gradient computation for input
    x = image_tensor.clone().requires_grad_(True)

    # Forward pass
    logits = model(x)
    score  = logits[0, target_class]

    # Backward pass
    model.zero_grad()
    score.backward()

    # Take max across color channels, then normalize
    saliency = x.grad.data.abs()            # [1, 3, H, W]
    saliency = saliency.max(dim=1)[0]       # [1, H, W] — max over channels
    saliency = saliency.squeeze().numpy()   # [H, W]

    # Normalize to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


def plot_saliency_maps(
    model,
    sample_images: list,          # list of (image_tensor, pil_image, label, pred)
    filename: str = "saliency_maps.png",
    n_cols: int = 4,
):
    """
    Plot saliency maps for a batch of sample images.
    Each image shown alongside its saliency overlay.

    sample_images: list of (tensor [1,3,H,W], PIL Image, true_label, pred_label)
    """
    n = len(sample_images)
    n_rows = (n + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3))
    fig.suptitle("Saliency Maps (Vanilla Gradient)\nHighlighted regions = model's focus",
                 fontsize=12)

    for i, (tensor, pil_img, true_label, pred_label) in enumerate(sample_images):
        # Compute saliency
        saliency = compute_vanilla_saliency(model, tensor, target_class=pred_label)

        # Convert PIL to numpy for display
        img_np = np.array(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)))

        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        # Overlay saliency as heatmap on image
        ax.imshow(img_np)
        ax.imshow(saliency, alpha=0.5, cmap="hot")

        true_str = "Real" if true_label == 0 else "Fake"
        pred_str = "Real" if pred_label == 0 else "Fake"
        color    = "green" if true_label == pred_label else "red"

        ax.set_title(
            f"True: {true_str}\nPred: {pred_str}",
            fontsize=9, color=color
        )
        ax.axis("off")

    plt.tight_layout()
    return save_fig(fig, filename)


# ─── 5. Confidence Distribution ─────────────────────────────────────────────
def plot_confidence_distributions(
    confidences_by_experiment: dict,
    filename: str = "confidence_distributions.png"
):
    """
    Plot softmax confidence distributions for each experiment.
    Shows how model confidence changes under distribution shifts.

    confidences_by_experiment: {name: {"correct": [...], "incorrect": [...]}}
    """
    n = len(confidences_by_experiment)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, conf) in zip(axes, confidences_by_experiment.items()):
        correct   = conf.get("correct", [])
        incorrect = conf.get("incorrect", [])

        if correct:
            ax.hist(correct,   bins=20, alpha=0.6, color="steelblue",
                    label="Correct",   density=True)
        if incorrect:
            ax.hist(incorrect, bins=20, alpha=0.6, color="tomato",
                    label="Incorrect", density=True)

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("P(Fake)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Softmax Confidence Distributions Under Distribution Shifts",
                 fontsize=12)
    plt.tight_layout()
    return save_fig(fig, filename)


# ─── 6. Sample Grid ──────────────────────────────────────────────────────────
def plot_sample_grid(real_images: list, fake_images: list,
                     n: int = 4, filename: str = "sample_images.png"):
    """Show a grid of real vs fake images from the dataset."""
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5))
    fig.suptitle("CIFAKE: Real vs AI-Generated Images", fontsize=12)

    for i, img in enumerate(real_images[:n]):
        axes[0, i].imshow(np.array(img))
        axes[0, i].set_title("REAL", fontsize=8, color="green")
        axes[0, i].axis("off")

    for i, img in enumerate(fake_images[:n]):
        axes[1, i].imshow(np.array(img))
        axes[1, i].set_title("FAKE", fontsize=8, color="red")
        axes[1, i].axis("off")

    plt.tight_layout()
    return save_fig(fig, filename)

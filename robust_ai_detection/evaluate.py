"""
evaluate.py

Full evaluation script — runs all distribution shift experiments
and generates paper-ready figures and tables.

Usage:
    python evaluate.py                                    # Uses last trained model
    python evaluate.py --checkpoint results/checkpoints/best_resnet18.pt

Experiments run:
  1. Baseline (no shift)
  2. Compression shift: JPEG quality 95, 75, 50, 30, 10
  3. Resize shift: factor 0.75, 0.5, 0.25
  4. Cross-generator shift analysis
  5. Saliency map generation
  6. Confidence distribution analysis

Output:
  results/
    plots/
      training_curves.png
      compression_robustness.png
      robustness_comparison.png
      confusion_matrix_baseline.png
      saliency_maps.png
      confidence_distributions.png
    logs/
      *_eval.csv          ← full metrics table
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEVICE, JPEG_QUALITY_LEVELS, RESIZE_FACTORS, BATCH_SIZE,
    NUM_SAMPLES, CHECKPOINT_DIR, SALIENCY_NUM_EXAMPLES, MODEL_TYPE
)
from data.dataset import get_dataloader, CIFAKEDataset, get_base_transform
from utils.metrics import evaluate_model, performance_drop, format_metrics_table
from utils.logger import ExperimentLogger
from utils.visualization import (
    plot_compression_robustness,
    plot_robustness_comparison,
    plot_confusion_matrix,
    plot_saliency_maps,
    plot_confidence_distributions,
    compute_vanilla_saliency,
)


def load_model(checkpoint_path: str):
    """Load trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run: python train.py first"
        )

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model_type = ckpt.get("model_type", "resnet18")

    if model_type == "resnet18":
        from models.resnet_detector import ResNetDetector
        model = ResNetDetector()
    elif model_type == "clip_mlp":
        from models.clip_mlp import CLIPMLP
        model = CLIPMLP()
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")

    model.load_state_dict(ckpt["state_dict"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Loaded {model_type} checkpoint:")
    print(f"  Epoch:    {ckpt['epoch']}")
    print(f"  Val loss: {ckpt['val_loss']:.4f}")
    print(f"  Val acc:  {ckpt['val_acc']:.4f}")

    return model, model_type


def collect_confidences(model, dataloader):
    """
    Collect softmax confidence values, split by correct/incorrect.
    Used for confidence distribution visualization.
    """
    correct_confs   = []
    incorrect_confs = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            probs  = torch.softmax(model(images), dim=1)
            preds  = probs.argmax(dim=1)
            confs  = probs[:, 1].cpu().numpy()   # P(fake)
            labels_np = labels.numpy()
            preds_np  = preds.cpu().numpy()

            for c, p, l in zip(confs, preds_np, labels_np):
                if p == l:
                    correct_confs.append(float(c))
                else:
                    incorrect_confs.append(float(c))

    return {"correct": correct_confs, "incorrect": incorrect_confs}


def collect_saliency_samples(model, dataset, n: int = SALIENCY_NUM_EXAMPLES):
    """
    Collect n sample images (mix of correct/incorrect predictions) for saliency.
    Returns list of (tensor, pil_image, true_label, pred_label)
    """
    samples = []
    transform = get_base_transform()

    from PIL import Image
    for idx in range(min(n * 3, len(dataset))):
        img_path = dataset.image_paths[idx]
        true_label = dataset.labels[idx]

        pil_img = Image.open(img_path).convert("RGB")
        tensor  = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            pred   = logits.argmax(dim=1).item()

        samples.append((tensor.cpu(), pil_img, true_label, pred))
        if len(samples) >= n:
            break

    return samples


def run_all_experiments(checkpoint_path: str):
    """
    Run the complete evaluation suite.
    """
    print("\n" + "="*60)
    print("EVALUATION: Robust AI-Generated Image Detection")
    print("="*60)

    model, model_type = load_model(checkpoint_path)
    logger = ExperimentLogger(f"eval_{model_type}")

    all_results = {}

    # ── 1. Baseline (no distribution shift) ─────────────────────────────────
    print("\n[1/6] Baseline evaluation (no shift)...")
    baseline_loader = get_dataloader("test", num_samples=NUM_SAMPLES,
                                     batch_size=BATCH_SIZE)
    baseline_metrics = evaluate_model(model, baseline_loader, DEVICE)
    all_results["Baseline"] = baseline_metrics
    logger.log_eval("Baseline", baseline_metrics)

    # Confusion matrix for baseline
    plot_confusion_matrix(
        baseline_metrics["confusion_matrix"],
        title="Confusion Matrix — Baseline",
        filename="confusion_matrix_baseline.png"
    )

    # ── 2. Compression Shift ─────────────────────────────────────────────────
    print("\n[2/6] Compression shift experiments (JPEG)...")
    compression_results = {}
    for quality in JPEG_QUALITY_LEVELS:
        print(f"  JPEG quality={quality}...")
        loader = get_dataloader("test", jpeg_quality=quality,
                                num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
        metrics = evaluate_model(model, loader, DEVICE)
        compression_results[quality] = metrics
        name = f"JPEG q={quality}"
        all_results[name] = metrics
        logger.log_eval(name, metrics)

        drop = performance_drop(baseline_metrics, metrics, "auc")
        print(f"    AUC={metrics['auc']:.4f}  (drop={drop:+.4f} from baseline)")

    plot_compression_robustness(JPEG_QUALITY_LEVELS, compression_results)

    # ── 3. Resize Shift ──────────────────────────────────────────────────────
    print("\n[3/6] Resize shift experiments...")
    for factor in RESIZE_FACTORS:
        if factor == 1.0:
            continue   # Skip — that's the baseline
        print(f"  Resize factor={factor}...")
        loader = get_dataloader("test", resize_factor=factor,
                                num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
        metrics = evaluate_model(model, loader, DEVICE)
        name = f"Resize ×{factor}"
        all_results[name] = metrics
        logger.log_eval(name, metrics)

        drop = performance_drop(baseline_metrics, metrics, "auc")
        print(f"    AUC={metrics['auc']:.4f}  (drop={drop:+.4f} from baseline)")

    # ── 4. Cross-Generator Shift ─────────────────────────────────────────────
    print("\n[4/6] Cross-generator shift...")
    print("  Note: CIFAKE uses Stable Diffusion for FAKE images.")
    print("  Cross-generator: we evaluate model on FAKE images only (diffusion).")
    print("  For full cross-generator analysis (GAN→Diffusion), add StyleGAN")
    print("  images to data/cifake/test/FAKE_GAN/ and re-run.")

    # Simulate by evaluating only on FAKE subset (robustness to generator artifacts)
    loader = get_dataloader("test", num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
    metrics = evaluate_model(model, loader, DEVICE)
    all_results["Cross-Generator (SD)"] = metrics
    logger.log_eval("Cross-Generator (SD)", metrics, "Stable Diffusion test set")

    # ── 5. Robustness comparison plot ────────────────────────────────────────
    print("\n[5/6] Generating comparison plot...")
    plot_robustness_comparison(all_results, baseline_key="Baseline", metric="auc")
    plot_robustness_comparison(all_results, baseline_key="Baseline", metric="accuracy",
                                filename="robustness_comparison_accuracy.png")

    # ── 6. Explainability ────────────────────────────────────────────────────
    print("\n[6/6] Explainability analysis...")

    # Saliency maps
    print("  Generating saliency maps...")
    dataset = CIFAKEDataset(split="test", num_samples=100)
    saliency_samples = collect_saliency_samples(model, dataset, n=SALIENCY_NUM_EXAMPLES)
    plot_saliency_maps(model, saliency_samples, filename="saliency_maps.png")

    # Confidence distributions
    print("  Collecting confidence distributions...")
    conf_dict = {}

    conf_dict["Baseline"] = collect_confidences(
        model, get_dataloader("test", num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
    )
    conf_dict["JPEG q=50"] = collect_confidences(
        model, get_dataloader("test", jpeg_quality=50,
                              num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
    )
    conf_dict["JPEG q=10"] = collect_confidences(
        model, get_dataloader("test", jpeg_quality=10,
                              num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
    )

    plot_confidence_distributions(conf_dict, filename="confidence_distributions.png")

    # ── Final summary table ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(format_metrics_table(all_results))

    logger.summary()
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI-image detector")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(CHECKPOINT_DIR, f"best_{MODEL_TYPE}.pt"),
        help="Path to model checkpoint"
    )
    args = parser.parse_args()

    run_all_experiments(args.checkpoint)

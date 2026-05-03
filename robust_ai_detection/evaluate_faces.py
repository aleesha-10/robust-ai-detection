"""
evaluate_faces.py

Experiment 4: Cross-domain evaluation on face images.
Tests CIFAKE-trained ResNet18 on StyleGAN faces — measures domain shift.

Usage:
    python evaluate_faces.py

No retraining. Uses existing best_resnet18_baseline.pt checkpoint.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, CHECKPOINT_DIR, BATCH_SIZE

# ─── Paths ────────────────────────────────────────────────────────────────────
FACES_TEST_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "faces", "real_vs_fake", "real-vs-fake", "test"
)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_resnet18_baseline.pt")

# ─── Dataset ─────────────────────────────────────────────────────────────────
class FacesDataset(Dataset):
    """
    Loads real/fake face images for cross-domain evaluation.
    Labels: real=0, fake=1  (matches CIFAKE convention)
    """
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        # lowercase folders: real/, fake/
        for label, cls_name in [(0, "real"), (1, "fake")]:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        print(f"Faces test set: {len(self.samples)} images "
              f"({sum(1 for _,l in self.samples if l==0)} real, "
              f"{sum(1 for _,l in self.samples if l==1)} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("Experiment 4: Cross-Domain Evaluation (Faces)")
    print("  Train domain: CIFAKE (CIFAR objects, Stable Diffusion)")
    print("  Test domain:  140k Faces (Flickr real, StyleGAN fake)")
    print("="*60)

    # Transform — 224px to match ResNet18 training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = FacesDataset(FACES_TEST_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    # Load ResNet18 checkpoint
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Make sure Experiment 1 (ResNet18 training) has completed."
        )

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    from models.resnet_detector import ResNetDetector
    model = ResNetDetector()
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Checkpoint: epoch={ckpt['epoch']}, "
          f"val_acc={ckpt['val_acc']:.4f} (on CIFAKE)")

    # Inference
    print("\nRunning inference on face images...")
    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)[:, 1]  # P(fake)
            preds  = logits.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # Metrics
    auc      = roc_auc_score(all_labels, all_probs)
    acc      = accuracy_score(all_labels, all_preds)
    cm       = confusion_matrix(all_labels, all_preds)
    report   = classification_report(all_labels, all_preds,
                                     target_names=["Real", "Fake"])

    print("\n" + "="*60)
    print("RESULTS — Cross-Domain (Faces)")
    print("="*60)
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {'':10s}  Pred Real  Pred Fake")
    print(f"  {'True Real':10s}  {cm[0][0]:9d}  {cm[0][1]:9d}")
    print(f"  {'True Fake':10s}  {cm[1][0]:9d}  {cm[1][1]:9d}")
    print(f"\nClassification Report:")
    print(report)

    # Interpretation
    print("="*60)
    print("INTERPRETATION")
    print("="*60)
    if auc < 0.6:
        print("  → Near-random performance (AUC < 0.6)")
        print("  → Model has NOT generalized across domains")
        print("  → Strong evidence of domain shift failure")
    elif auc < 0.75:
        print("  → Below-chance or weak generalization (AUC 0.6–0.75)")
        print("  → Partial domain shift failure")
    else:
        print("  → Surprisingly strong cross-domain generalization (AUC > 0.75)")

    print(f"\n  Note: Model was trained on CIFAR-scale object images (32px upscaled).")
    print(f"  Face images are StyleGAN-generated — different generator + different domain.")
    print(f"  Any drop from CIFAKE AUC is attributable to combined domain+generator shift.")


if __name__ == "__main__":
    main()

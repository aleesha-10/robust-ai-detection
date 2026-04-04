"""
data/dataset.py

PyTorch Dataset classes for CIFAKE with support for:
  - Standard loading (real vs. AI-generated)
  - JPEG compression shift
  - Resize shift
  - Cross-domain split (by CIFAR category)
  - Cross-generator split (real CIFAR vs Stable Diffusion)
"""

import os
import sys
import io
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, IMAGE_SIZE, NUM_SAMPLES, BATCH_SIZE, NUM_WORKERS,
    SEED, DEVICE
)


# ─── Label mapping ─────────────────────────────────────────────────────────
LABEL_MAP = {"REAL": 0, "FAKE": 1}


def get_base_transform(image_size=IMAGE_SIZE):
    """Standard preprocessing: resize + normalize."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_augment_transform(image_size=IMAGE_SIZE):
    """Training augmentation (mild — preserves forensic artifacts)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ─── JPEG compression utility ───────────────────────────────────────────────
def apply_jpeg_compression(pil_image: Image.Image, quality: int) -> Image.Image:
    """
    Simulate JPEG compression artifact — the most common real-world shift.
    Quality: 95 (near-lossless) → 10 (heavy compression).

    This mimics what happens when images are shared on social media:
    platforms like Twitter/Instagram automatically compress uploads.
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def apply_resize_shift(pil_image: Image.Image, factor: float,
                       original_size: int) -> Image.Image:
    """
    Downsample then upsample — simulates lossy resizing artifacts.
    factor=0.5 means: shrink to 50%, then scale back up.
    """
    small_size = max(4, int(original_size * factor))
    downsampled = pil_image.resize((small_size, small_size), Image.BILINEAR)
    return downsampled.resize((original_size, original_size), Image.BILINEAR)


# ─── Main Dataset class ──────────────────────────────────────────────────────
class CIFAKEDataset(Dataset):
    """
    CIFAKE dataset loader.

    Args:
        root_dir:       Path to cifake/ folder (contains train/, test/)
        split:          "train", "val", or "test"
        transform:      torchvision transform to apply
        jpeg_quality:   If set (int), apply JPEG compression before transform
        resize_factor:  If set (float < 1.0), apply resize shift
        num_samples:    Max samples per class (-1 = use all)
        seed:           Random seed for reproducibility
    """

    def __init__(
        self,
        root_dir: str = DATA_DIR,
        split: str = "train",
        transform=None,
        jpeg_quality: int = None,
        resize_factor: float = None,
        num_samples: int = NUM_SAMPLES,
        seed: int = SEED,
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or get_base_transform()
        self.jpeg_quality = jpeg_quality
        self.resize_factor = resize_factor

        # CIFAKE has train/ and test/ splits natively
        # We carve out val from train using the seed
        self.image_paths = []
        self.labels = []

        self._load_data(num_samples, seed)

    def _load_data(self, num_samples, seed):
        """Load image paths and labels, respecting split."""
        rng = random.Random(seed)

        # Determine source split folder
        # CIFAKE: train → 50k/class, test → 10k/class
        # We split CIFAKE train → our train + val
        if self.split in ("train", "val"):
            folder = os.path.join(self.root_dir, "train")
        else:
            folder = os.path.join(self.root_dir, "test")

        all_paths = []
        all_labels = []

        for cls_name, label in LABEL_MAP.items():
            cls_dir = os.path.join(folder, cls_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(
                    f"Expected directory not found: {cls_dir}\n"
                    f"Run: python data/download_data.py"
                )

            paths = sorted([
                os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

            # Limit samples per class
            if num_samples > 0:
                paths = paths[:num_samples]

            all_paths.extend(paths)
            all_labels.extend([label] * len(paths))

        # Shuffle together (seeded for reproducibility)
        combined = list(zip(all_paths, all_labels))
        rng.shuffle(combined)
        all_paths, all_labels = zip(*combined)

        # Split train → train (85%) + val (15%)
        if self.split in ("train", "val"):
            n = len(all_paths)
            val_start = int(n * 0.85)
            if self.split == "train":
                all_paths = all_paths[:val_start]
                all_labels = all_labels[:val_start]
            else:
                all_paths = all_paths[val_start:]
                all_labels = all_labels[val_start:]

        self.image_paths = list(all_paths)
        self.labels = list(all_labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply distribution shifts (before normalization)
        if self.resize_factor is not None and self.resize_factor < 1.0:
            image = apply_resize_shift(image, self.resize_factor, image.size[0])

        if self.jpeg_quality is not None:
            image = apply_jpeg_compression(image, self.jpeg_quality)

        # Apply transform (resize, normalize, to tensor)
        image = self.transform(image)

        return image, label

    def get_class_counts(self):
        """Returns {0: count_real, 1: count_fake}."""
        counts = {0: 0, 1: 0}
        for label in self.labels:
            counts[label] += 1
        return counts

    def get_sample_images(self, n: int = 4):
        """Return n sample images as PIL Images (for visualization)."""
        samples = {"REAL": [], "FAKE": []}
        for path, label in zip(self.image_paths, self.labels):
            cls = "REAL" if label == 0 else "FAKE"
            if len(samples[cls]) < n:
                img = Image.open(path).convert("RGB")
                if self.jpeg_quality:
                    img = apply_jpeg_compression(img, self.jpeg_quality)
                samples[cls].append(img)
            if all(len(v) >= n for v in samples.values()):
                break
        return samples


# ─── DataLoader factories ────────────────────────────────────────────────────
def get_dataloader(
    split: str,
    jpeg_quality: int = None,
    resize_factor: float = None,
    batch_size: int = BATCH_SIZE,
    num_samples: int = NUM_SAMPLES,
    augment: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for a given split with optional distribution shift.

    Args:
        split:          "train", "val", or "test"
        jpeg_quality:   Apply JPEG compression (e.g., 50 for medium compression)
        resize_factor:  Apply resize shift (e.g., 0.5)
        batch_size:     Samples per batch
        num_samples:    Max per class (-1 = all)
        augment:        Use data augmentation (True for training only)
    """
    transform = get_augment_transform() if (augment and split == "train") \
                else get_base_transform()

    dataset = CIFAKEDataset(
        split=split,
        transform=transform,
        jpeg_quality=jpeg_quality,
        resize_factor=resize_factor,
        num_samples=num_samples,
    )

    shuffle = (split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )


def get_all_dataloaders(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE):
    """
    Returns standard train/val/test loaders (no distribution shifts).
    Use this for initial training.
    """
    return {
        "train": get_dataloader("train", augment=True,
                                num_samples=num_samples, batch_size=batch_size),
        "val":   get_dataloader("val",
                                num_samples=num_samples, batch_size=batch_size),
        "test":  get_dataloader("test",
                                num_samples=num_samples, batch_size=batch_size),
    }


# ─── Quick verification ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing dataset loading...")

    ds = CIFAKEDataset(split="train", num_samples=100)
    print(f"Train samples: {len(ds)}")
    print(f"Class counts: {ds.get_class_counts()}")

    img, label = ds[0]
    print(f"Image tensor shape: {img.shape}")
    print(f"Label: {label} ({'REAL' if label == 0 else 'FAKE'})")

    # Test JPEG compression
    ds_jpeg = CIFAKEDataset(split="test", num_samples=100, jpeg_quality=30)
    img_j, _ = ds_jpeg[0]
    print(f"JPEG-compressed image shape: {img_j.shape}")

    print("\nDataset OK!")

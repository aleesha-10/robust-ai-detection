"""
models/clip_mlp.py

CLIP-based MLP detector — directly mirrors the base paper architecture.

Paper: "Robust CLIP-Based Detector for Exposing Diffusion Model-Generated Images"
       (Santosh et al., 2024, Purdue-M2/Robust_DM_Generated_Image_Detection)

Architecture:
  CLIP ViT-B/32 (frozen) → [image features: 512-dim]
       ↓ concatenate
  CLIP ViT-B/32 (frozen) → [text features: 512-dim]  (optional)
       ↓
  MLP: 512 → 256 → 128 → 2

NOTE: The original paper uses ViT-L/14 (1536-dim). We use ViT-B/32 (512-dim)
which runs comfortably on CPU. The architecture is otherwise identical.
To reproduce the paper exactly, change CLIP_MODEL in config.py to "ViT-L/14"
(requires more RAM and much slower on CPU — recommended on GPU).

CPU-friendly choices:
  - ViT-B/32: 512-dim, ~10s per batch, works on 16GB RAM ✓
  - ViT-L/14: 768-dim, ~60s per batch, may be slow on CPU
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CLIP_MODEL, CLIP_FEATURE_DIM, MLP_HIDDEN_DIMS,
    NUM_CLASSES, DROPOUT_RATE, DEVICE
)


class CLIPMLP(nn.Module):
    """
    CLIP image encoder (frozen) + trainable MLP classifier.

    The CLIP backbone is frozen — we only train the MLP head.
    This is fast on CPU because CLIP features only need to be
    extracted once and can be cached.

    Args:
        clip_model_name: e.g. "ViT-B/32" (CPU) or "ViT-L/14" (paper default)
        feature_dim:     Dimension of CLIP features (512 for ViT-B/32)
        hidden_dims:     List of hidden layer sizes in MLP
        num_classes:     2 (real vs fake)
        dropout_rate:    Regularization
    """

    def __init__(
        self,
        clip_model_name: str = CLIP_MODEL,
        feature_dim: int = CLIP_FEATURE_DIM,
        hidden_dims: list = MLP_HIDDEN_DIMS,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = DROPOUT_RATE,
    ):
        super(CLIPMLP, self).__init__()

        self.feature_dim = feature_dim
        self.clip_model_name = clip_model_name

        # Load CLIP model (frozen — not trained)
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load(
                clip_model_name, device="cpu"
            )
            # Freeze all CLIP parameters
            for param in self.clip_model.parameters():
                param.requires_grad = False
            print(f"Loaded CLIP {clip_model_name} (frozen)")
        except ImportError:
            raise ImportError(
                "CLIP not installed. Run: pip install clip-by-openai"
            )

        # Build MLP classifier on top of CLIP features
        layers = []
        in_dim = feature_dim
        for out_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def extract_clip_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP image features.
        Args:
            x: [batch, 3, H, W] — normalized image tensor (standard preprocessing)
        Returns:
            features: [batch, feature_dim] — L2-normalized CLIP features
        """
        with torch.no_grad():
            features = self.clip_model.encode_image(x)
            features = features.float()
            # L2 normalize (standard CLIP practice)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: CLIP feature extraction + MLP classification.
        Args:
            x: [batch, 3, H, W]
        Returns:
            logits: [batch, 2]
        """
        features = self.extract_clip_features(x)
        logits = self.mlp(features)
        return logits

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        MLP forward pass from pre-extracted features.
        Use this when features are cached to disk (faster training).
        Args:
            features: [batch, feature_dim]
        Returns:
            logits: [batch, 2]
        """
        return self.mlp(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class probabilities [batch, 2]."""
        return torch.softmax(self.forward(x), dim=1)

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.mlp.parameters())
        return trainable


class CachedCLIPFeatureExtractor:
    """
    Extract and cache CLIP features to disk.
    
    On CPU, CLIP feature extraction is slow (~1-2s per image with ViT-L/14).
    Extract all features once, save to disk, load for training.
    This makes subsequent training runs near-instant.

    Usage:
        extractor = CachedCLIPFeatureExtractor("ViT-B/32")
        features, labels = extractor.extract_and_cache(dataloader, "cache/train_features.pt")
    """

    def __init__(self, clip_model_name: str = CLIP_MODEL):
        import clip
        self.model, self.preprocess = clip.load(clip_model_name, device="cpu")
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def extract_and_cache(self, dataloader, cache_path: str):
        """
        Extract CLIP features for all images in dataloader.
        Saves to cache_path. If cache exists, loads from disk.
        """
        import os
        from tqdm import tqdm

        if os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            data = torch.load(cache_path)
            return data["features"], data["labels"]

        print(f"Extracting CLIP features (this takes a while on CPU)...")
        all_features = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting features"):
                features = self.model.encode_image(images)
                features = features.float()
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features)
                all_labels.append(labels)

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"features": features, "labels": labels}, cache_path)
        print(f"Saved features to {cache_path}")

        return features, labels


# ─── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        model = CLIPMLP()
        trainable = model.count_parameters()
        print(f"CLIP-MLP trainable parameters (MLP only): {trainable:,}")

        dummy = torch.randn(2, 3, 224, 224)
        logits = model(dummy)
        print(f"Output shape: {logits.shape}")
        print("CLIP-MLP OK!")
    except ImportError as e:
        print(f"Skipping CLIP test: {e}")
        print("Install CLIP with: pip install clip-by-openai")

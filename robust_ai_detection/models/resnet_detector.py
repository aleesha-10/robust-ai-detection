"""
models/resnet_detector.py

Fine-tuned ResNet18 binary classifier for AI-generated image detection.

Why ResNet18 (not ResNet50 or larger)?
  - Runs efficiently on CPU (ThinkPad i5 10th Gen)
  - Pre-trained on ImageNet → good low-level feature extraction
  - Small enough to train in <30 min on CPU with NUM_SAMPLES=2000

Architecture:
  ResNet18 backbone (pretrained) → Global Average Pool → 
  FC(512→256) → ReLU → Dropout → FC(256→2) → Softmax

This closely mirrors the "binary classification head on top of a frozen/
fine-tuned backbone" paradigm described in the base paper.
"""

import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_CLASSES, DROPOUT_RATE, RESNET_PRETRAINED, RESNET_FREEZE_BACKBONE


class ResNetDetector(nn.Module):
    """
    Fine-tuned ResNet18 for binary real/fake image classification.

    Args:
        pretrained:        Load ImageNet weights (strongly recommended)
        freeze_backbone:   If True, only train the classification head.
                           Faster on CPU; slightly lower accuracy.
        num_classes:       2 (real vs fake)
        dropout_rate:      Regularization (0.3 works well empirically)
    """

    def __init__(
        self,
        pretrained: bool = RESNET_PRETRAINED,
        freeze_backbone: bool = RESNET_FREEZE_BACKBONE,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = DROPOUT_RATE,
    ):
        super(ResNetDetector, self).__init__()

        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Optionally freeze backbone (train only the head)
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Remove ResNet's original classifier
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Output: [batch, 512, 1, 1] after Global Average Pool

        # Build a custom 2-layer classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # [batch, 512]
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),         # [batch, 2]
        )

        # Initialize the classifier weights
        self._init_weights()

    def _init_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 3, H, W] — normalized image tensor
        Returns:
            logits: [batch, 2] — raw scores (apply softmax for probabilities)
        """
        features = self.backbone(x)     # [batch, 512, 1, 1]
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature vectors (before classifier head).
        Useful for saliency analysis and feature visualization.
        Returns: [batch, 512]
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features.squeeze(-1).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns class probabilities [batch, 2].
        Index 0 = P(real), Index 1 = P(fake).
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def count_parameters(self):
        """Count trainable parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


# ─── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import IMAGE_SIZE, DEVICE

    model = ResNetDetector().to(DEVICE)
    trainable, total = model.count_parameters()
    print(f"ResNetDetector architecture:")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    # Test forward pass
    dummy = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    logits = model(dummy)
    probs  = model.predict_proba(dummy)

    print(f"\nForward pass:")
    print(f"  Input:   {dummy.shape}")
    print(f"  Logits:  {logits.shape}  (values: {logits[0].tolist()})")
    print(f"  Probs:   {probs.shape}   (sum: {probs[0].sum().item():.4f})")
    print(f"\nModel OK on {DEVICE}!")

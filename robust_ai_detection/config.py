"""
config.py — Central configuration for all experiments.

To switch from CPU → GPU later:
    DEVICE = "cuda"
    BATCH_SIZE = 128
    NUM_SAMPLES = -1   # use full dataset
"""

import os
import torch

# ─── Hardware ─────────────────────────────────────────────────────────────────
DEVICE = "cpu"          # Change to "cuda" when GPU is available
NUM_WORKERS = 0         # 0 is safest on Windows/CPU; increase on Linux with GPU

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "cifake")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

for d in [RESULTS_DIR, CHECKPOINT_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
# CIFAKE: 60k REAL (CIFAR-10) + 60k FAKE (Stable Diffusion equivalents)
# Classes: REAL=0, FAKE=1
IMAGE_SIZE = 32          # CIFAKE native; set to 224 if using other datasets
NUM_CLASSES = 2

# For CPU: limit samples to keep training feasible
# Set NUM_SAMPLES = -1 to use the full dataset (recommended with GPU)
NUM_SAMPLES = 2000       # per class per split — gives 4000 train, fast on CPU
                         # Change to -1 for full ~60k per class

TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# ─── Model ────────────────────────────────────────────────────────────────────
# Two model options (set MODEL_TYPE to switch):
# "resnet18"  — Fine-tuned ResNet18 (CPU-friendly, good baseline)
# "clip_mlp"  — CLIP ViT-L/14 + MLP (matches base paper; needs more RAM)
MODEL_TYPE = "resnet18"

# ResNet18 settings
RESNET_PRETRAINED = True
RESNET_FREEZE_BACKBONE = False   # True = only train final layer (faster on CPU)

# CLIP-MLP settings (used when MODEL_TYPE = "clip_mlp")
CLIP_MODEL = "ViT-B/32"          # ViT-B/32 is CPU-feasible; paper uses ViT-L/14
CLIP_FEATURE_DIM = 512           # 512 for ViT-B/32; 768 for ViT-L/14
MLP_HIDDEN_DIMS = [256, 128]     # Hidden layer sizes

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE    = 16      # CPU: 16–32; GPU: 128–256
NUM_EPOCHS    = 15      # 15 epochs is enough to show convergence
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
DROPOUT_RATE  = 0.3

# Early stopping
EARLY_STOPPING_PATIENCE = 5    # Stop if val loss doesn't improve for 5 epochs

# ─── Distribution Shift Experiments ───────────────────────────────────────────
# 1. Compression Shift — JPEG quality levels
JPEG_QUALITY_LEVELS = [95, 75, 50, 30, 10]

# 2. Resizing Shift — downsample then upsample
RESIZE_FACTORS = [1.0, 0.75, 0.5, 0.25]   # fraction of original size

# 3. Cross-Generator Shift
# When True, train on one generator type, test on another
CROSS_GENERATOR_TEST = True

# 4. Cross-Domain Shift
# CIFAKE mixes multiple categories; we split REAL→FAKE for domain analysis
CROSS_DOMAIN_TEST = True

# ─── Explainability ───────────────────────────────────────────────────────────
# Gradient-based saliency maps (no external library needed)
SALIENCY_NUM_EXAMPLES = 10     # how many images to generate saliency maps for
SALIENCY_CLASS = 1             # 1 = explain "AI-generated" predictions

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_INTERVAL = 10    # print loss every N batches
SAVE_BEST_MODEL = True

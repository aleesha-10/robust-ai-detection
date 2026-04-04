"""
train.py

Training script for the AI-generated image detector.

Usage:
    python train.py                         # Default settings from config.py
    python train.py --epochs 20             # Override epochs
    python train.py --model resnet18        # Use ResNet18 (default)
    python train.py --model clip_mlp        # Use CLIP-MLP (slower on CPU)
    python train.py --samples 500           # Use 500 samples per class

The script:
  1. Loads data from data/cifake/
  2. Trains the model with early stopping
  3. Saves best model checkpoint
  4. Logs training curves to results/logs/
  5. Plots training curves to results/plots/
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    BATCH_SIZE, NUM_SAMPLES, SEED, CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE, LOG_INTERVAL, SAVE_BEST_MODEL,
    MODEL_TYPE
)
from data.dataset import get_all_dataloaders
from utils.logger import ExperimentLogger
from utils.visualization import plot_training_curves


# ─── Reproducibility ─────────────────────────────────────────────────────────
def set_seed(seed: int):
    """Lock all randomness for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Model factory ───────────────────────────────────────────────────────────
def build_model(model_type: str):
    """Instantiate the appropriate model."""
    if model_type == "resnet18":
        from models.resnet_detector import ResNetDetector
        model = ResNetDetector()
    elif model_type == "clip_mlp":
        from models.clip_mlp import CLIPMLP
        model = CLIPMLP()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'resnet18' or 'clip_mlp'")

    return model.to(DEVICE)


# ─── One epoch of training ───────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, log_interval):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="  Train", leave=False)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        if (batch_idx + 1) % log_interval == 0:
            running_acc = correct / total
            # (verbose logging handled by tqdm; detailed logs go to CSV)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ─── Validation ──────────────────────────────────────────────────────────────
def validate(model, loader, criterion, device):
    """Evaluate on validation set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ─── Early stopping ──────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE, delta: float = 1e-4):
        self.patience   = patience
        self.delta      = delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.stop       = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                print(f"\n  Early stopping triggered (patience={self.patience})")
        return self.stop


# ─── Main training loop ───────────────────────────────────────────────────────
def train(
    model_type:  str = MODEL_TYPE,
    num_epochs:  int = NUM_EPOCHS,
    batch_size:  int = BATCH_SIZE,
    num_samples: int = NUM_SAMPLES,
    lr:          float = LEARNING_RATE,
    seed:        int = SEED,
):
    set_seed(seed)

    print(f"\nDevice: {DEVICE}")
    print(f"Model:  {model_type}")
    print(f"Epochs: {num_epochs}, Batch: {batch_size}, Samples/class: {num_samples}")
    print(f"LR: {lr}, Weight decay: {WEIGHT_DECAY}")

    # Data
    loaders = get_all_dataloaders(num_samples=num_samples, batch_size=batch_size)
    print(f"\nTrain: {len(loaders['train'].dataset)} samples")
    print(f"Val:   {len(loaders['val'].dataset)} samples")
    print(f"Test:  {len(loaders['test'].dataset)} samples")

    # Model
    model     = build_model(model_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    stopper   = EarlyStopping()
    logger    = ExperimentLogger(f"train_{model_type}")
    logger.save_config({"model_type": model_type, "num_epochs": num_epochs})

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_{model_type}.pt")

    # Training loop
    print(f"\n{'─'*60}")
    print("Training...")
    print(f"{'─'*60}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, DEVICE, LOG_INTERVAL
        )
        val_loss, val_acc = validate(model, loaders["val"], criterion, DEVICE)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)

        scheduler.step(val_loss)

        # Save best model
        if SAVE_BEST_MODEL and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":       epoch,
                "model_type":  model_type,
                "state_dict":  model.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "config": {
                    "model_type":  model_type,
                    "num_samples": num_samples,
                    "batch_size":  batch_size,
                    "lr":          lr,
                    "seed":        seed,
                }
            }, checkpoint_path)

        if stopper(val_loss):
            break

    # Plot training curves
    plot_training_curves(logger.training_rows)
    logger.summary()

    print(f"\nBest model saved: {checkpoint_path}")
    return checkpoint_path


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI-image detector")
    parser.add_argument("--model",   type=str, default=MODEL_TYPE,
                        choices=["resnet18", "clip_mlp"])
    parser.add_argument("--epochs",  type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE)
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES,
                        help="Samples per class per split. -1 = use all.")
    parser.add_argument("--lr",      type=float, default=LEARNING_RATE)
    parser.add_argument("--seed",    type=int,   default=SEED)
    args = parser.parse_args()

    train(
        model_type  = args.model,
        num_epochs  = args.epochs,
        batch_size  = args.batch,
        num_samples = args.samples,
        lr          = args.lr,
        seed        = args.seed,
    )

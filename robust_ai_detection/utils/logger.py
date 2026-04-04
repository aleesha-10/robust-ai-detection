"""
utils/logger.py

Experiment logging to CSV and console.
Every run produces a timestamped log so you always have a paper trail.
"""

import os
import sys
import csv
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOGS_DIR


class ExperimentLogger:
    """
    Logs training curves and evaluation results.

    Creates:
        results/logs/YYYYMMDD_HHMMSS_training.csv   — epoch-by-epoch loss/acc
        results/logs/YYYYMMDD_HHMMSS_eval.csv        — per-experiment metrics
        results/logs/YYYYMMDD_HHMMSS_config.json     — full config snapshot
    """

    def __init__(self, experiment_name: str = "experiment"):
        self.name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()

        # File paths
        prefix = os.path.join(LOGS_DIR, f"{self.timestamp}_{experiment_name}")
        self.training_csv = prefix + "_training.csv"
        self.eval_csv     = prefix + "_eval.csv"
        self.config_json  = prefix + "_config.json"

        # State
        self.training_rows = []
        self.eval_rows     = []
        self.best_val_loss = float("inf")
        self.best_epoch    = 0

        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"Logs: {LOGS_DIR}")
        print(f"{'='*60}\n")

    def save_config(self, extra: dict = None):
        """Save hyperparameters to JSON for full reproducibility."""
        import config as cfg
        config_dict = {
            k: v for k, v in vars(cfg).items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool, list))
        }
        if extra:
            config_dict.update(extra)
        config_dict["experiment_name"] = self.name
        config_dict["timestamp"] = self.timestamp

        with open(self.config_json, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"Config saved: {self.config_json}")

    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  val_loss: float, val_acc: float, lr: float = None):
        """Log one training epoch."""
        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(val_loss, 5),
            "val_acc":    round(val_acc, 4),
            "lr":         lr,
            "time_s":     round(time.time() - self.start_time, 1),
        }
        self.training_rows.append(row)
        self._write_csv(self.training_csv, self.training_rows)

        # Update best
        improved = "  ← best" if val_loss < self.best_val_loss else ""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        print(
            f"Epoch {epoch:>3} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val:   loss={val_loss:.4f} acc={val_acc:.4f}"
            + improved
        )

    def log_eval(self, experiment_name: str, metrics: dict, notes: str = ""):
        """Log evaluation results for one experiment (e.g., JPEG q=50)."""
        row = {"experiment": experiment_name, **metrics, "notes": notes}
        self.eval_rows.append(row)
        self._write_csv(self.eval_csv, self.eval_rows)

        print(
            f"  [{experiment_name}] "
            f"acc={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"auc={metrics['auc']:.4f}"
        )

    def _write_csv(self, path: str, rows: list):
        """Write rows to CSV, creating header on first write."""
        if not rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def summary(self):
        """Print final experiment summary."""
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"Experiment Complete: {self.name}")
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        print(f"Total time: {elapsed/60:.1f} min")
        print(f"Training log:  {self.training_csv}")
        print(f"Eval log:      {self.eval_csv}")
        print(f"{'='*60}\n")

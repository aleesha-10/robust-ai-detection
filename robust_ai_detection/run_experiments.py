"""
run_experiments.py

Master script — runs the COMPLETE pipeline end-to-end:
  1. Verify dataset exists
  2. Train the model
  3. Run all evaluation experiments
  4. Generate all figures
  5. Print final summary

Usage:
    python run_experiments.py                  # Full run with defaults
    python run_experiments.py --quick          # Tiny subset (debug/test)
    python run_experiments.py --skip-train     # Skip training, eval only
    python run_experiments.py --gpu            # Enable GPU (if available)

For your laptop (CPU, 16GB RAM):
    python run_experiments.py
    # Expected time: ~55 min with NUM_SAMPLES=2000
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_dataset():
    from config import DATA_DIR
    required = [
        os.path.join(DATA_DIR, "train", "REAL"),
        os.path.join(DATA_DIR, "train", "FAKE"),
        os.path.join(DATA_DIR, "test",  "REAL"),
        os.path.join(DATA_DIR, "test",  "FAKE"),
    ]
    missing = [d for d in required if not os.path.isdir(d)]
    if missing:
        print("\nDataset not found. Please run:")
        print("  python data/download_data.py")
        print("\nOr download CIFAKE manually from Kaggle:")
        print("  https://www.kaggle.com/datasets/bird012/cifake-real-and-ai-generated-synthetic-images")
        print(f"\nExpected location: {DATA_DIR}")
        sys.exit(1)
    print("Dataset verified.")


def main():
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument("--quick",       action="store_true",
                        help="Use 200 samples/class for a quick test run")
    parser.add_argument("--skip-train",  action="store_true",
                        help="Skip training, run evaluation only")
    parser.add_argument("--gpu",         action="store_true",
                        help="Use GPU (sets DEVICE=cuda)")
    parser.add_argument("--model",       type=str, default=None,
                        choices=["resnet18", "clip_mlp"])
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--samples",     type=int, default=None)
    args = parser.parse_args()

    # ── Override config if flags given ───────────────────────────────────────
    import config
    if args.gpu:
        import torch
        if torch.cuda.is_available():
            config.DEVICE = "cuda"
            config.BATCH_SIZE = 128
            print("GPU mode enabled.")
        else:
            print("Warning: --gpu flag set but CUDA not available. Using CPU.")

    if args.quick:
        config.NUM_SAMPLES = 200
        config.NUM_EPOCHS  = 3
        config.SALIENCY_NUM_EXAMPLES = 4
        print("Quick mode: 200 samples/class, 3 epochs")

    if args.model:
        config.MODEL_TYPE = args.model

    if args.epochs:
        config.NUM_EPOCHS = args.epochs

    if args.samples:
        config.NUM_SAMPLES = args.samples

    # ── Print configuration ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"  Device:      {config.DEVICE}")
    print(f"  Model:       {config.MODEL_TYPE}")
    print(f"  Epochs:      {config.NUM_EPOCHS}")
    print(f"  Batch size:  {config.BATCH_SIZE}")
    print(f"  Samples/cls: {config.NUM_SAMPLES} (-1 = all)")
    print(f"  JPEG levels: {config.JPEG_QUALITY_LEVELS}")
    print(f"  Seed:        {config.SEED}")
    print("="*60)

    start = time.time()

    # ── Step 1: Verify dataset ────────────────────────────────────────────────
    print("\n[Step 1] Verifying dataset...")
    verify_dataset()

    # ── Step 2: Visualize sample images ──────────────────────────────────────
    print("\n[Step 2] Saving sample image grid...")
    try:
        from data.dataset import CIFAKEDataset
        from utils.visualization import plot_sample_grid
        ds = CIFAKEDataset(split="test", num_samples=20)
        samples = ds.get_sample_images(n=4)
        plot_sample_grid(samples["REAL"], samples["FAKE"])
        print("  Sample grid saved.")
    except Exception as e:
        print(f"  Skipping sample grid: {e}")

    # ── Step 3: Train model ───────────────────────────────────────────────────
    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR, f"best_{config.MODEL_TYPE}.pt"
    )

    if not args.skip_train:
        print(f"\n[Step 3] Training {config.MODEL_TYPE}...")
        from train import train
        checkpoint_path = train(
            model_type  = config.MODEL_TYPE,
            num_epochs  = config.NUM_EPOCHS,
            batch_size  = config.BATCH_SIZE,
            num_samples = config.NUM_SAMPLES,
            lr          = config.LEARNING_RATE,
            seed        = config.SEED,
        )
    else:
        print(f"\n[Step 3] Skipping training. Using: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            print("Remove --skip-train to train first.")
            sys.exit(1)

    # ── Step 4: Full evaluation ───────────────────────────────────────────────
    print(f"\n[Step 4] Running all evaluation experiments...")
    from evaluate import run_all_experiments
    results = run_all_experiments(checkpoint_path)

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = (time.time() - start) / 60
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed:.1f} minutes.")
    print(f"\nOutputs:")
    print(f"  Plots:  {config.PLOTS_DIR}")
    print(f"  Logs:   {config.LOGS_DIR}")
    print(f"  Model:  {checkpoint_path}")
    print(f"\nNext steps:")
    print(f"  1. Open results/plots/ to see all figures")
    print(f"  2. Open results/logs/*_eval.csv for the metrics table")
    print(f"  3. Open notebooks/analysis.ipynb for interactive exploration")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

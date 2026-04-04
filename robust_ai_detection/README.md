# Robust Detection of AI-Generated Images Under Real-World Distribution Shifts

## Hardware This Was Designed For
- **CPU**: Intel Core i5-10210U / i5-10310U (10th Gen, quad-core)
- **RAM**: 16GB DDR4
- **GPU**: None (Integrated Intel UHD Graphics — CPU-only mode)
- **Storage**: NVMe SSD recommended (fast I/O for dataset loading)

> **GPU Note**: All code runs on CPU. If you later get GPU access (Colab, Kaggle),
> just change `DEVICE = "cpu"` → `DEVICE = "cuda"` in `config.py`. Nothing else changes.

---

## Project Structure
```
robust_ai_detection/
├── README.md
├── requirements.txt
├── config.py                  # All hyperparameters and paths in one place
├── setup_env.sh               # One-command environment setup
│
├── data/
│   ├── download_data.py       # Downloads CIFAKE (CPU-friendly dataset)
│   ├── dataset.py             # PyTorch Dataset class
│   └── augmentations.py       # JPEG, resizing, domain augmentations
│
├── models/
│   ├── clip_mlp.py            # CLIP-based MLP detector (from base paper)
│   ├── resnet_detector.py     # ResNet18 fine-tuned detector (CPU-friendly)
│   └── feature_extractor.py   # Shared feature extraction utilities
│
├── utils/
│   ├── metrics.py             # Accuracy, F1, AUC, confusion matrix
│   ├── logger.py              # Experiment logging (CSV + console)
│   └── visualization.py       # Saliency maps, confidence plots, result graphs
│
├── train.py                   # Training script
├── evaluate.py                # Full evaluation + distribution shift testing
├── run_experiments.py         # Runs ALL experiments end-to-end
│
├── results/                   # Auto-created: logs, plots, model checkpoints
└── notebooks/
    └── analysis.ipynb         # Interactive analysis notebook
```

---

## Quick Start (Your Laptop)

### Step 1: Set Up Environment
```bash
cd robust_ai_detection
bash setup_env.sh
conda activate robust_ai_det
```

### Step 2: Download Dataset
```bash
python data/download_data.py
```
This downloads **CIFAKE** (~200MB) — a purpose-built real vs AI-generated dataset
that works well on CPU. See `config.py` to switch to a larger dataset if needed.

### Step 3: Run All Experiments
```bash
python run_experiments.py
```
This runs the full pipeline:
1. Trains the model
2. Tests in-distribution performance
3. Tests compression shift (JPEG q=95, 75, 50, 30, 10)
4. Tests cross-generator shift
5. Tests cross-domain shift
6. Generates saliency maps
7. Plots all results and saves to `results/`

### Step 4 (Optional): Interactive Analysis
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Reproducing on GPU Later
When you have Colab/Kaggle access:
1. Open `config.py`
2. Change `DEVICE = "cpu"` → `DEVICE = "cuda"`
3. Change `BATCH_SIZE = 16` → `BATCH_SIZE = 128`
4. Change `NUM_SAMPLES = 1000` → `NUM_SAMPLES = -1` (use full dataset)
5. Re-run `python run_experiments.py`

That's it. All results will be comparable, just faster and with more data.

---

## Expected Runtime (Your Laptop, CPU)
| Step | Time |
|------|------|
| Data download | ~5 min |
| Feature extraction (1000 samples) | ~15 min |
| Training (10 epochs) | ~20 min |
| Compression shift eval | ~10 min |
| Saliency map generation | ~5 min |
| **Total** | **~55 min** |

With GPU (Colab T4, full dataset): ~15 min total.

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

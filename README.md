# Convolutional vs Transformer Detectors: A Robustness Study Under Real-World Distribution Shifts

This repository contains the full implementation for a comparative study of CNN and Transformer-based architectures for detecting AI-generated images under real-world distribution shifts. The study evaluates ResNet18 and CLIP ViT-L/14 across four perturbation types: JPEG compression, spatial resize, generator shift, and domain shift.

The core finding is that spatial resize is significantly more destructive to detector performance than JPEG compression, a result absent from prior work. Multi-scale training augmentation is proposed as a solution, recovering 94% of resize-induced performance loss without any architectural changes.

---

## Key Results

| Model | Baseline AUC | Resize x0.75 AUC | JPEG q=10 AUC |
|---|---|---|---|
| ResNet18 (baseline) | 0.9957 | 0.9207 | 0.9516 |
| ResNet18 + Augmentation | 0.9964 | 0.9924 | 0.9644 |
| CLIP ViT-L/14 | 0.9929 | 0.9762 | 0.8152 |
| Ensemble (soft-vote) | 0.9912 | 0.9475 | 0.7996 |

---

## Project Structure

```
robust_ai_detection/
├── README.md
├── requirements.txt
├── config.py                    # All hyperparameters and paths in one place
├── setup_env.sh                 # One-command environment setup
│
├── data/
│   ├── download_data.py         # Downloads CIFAKE dataset
│   ├── dataset.py               # PyTorch Dataset class
│   └── augmentations.py        # JPEG, resizing, and domain augmentations
│
├── models/
│   ├── clip_mlp.py              # CLIP ViT-L/14 with trainable MLP head
│   ├── resnet_detector.py       # Fine-tuned ResNet18 detector
│   └── feature_extractor.py    # Shared feature extraction utilities
│
├── utils/
│   ├── metrics.py               # Accuracy, F1, AUC, confusion matrix
│   ├── logger.py                # Experiment logging (CSV + console)
│   └── visualization.py        # Grad-CAM, confidence plots, result graphs
│
├── train.py                     # Training script
├── evaluate.py                  # Full evaluation with distribution shift testing
├── run_experiments.py           # Runs all experiments end-to-end
│
├── results/                     # Auto-created: logs, plots, model checkpoints
└── notebooks/
    └── analysis.ipynb           # Interactive analysis notebook
```

---

## Setup

### Requirements

- Python 3.9 or higher
- CUDA 12.2 (for GPU training; CPU mode is supported for ResNet18)
- 10 GB VRAM recommended for CLIP ViT-L/14 training

### Install dependencies

```bash
bash setup_env.sh
```

Or manually:

```bash
pip install -r requirements.txt
```

### Download data

```bash
python data/download_data.py
```

This downloads the CIFAKE dataset (real images from CIFAR-10 paired with Stable Diffusion generated fakes). The dataset contains 100,000 training images and 20,000 test images. For generator shift evaluation, the GenImage GAN subset (BigGAN-generated objects) is also downloaded. The faces dataset (Flickr real faces and StyleGAN fakes) is used for zero-shot cross-domain evaluation.

---

## Configuration

All hyperparameters and dataset paths are centralized in `config.py`. Key defaults:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 128 |
| Max epochs | 20 |
| Early stopping patience | 5 |
| Dropout rate | 0.3 |
| Random seed | 42 |

The CLIP backbone is frozen during training. Only the MLP projection head (hidden dimensions 256, 128) is optimized. ResNet18 is fully fine-tuned with the final classification layer replaced by a two-class output head.

---

## Training

Train ResNet18 baseline:

```bash
python train.py --model resnet18
```

Train ResNet18 with multi-scale augmentation:

```bash
python train.py --model resnet18 --augment
```

Train CLIP ViT-L/14:

```bash
python train.py --model clip
```

Checkpoints are saved to `results/checkpoints/`. Early stopping is applied based on validation loss with a patience of 5 epochs.

---

## Evaluation

Run full evaluation including all distribution shift conditions:

```bash
python evaluate.py --model resnet18 --checkpoint results/checkpoints/resnet18_best.pt
python evaluate.py --model clip --checkpoint results/checkpoints/clip_best.pt
```

Shift types evaluated:
- JPEG compression at quality levels 95, 75, 50, 30, and 10
- Spatial resize at factors 0.75, 0.5, and 0.25 (bilinear downsample then upsample)
- Generator shift: zero-shot on GenImage BigGAN subset
- Domain shift: zero-shot on StyleGAN faces dataset

All shifts are applied at test time only.

---

## Running All Experiments

To reproduce all results from the paper in sequence:

```bash
python run_experiments.py
```

This runs the following experiments:

1. ResNet18 baseline on full CIFAKE
2. ResNet18 with multi-scale augmentation
3. CLIP ViT-L/14 on full CIFAKE
4. Cross-domain zero-shot evaluation on faces dataset
5. Generator shift zero-shot evaluation on GenImage GAN subset
6. Soft-vote ensemble of ResNet18 and CLIP

Logs, metrics, and plots are written to `results/`.

---

## Ensemble

The soft-vote ensemble averages softmax probabilities from ResNet18 and CLIP ViT-L/14 with equal weights. No additional training is required. The ensemble achieves the highest baseline AUC (0.9912) but is outperformed by CLIP alone under severe resize conditions (resize x0.5: CLIP 0.9157, ensemble 0.8194), indicating that equal-weight averaging is pulled down by the weaker model at extreme shifts.

---

## Datasets

| Dataset | Purpose | Size |
|---|---|---|
| CIFAKE | Primary training and evaluation | 120,000 images |
| GenImage GAN subset | Generator shift evaluation (BigGAN) | 998 images |
| Faces dataset | Cross-domain evaluation (StyleGAN) | 20,000 images |

CIFAKE uses real images from CIFAR-10 and fake images generated by Stable Diffusion. It is split into training (70%), validation (15%), and test (15%) partitions with a fixed seed of 42. The faces and GenImage datasets are used only for zero-shot evaluation; no fine-tuning is performed on either.

Note: The base paper's ELSA_D3 dataset (approximately 3 TB, H5 format) is not used here due to storage and compute constraints. All results use CIFAKE as the benchmark and are not directly comparable to numbers reported in the base paper.

---

## Models

### ResNet18

A pretrained ResNet18 backbone with ImageNet weights, fully fine-tuned for binary classification. Images are resized to 32x32 and normalized using ImageNet mean and standard deviation. This is the CPU-feasible detector.

### CLIP ViT-L/14

A frozen CLIP ViT-L/14 vision encoder with a trainable three-layer MLP head (hidden dimensions 256, 128). Images are resized to 224x224. Freezing the backbone preserves the pretrained semantic representations and prevents overfitting to dataset-specific texture cues.

Note: Assignment 2 used CLIP ViT-B/32 for CPU compatibility. All results in the paper use CLIP ViT-L/14 on GPU. The ViT-B/32 results are not included in final comparisons.

---

## Visualization and Explainability

Grad-CAM is applied to ResNet18 across clean, resized, and out-of-domain conditions to provide visual evidence of where the model attends. On clean CIFAKE fakes the model attends to structured object regions. Under resize, attention becomes diffuse. On face images, attention is unfocused and broadly distributed across facial texture, consistent with quantitative collapse.

Visualization utilities are in `utils/visualization.py` and include:
- Grad-CAM heatmaps
- Softmax confidence distribution plots under shift conditions
- AUC and accuracy bar charts across all shift types

---

## Findings Summary

Resize is more destructive than compression. ResNet18 maintains AUC above 0.98 at JPEG quality 50 but drops to 0.9207 at resize factor 0.75. This occurs because detectors rely on spatial frequency patterns that are destroyed by downsampling and upsampling, while JPEG artifacts preserve enough spatial structure for detection to continue.

Multi-scale augmentation works by exposing the model to different spatial scales during training via randomly resized crops. It recovers 94% of resize-induced performance loss at resize x0.75 with no architectural changes.

CNN and Transformer failure modes differ. Under resize, ResNet18 loses both AUC and accuracy, while CLIP retains high AUC but experiences accuracy collapse. This means CLIP continues to rank real and fake images correctly under moderate resize but its decision boundary shifts, causing misclassification despite correct relative ordering.

Detectors learn generator-specific artifacts. Switching from Stable Diffusion (training) to BigGAN (test) with domain held constant drops AUC from 0.9771 to 0.7445 and reduces fake recall to 3.4%. Combined generator and domain shift (StyleGAN faces) produces complete prediction collapse at AUC 0.4890.

---

## Limitations

- Isolated domain shift using Stable Diffusion generated faces was not completed; the combined shift result (StyleGAN faces, AUC=0.4890) cannot be fully decomposed without it.
- Combined resize and JPEG shift applied simultaneously was not evaluated.
- The ensemble uses equal weighting only; adaptive weighting based on shift severity is left as future work.
- Results are not directly comparable to the base paper (Santosh et al., AVSS 2024) due to dataset differences.

---

## References

- Bird and Lotfi, "CIFAKE: Image classification and explainable identification of AI-generated synthetic images," IEEE Access, 2024.
- Santosh et al., "Robust CLIP-based detector for exposing diffusion model-generated images," AVSS 2024.
- Radford et al., "Learning transferable visual models from natural language supervision," ICML 2021.
- Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," ICCV 2017.
- Zhu et al., "GenImage: A million-scale benchmark for detecting AI-generated image," NeurIPS 2023.
- Wang et al., "CNN-generated images are surprisingly easy to spot... for now," CVPR 2020.
- Guarnera et al., "Advancing deepfake detection: New insights into GAN and diffusion model artifacts," AVSS 2024.

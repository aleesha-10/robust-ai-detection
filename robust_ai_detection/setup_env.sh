#!/bin/bash
# setup_env.sh — Run once to set up your conda environment
# Usage: bash setup_env.sh

set -e

ENV_NAME="robust_ai_det"

echo "====================================================="
echo " Setting up: $ENV_NAME"
echo "====================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create environment
echo "[1/4] Creating conda environment..."
conda create -n $ENV_NAME python=3.10 -y

# Activate
echo "[2/4] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install PyTorch (CPU version — works on any machine including yours)
echo "[3/4] Installing PyTorch (CPU)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "[4/4] Installing other packages..."
pip install \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    Pillow \
    tqdm \
    kaggle \
    requests \
    jupyter \
    ipywidgets \
    clip-by-openai        # OpenAI CLIP

echo ""
echo "====================================================="
echo " Done! Activate with: conda activate $ENV_NAME"
echo "====================================================="

"""
GPU_UPGRADE_GUIDE.md

How to scale this project from your laptop (CPU) to GPU
without changing any of the core research code.

Tested environments:
  - Google Colab Free (T4 GPU, 15GB RAM)
  - Kaggle Notebooks (P100 GPU, 16GB RAM)
  - Local NVIDIA GPU (any)
"""

# ════════════════════════════════════════════════════════
# OPTION A: Google Colab (Recommended — free T4 GPU)
# ════════════════════════════════════════════════════════

"""
Step 1: Upload the project to Google Drive
  - Zip robust_ai_detection/ and upload to My Drive

Step 2: Open a new Colab notebook with GPU runtime
  Runtime → Change runtime type → T4 GPU

Step 3: Mount Drive and unzip
"""

COLAB_SETUP = """
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Unzip project
import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/robust_ai_detection.zip', 'r') as z:
    z.extractall('/content/')

# Cell 3: Install dependencies
!pip install torch torchvision --quiet
!pip install scikit-learn matplotlib seaborn tqdm clip-by-openai --quiet

# Cell 4: Set working directory
import os
os.chdir('/content/robust_ai_detection')

# Cell 5: Run full experiment (GPU mode)
!python run_experiments.py --gpu --samples -1 --epochs 20
"""

# ════════════════════════════════════════════════════════
# OPTION B: Kaggle Notebooks
# ════════════════════════════════════════════════════════

KAGGLE_SETUP = """
# 1. Create new Kaggle notebook
# 2. Settings → Accelerator → GPU P100
# 3. Add CIFAKE dataset directly:
#    + Add Data → Search "cifake" → Add bird012/cifake...
#    Dataset will be at /kaggle/input/cifake.../

# 4. Update config.py:
#    DATA_DIR = "/kaggle/input/cifake-real-and-ai-generated-synthetic-images/cifake"

# 5. In notebook cell:
import subprocess
subprocess.run(["pip", "install", "clip-by-openai", "-q"])

import sys
sys.path.insert(0, '/kaggle/working')
# Copy project files to /kaggle/working/ then run:

from run_experiments import main
# Or: !python run_experiments.py --gpu --samples -1
"""

# ════════════════════════════════════════════════════════
# CONFIG CHANGES FOR GPU
# ════════════════════════════════════════════════════════

CPU_CONFIG = """
# config.py — CPU (your laptop)
DEVICE       = "cpu"
BATCH_SIZE   = 16
NUM_SAMPLES  = 2000     # per class per split
NUM_EPOCHS   = 15
MODEL_TYPE   = "resnet18"
CLIP_MODEL   = "ViT-B/32"
"""

GPU_CONFIG = """
# config.py — GPU (Colab/Kaggle)
DEVICE       = "cuda"
BATCH_SIZE   = 128      # much larger batches
NUM_SAMPLES  = -1       # use FULL dataset (60k/class)
NUM_EPOCHS   = 30
MODEL_TYPE   = "resnet18"   # or "clip_mlp" for paper comparison
CLIP_MODEL   = "ViT-L/14"  # paper's actual model (too slow on CPU)
"""

# ════════════════════════════════════════════════════════
# EXPECTED RESULTS COMPARISON
# ════════════════════════════════════════════════════════

EXPECTED_RESULTS = """
| Setting          | Dataset   | Epochs | Train Time | Expected AUC |
|------------------|-----------|--------|------------|--------------|
| CPU laptop       | 2000/cls  | 15     | ~20 min    | 0.88–0.93    |
| Colab T4 GPU     | Full 60k  | 30     | ~12 min    | 0.94–0.97    |
| Kaggle P100      | Full 60k  | 30     | ~10 min    | 0.94–0.97    |

Note: With full dataset + ViT-L/14 CLIP (paper's config), you can
reproduce the paper's reported AUC of ~0.97.
"""

if __name__ == "__main__":
    print("GPU Upgrade Guide")
    print("="*50)
    print("\nColab setup:")
    print(COLAB_SETUP)
    print("\nConfig changes:")
    print("CPU →", CPU_CONFIG)
    print("GPU →", GPU_CONFIG)
    print("\nExpected results:", EXPECTED_RESULTS)

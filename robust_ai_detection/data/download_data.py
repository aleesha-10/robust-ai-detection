"""
data/download_data.py

Downloads and organizes the CIFAKE dataset.

CIFAKE is an ideal dataset for CPU-based research:
  - 60,000 REAL images (CIFAR-10 test images)  
  - 60,000 FAKE images (Stable Diffusion v1.4 generated)
  - 32x32 pixels — very fast to process on CPU
  - Publicly available on Kaggle: bird012/cifake-real-and-ai-generated-synthetic-images
  - Cited in peer-reviewed research (Bird & Lotfi, 2023)

Dataset structure after download:
    data/cifake/
        train/
            REAL/   (50,000 images)
            FAKE/   (50,000 images)
        test/
            REAL/   (10,000 images)
            FAKE/   (10,000 images)

Reference:
    Bird, J.J. & Lotfi, A. (2024). CIFAKE: Image Classification and
    Explainable Identification of AI-Generated Synthetic Images.
    IEEE Access.
"""

import os
import sys
import zipfile
import shutil
import subprocess
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, BASE_DIR

KAGGLE_DATASET = "bird012/cifake-real-and-ai-generated-synthetic-images"
DOWNLOAD_DIR   = os.path.join(BASE_DIR, "data", "downloads")
CIFAKE_ZIP     = os.path.join(DOWNLOAD_DIR, "cifake.zip")


def check_kaggle_credentials():
    """
    Check if Kaggle API credentials are set up.
    If not, print instructions.
    """
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("\n" + "="*60)
        print("KAGGLE CREDENTIALS NOT FOUND")
        print("="*60)
        print("\nTo download CIFAKE, you need a Kaggle account.")
        print("Steps:")
        print("  1. Sign up at https://www.kaggle.com (free)")
        print("  2. Go to: https://www.kaggle.com/settings")
        print("  3. Scroll to 'API' section → 'Create New Token'")
        print("  4. Download kaggle.json")
        print("  5. Move it: cp ~/Downloads/kaggle.json ~/.kaggle/")
        print("  6. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("  7. Re-run this script")
        print("\nAlternative (manual):")
        print("  Download from: https://www.kaggle.com/datasets/bird012/cifake-real-and-ai-generated-synthetic-images")
        print(f"  Extract to: {DATA_DIR}")
        print("="*60 + "\n")
        return False
    return True


def download_cifake():
    """Download CIFAKE dataset from Kaggle."""
    if not check_kaggle_credentials():
        sys.exit(1)

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print(f"Downloading CIFAKE dataset (~200MB)...")
    print(f"Source: kaggle datasets download -d {KAGGLE_DATASET}")

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-p", DOWNLOAD_DIR, "--unzip"],
            check=True
        )
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        print("Try downloading manually (see instructions above).")
        sys.exit(1)


def organize_dataset():
    """
    Ensure dataset is organized as:
        data/cifake/train/REAL/
        data/cifake/train/FAKE/
        data/cifake/test/REAL/
        data/cifake/test/FAKE/
    """
    print("Organizing dataset structure...")

    # CIFAKE comes pre-organized in train/REAL, train/FAKE, test/REAL, test/FAKE
    # Check if already in right place
    expected_dirs = [
        os.path.join(DATA_DIR, "train", "REAL"),
        os.path.join(DATA_DIR, "train", "FAKE"),
        os.path.join(DATA_DIR, "test", "REAL"),
        os.path.join(DATA_DIR, "test", "FAKE"),
    ]

    # Common: Kaggle extracts to a subfolder
    possible_source = os.path.join(DOWNLOAD_DIR, "cifake")
    if not os.path.exists(DATA_DIR) and os.path.exists(possible_source):
        print(f"Moving {possible_source} → {DATA_DIR}")
        shutil.move(possible_source, DATA_DIR)

    # Verify
    all_exist = all(os.path.isdir(d) for d in expected_dirs)
    if not all_exist:
        print("WARNING: Dataset structure is unexpected. Please check:")
        for d in expected_dirs:
            status = "✓" if os.path.isdir(d) else "✗"
            print(f"  {status} {d}")
        print(f"\nDownloaded files are in: {DOWNLOAD_DIR}")
        print("Please manually organize to match the expected structure.")
        return False

    # Count images
    for split in ["train", "test"]:
        for cls in ["REAL", "FAKE"]:
            path = os.path.join(DATA_DIR, split, cls)
            n = len([f for f in os.listdir(path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {split}/{cls}: {n:,} images")

    return True


def main():
    print("\n" + "="*60)
    print("CIFAKE Dataset Setup")
    print("="*60)

    # Check if already downloaded
    already_done = all(os.path.isdir(os.path.join(DATA_DIR, s, c))
                       for s in ["train", "test"] for c in ["REAL", "FAKE"])
    if already_done:
        print("Dataset already present. Counting images...")
        organize_dataset()
        print("\nDataset is ready!")
        return

    download_cifake()
    success = organize_dataset()

    if success:
        print("\nDataset ready. You can now run:")
        print("  python run_experiments.py")


if __name__ == "__main__":
    main()

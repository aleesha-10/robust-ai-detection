"""
gradcam.py

Grad-CAM visualization for ResNet18 detector.
Shows which image regions the model attends to under:
  1. Clean CIFAKE image (correct prediction)
  2. Resized CIFAKE image (model fails)
  3. Face image (complete domain collapse)

Output: results/plots/gradcam_analysis.png
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")  # No display needed — saves to file
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, CHECKPOINT_DIR, PLOTS_DIR, DATA_DIR
from models.resnet_detector import ResNetDetector

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_resnet18_baseline.pt")

FACES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "faces", "real_vs_fake", "real-vs-fake", "test"
)

# ─── Grad-CAM ────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM implementation for ResNet18.
    Hooks into the last convolutional block (layer4).
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook into backbone[-3] = layer4
        target_layer = model.backbone[-3]

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_idx=1):
        """
        Generate Grad-CAM heatmap.
        class_idx=1 means we visualize what makes the model say "FAKE"
        """
        self.model.zero_grad()
        image_tensor = image_tensor.requires_grad_(True)

        logits = self.model(image_tensor)
        score = logits[0, class_idx]
        score.backward()

        # Weight activations by gradient
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to image size
        cam = F.interpolate(cam, size=(224, 224),
                           mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_model():
    model = ResNetDetector()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(DEVICE)
    model.train()  # Grad-CAM needs train mode for gradients
    # But disable dropout for deterministic results
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    return model


def get_transform(resize_factor=None):
    ops = []
    if resize_factor and resize_factor < 1.0:
        small = max(4, int(224 * resize_factor))
        ops.append(transforms.Resize((small, small)))
    ops.append(transforms.Resize((224, 224)))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transforms.Compose(ops)


def load_image(path, resize_factor=None):
    """Load image as PIL and as normalized tensor."""
    pil = Image.open(path).convert("RGB").resize((224, 224))
    transform = get_transform(resize_factor)

    # Apply resize factor to PIL for display too
    if resize_factor and resize_factor < 1.0:
        small = max(4, int(224 * resize_factor))
        degraded = pil.resize((small, small)).resize((224, 224), Image.BILINEAR)
        display_pil = degraded
    else:
        display_pil = pil

    tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    return display_pil, tensor


def get_prediction(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        conf = probs[0, pred].item()
    label = "FAKE" if pred == 1 else "REAL"
    return label, conf


def overlay_heatmap(pil_image, cam, alpha=0.5):
    """Overlay Grad-CAM heatmap on PIL image."""
    heatmap = plt.cm.jet(cam)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap).resize((224, 224))

    img_array = np.array(pil_image).astype(float)
    heat_array = np.array(heatmap_pil).astype(float)
    blended = (alpha * heat_array + (1 - alpha) * img_array).astype(np.uint8)
    return Image.fromarray(blended)


def find_sample_images():
    """Find one real and one fake image from CIFAKE test set."""
    real_dir = os.path.join(DATA_DIR, "test", "REAL")
    fake_dir = os.path.join(DATA_DIR, "test", "FAKE")

    real_img = os.path.join(real_dir, sorted(os.listdir(real_dir))[0])
    fake_img = os.path.join(fake_dir, sorted(os.listdir(fake_dir))[0])

    return real_img, fake_img


def find_face_image():
    """Find one fake face image."""
    fake_dir = os.path.join(FACES_DIR, "fake")
    img = os.path.join(fake_dir, sorted(os.listdir(fake_dir))[0])
    return img


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("Grad-CAM Analysis — ResNet18 Detector")
    print("="*60)

    model = load_model()
    gradcam = GradCAM(model)

    real_path, fake_path = find_sample_images()
    face_path = find_face_image()

    # Define all conditions to visualize
    conditions = [
        ("Clean CIFAKE\n(Real image)", real_path,  None,  0),  # class 0 = real
        ("Clean CIFAKE\n(Fake image)", fake_path,  None,  1),  # class 1 = fake
        ("Resized ×0.25\n(Fake image)", fake_path, 0.25,  1),  # degraded
        ("Face image\n(StyleGAN fake)", face_path, None,  1),  # domain shift
    ]

    fig, axes = plt.subplots(
        3, len(conditions),
        figsize=(4 * len(conditions), 10)
    )

    fig.suptitle(
        "Grad-CAM: Where ResNet18 Looks When Detecting AI-Generated Images",
        fontsize=13, fontweight="bold", y=1.01
    )

    row_labels = ["Original Image", "Grad-CAM Heatmap", "Overlay"]

    for col, (title, path, resize_factor, class_idx) in enumerate(conditions):
        print(f"  Processing: {title.replace(chr(10), ' ')}...")

        pil_img, tensor = load_image(path, resize_factor)

        # Get prediction
        pred_label, conf = get_prediction(model, tensor)

        # Generate Grad-CAM — need fresh tensor with grad
        tensor_grad = tensor.clone().requires_grad_(True)
        cam = gradcam.generate(tensor_grad, class_idx=class_idx)

        # Create overlay
        overlay = overlay_heatmap(pil_img, cam)

        # Plot row 0: original
        axes[0, col].imshow(pil_img)
        axes[0, col].set_title(
            f"{title}\nPred: {pred_label} ({conf:.2f})",
            fontsize=9
        )
        axes[0, col].axis("off")

        # Plot row 1: heatmap
        axes[1, col].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[1, col].axis("off")

        # Plot row 2: overlay
        axes[2, col].imshow(overlay)
        axes[2, col].axis("off")

    # Row labels on left
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10, rotation=90,
                                labelpad=10, va="center")

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, "gradcam_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()

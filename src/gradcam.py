import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DEVICE
from src.model import EurosatCNN
from src.utils import load_model
from src.dataset import load_stats

# ── Load stats from saved JSON (same as training) ────────
MEAN, STD = load_stats()

# ── GradCAM class ────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_idx=None):
        self.model.eval()

        output = self.model(tensor)               # [1, 10]

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam    -= cam.min()
        cam    /= cam.max() + 1e-8

        return cam.squeeze().cpu().numpy(), class_idx


# ── Visualise single image ────────────────────────────────
def visualise_gradcam(image_path, model, gradcam, classes, save_path=None):
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    image  = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)           # [1, 3, 64, 64]

    cam, pred_idx = gradcam.generate(tensor)
    pred_class    = classes[pred_idx]

    cam_image = Image.fromarray((cam * 255).astype(np.uint8))
    cam_image = cam_image.resize((64, 64), Image.BILINEAR)
    cam_array = np.array(cam_image) / 255.0

    img_array = np.array(image.resize((64, 64))) / 255.0
    heatmap   = plt.cm.jet(cam_array)[:, :, :3]
    overlay   = 0.5 * img_array + 0.5 * heatmap

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(img_array)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam_array, cmap="jet")
    axes[1].set_title("GradCAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Pred: {pred_class}")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()
    plt.close()


# ── Run GradCAM on a list of images ──────────────────────
# FIX: accepts model + classes as arguments — no hardcoding
def run_gradcam(model, classes, test_images: list = None, data_dir: str = "data/raw"):
    os.makedirs("outputs/gradcam", exist_ok=True)

    # Default: one image per class from data_dir
    if test_images is None:
        test_images = []
        for cls in classes:
            folder = os.path.join(data_dir, cls)
            if os.path.exists(folder):
                first_img = sorted(os.listdir(folder))[0]
                test_images.append(os.path.join(folder, first_img))

    # Target last conv layer in block3
    target_layer = model.block3[3]
    gradcam      = GradCAM(model, target_layer)

    for img_path in test_images:
        name      = os.path.splitext(os.path.basename(img_path))[0]
        save_path = f"outputs/gradcam/{name}_gradcam.png"
        print(f"  Processing: {img_path}")
        visualise_gradcam(img_path, model, gradcam, classes, save_path)


# ── Entry point (standalone use only) ────────────────────
if __name__ == "__main__":
    from config import CLASSES, DATA_DIR
    device = torch.device(DEVICE)
    model  = EurosatCNN(num_classes=len(CLASSES)).to(device)
    model  = load_model(model, "outputs/best_model.pth", device)
    run_gradcam(model, CLASSES, data_dir=DATA_DIR)
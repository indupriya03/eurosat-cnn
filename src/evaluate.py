import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sklearn.metrics import classification_report
from src.dataset import get_dataloaders
from src.model import EurosatCNN
from src.utils import save_confusion_matrix, load_model
from config import DEVICE, MODEL_PATH

# ── Device ───────────────────────────────────────────────
device = torch.device(DEVICE)
print(f"Using device: {device}")

# ── Evaluate function ────────────────────────────────────
# FIX: accepts loaders as arguments instead of creating them internally
def evaluate(model, test_loader, classes):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Classification report ────────────────────────────
    print("=" * 60)
    print("TEST SET — CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        all_labels, all_preds,
        target_names=classes,
        digits=3
    ))

    # ── Overall accuracy ─────────────────────────────────
    acc = (all_preds == all_labels).mean()
    print(f"Overall Test Accuracy: {acc:.3f} ({acc*100:.1f}%)")

    # ── Per-class accuracy ───────────────────────────────
    print("\nPer-class accuracy:")
    for i, cls in enumerate(classes):
        cls_mask = all_labels == i
        cls_acc  = (all_preds[cls_mask] == all_labels[cls_mask]).mean()
        print(f"  {cls:<25} {cls_acc:.3f}")

    # ── Confusion matrix ─────────────────────────────────
    save_confusion_matrix(all_labels, all_preds, classes)

    return all_preds, all_labels

# ── Entry point (standalone use only) ───────────────────
if __name__ == "__main__":
    _, _, test_loader, classes = get_dataloaders()
    model = EurosatCNN(num_classes=len(classes)).to(device)
    model = load_model(model, MODEL_PATH, device)
    evaluate(model, test_loader, classes)
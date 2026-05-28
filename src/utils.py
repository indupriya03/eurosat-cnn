import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Used by: train.py ────────────────────────────────────
def save_training_curves(history: dict, save_path: str = "outputs/training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"],   label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")

# ── Used by: evaluate.py ─────────────────────────────────
def save_confusion_matrix(all_labels, all_preds, classes,
                          save_path: str = "outputs/confusion_matrix.png"):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5
    )
    plt.title("Confusion Matrix — Test Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")

# ── Used by: app.py + evaluate.py ────────────────────────
def load_model(model, checkpoint_path: str, device: torch.device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# ── Used by: train.py ────────────────────────────────────
def save_checkpoint(model, val_acc: float, best_val_acc: float,
                    epoch: int, save_path: str = "outputs/best_model.pth"):
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), save_path)
        print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")
        return val_acc, epoch
    return best_val_acc, epoch
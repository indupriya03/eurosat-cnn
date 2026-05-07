import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import LR, WEIGHT_DECAY, EPOCHS, DEVICE
from src.dataset import get_dataloaders
from src.model import EurosatCNN
from src.utils import save_training_curves, save_checkpoint

# ── Device ───────────────────────────────────────────────
device = torch.device(DEVICE)
print(f"Using device: {device}")

# ── Single epoch ─────────────────────────────────────────
def run_epoch(loader, model, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs    = model(images)
            loss       = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += len(labels)
    return total_loss / total, correct / total

# ── Training loop ────────────────────────────────────────
# FIX: accepts loaders as arguments instead of creating them internally
def train(model, train_loader, val_loader, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch   = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, model, optimizer, criterion, train=True)
        vl_loss, vl_acc = run_epoch(val_loader,   model, optimizer, criterion, train=False)
        scheduler.step(vl_acc)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        best_val_acc, best_epoch = save_checkpoint(
            model, vl_acc, best_val_acc, epoch
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {tr_loss:.4f}  acc {tr_acc:.3f} | "
            f"val loss {vl_loss:.4f}  acc {vl_acc:.3f}"
            + (" ← best" if epoch == best_epoch else "")
        )

    save_training_curves(history)
    print(f"\nBest val acc: {best_val_acc:.3f} at epoch {best_epoch}")
    return model

# ── Entry point (standalone use only) ───────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    model = EurosatCNN(num_classes=len(classes)).to(device)
    train(model, train_loader, val_loader)
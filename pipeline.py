import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DEVICE, MODEL_PATH, DATA_DIR
from src.dataset import get_dataloaders
from src.model import EurosatCNN
from src.train import train
from src.evaluate import evaluate
from src.gradcam import run_gradcam
from src.utils import load_model

# ── Device ───────────────────────────────────────────────
device = torch.device(DEVICE)

def run_pipeline(skip_train: bool = False, run_cam: bool = True):
    print("=" * 60)
    print("EUROSAT PIPELINE")
    print("=" * 60)

    # Step 1 — Data (single call, shared across all steps)
    print("\n[1/4] Loading data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    # Step 2 — Model
    model = EurosatCNN(num_classes=len(classes)).to(device)

    # Step 3 — Train or load existing checkpoint
    if skip_train and os.path.exists(MODEL_PATH):
        print(f"\n[2/4] Skipping training — loading checkpoint from {MODEL_PATH}")
        model = load_model(model, MODEL_PATH, device)
    else:
        print("\n[2/4] Training...")
        model = train(model, train_loader, val_loader)
        model = load_model(model, MODEL_PATH, device)   # load best checkpoint

    # Step 4 — Evaluate
    print("\n[3/4] Evaluating on test set...")
    evaluate(model, test_loader, classes)

    # Step 5 — GradCAM
    if run_cam:
        print("\n[4/4] Running GradCAM...")
        run_gradcam(model, classes, data_dir=DATA_DIR)
    else:
        print("\n[4/4] Skipping GradCAM")

    print("\n✓ Pipeline complete")
    print("  Outputs saved → outputs/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training and load existing checkpoint"
    )
    parser.add_argument(
        "--no-gradcam", action="store_true",
        help="Skip GradCAM visualisation"
    )
    args = parser.parse_args()
    run_pipeline(skip_train=args.skip_train, run_cam=not args.no_gradcam)
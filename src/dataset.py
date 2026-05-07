import json
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR, BATCH_SIZE, STATS_PATH, SEED, PIN_MEMORY

# ── Compute mean/std on train set only ───────────────────
def compute_and_save_stats(train_indices, full_dataset):
    print("Computing mean/std on train set...")

    channel_sum    = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    n_pixels       = 0

    to_tensor = transforms.ToTensor()   # only ToTensor, no normalize

    for idx in train_indices:
        img_path, _ = full_dataset.imgs[idx]
        try:
            img             = to_tensor(Image.open(img_path).convert("RGB"))  # [3, H, W]
            channel_sum    += img.sum(dim=[1, 2])
            channel_sum_sq += (img ** 2).sum(dim=[1, 2])
            n_pixels       += img.shape[1] * img.shape[2]   # H × W
        except Exception:
            pass

    mean = channel_sum    / n_pixels
    std  = (channel_sum_sq / n_pixels - mean ** 2).sqrt()

    mean = mean.tolist()
    std  = std.tolist()

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=4)

    print(f"✓ Saved stats → {STATS_PATH}")
    print(f"  Mean : {[round(m, 4) for m in mean]}")
    print(f"  Std  : {[round(s, 4) for s in std]}")
    return mean, std

# ── Load stats ───────────────────────────────────────────
def load_stats():
    with open(STATS_PATH) as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]

# ── Transforms ───────────────────────────────────────────
def get_transforms(split: str, mean, std):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomResizedCrop(
                size=64,
                scale=(0.8, 1.0),    # keep 80-100% of image
                ratio=(0.9, 1.1)     # near square crops only
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,      # slight saturation for spectral variation
                hue=0.02             # very small hue shift
            ),
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 0.5)     # very mild blur only
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

# ── Main function ─────────────────────────────────────────
def get_dataloaders(data_dir: str = DATA_DIR, batch_size: int = BATCH_SIZE):

    # Step 1 — load raw dataset, no transforms yet
    full    = datasets.ImageFolder(root=data_dir)
    indices = list(range(len(full)))
    labels  = [label for _, label in full.imgs]

    # Step 2 — stratified split using sklearn
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=SEED,
        stratify=labels
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_labels
    )

    print(f"Split → Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Step 3 — compute mean/std on train set only
    train_subset = Subset(full, train_idx)   # needed for compute_and_save_stats
    train_subset.indices = train_idx         # explicit for clarity

    if os.path.exists(STATS_PATH):
        print("✓ Loading existing stats...")
        mean, std = load_stats()
    else:
        mean, std = compute_and_save_stats(train_idx, full)

    # Step 4 — apply transforms per split
    full.transform = None                    # reset first

    train_dataset = Subset(
        datasets.ImageFolder(root=data_dir, transform=get_transforms("train", mean, std)),
        train_idx
    )
    val_dataset = Subset(
        datasets.ImageFolder(root=data_dir, transform=get_transforms("val", mean, std)),
        val_idx
    )
    test_dataset = Subset(
        datasets.ImageFolder(root=data_dir, transform=get_transforms("test", mean, std)),
        test_idx
    )

    # Step 5 — dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=PIN_MEMORY)

    print(f"Classes : {full.classes}")
    return train_loader, val_loader, test_loader, full.classes


# ── Sanity check ─────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape : {images.shape}")  # [32, 3, 64, 64]
    print(f"Labels shape: {labels.shape}")   # [32]
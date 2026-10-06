
"""
Separate corrupted-training experiment.

Uses the existing Week 4 MatchedBackbone and configuration.
The original week4_deliverables/train.py is not modified.

Training:
- CIFAR-10 training split
- RandomCrop + RandomHorizontalFlip
- Randomly selected corruption and severity 1-5
- Cross-entropy + SGD + cosine annealing
- Checkpoint and CSV log saved separately

Evaluation:
- Does not use the CIFAR-10 test set during training.
"""

import sys
import csv
import json
import os
import random
import time
from pathlib import Path

# Make the existing Week 4 model/config modules importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEEK4_DIR = PROJECT_ROOT / "week4_deliverables"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(WEEK4_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as trn

import config
from models.backbone import build_model
from week4_corrupted.corruptions import CIFAR10CCorruption


# Keep all outputs for this experiment separate.
OUTPUT_DIR = PROJECT_ROOT / "week4_corrupted"
CHECKPOINT_ROOT = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_corrupted_transform():
    """Build the training transform with random corruption augmentation."""
    return trn.Compose([
        trn.RandomCrop(32, padding=4),
        trn.RandomHorizontalFlip(),
        CIFAR10CCorruption(p=1.0),
        trn.ToTensor(),
        trn.Normalize(
            config.NORMALIZE_MEAN,
            config.NORMALIZE_STD
        ),
    ])


def load_corrupted_cifar10(subset=None):
    """
    Load CIFAR-10 with corruptions applied only to training images.
    The test dataset is deliberately not loaded by the training loop.
    """
    train_ds = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR,
        train=True,
        download=False,
        transform=get_corrupted_transform()
    )

    if subset is not None and subset < len(train_ds):
        indices = list(range(subset))
        train_ds = torch.utils.data.Subset(train_ds, indices)

    return train_ds


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    variant,
    epoch
):
    """Run one training epoch."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        batch_count = x.size(0)
        running_loss += loss.item() * batch_count
        correct += (outputs.argmax(dim=1) == y).sum().item()
        total += batch_count

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    current_lr = optimizer.param_groups[0]["lr"]

    row = {
        "experiment": "corrupted_training",
        "variant": variant,
        "epoch": epoch,
        "train_loss": average_loss,
        "train_acc": accuracy,
        "lr": current_lr,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return average_loss, accuracy, current_lr, row


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    seed,
    variant,
    metrics
):
    """Save a checkpoint with experiment identification."""
    config_values = {
        key: value
        for key, value in vars(config).items()
        if key.isupper() and not key.startswith("__")
    }

    torch.save({
        "experiment": "corrupted_training",
        "variant": variant,
        "seed": seed,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_values,
        "metrics": metrics,
    }, path)


def write_log_csv(path, rows):
    """Write the epoch history to CSV."""
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def train_corrupted_variant(
    variant="none",
    epochs=None,
    subset=None,
    seed=None,
    batch_size=None,
    lr=None,
    device=None
):
    """Train one attention variant using corrupted training images."""

    epochs = config.EPOCHS if epochs is None else epochs
    subset = config.TRAIN_SUBSET if subset is None else subset
    seed = config.SEED if seed is None else seed
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    lr = config.LEARNING_RATE if lr is None else lr

    if variant not in {"none", "se", "bam", "cbam"}:
        raise ValueError(
            "variant must be one of: none, se, bam, cbam"
        )

    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_dir = CHECKPOINT_ROOT / f"week4_corr_ckpts_{variant}_s{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    train_ds = load_corrupted_cifar10(subset=subset)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    model = build_model(attention_type=variant).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    log_rows = []
    train_loss = 0.0
    train_acc = 0.0

    print(f"Experiment: corrupted training")
    print(f"Variant: {variant}")
    print(f"Seed: {seed}")
    print(f"Epochs: {epochs}")
    print(f"Training images: {len(train_ds)}")
    print(f"Device: {device}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Log directory: {LOG_DIR}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, current_lr, row = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            variant,
            epoch
        )

        log_rows.append(row)

        checkpoint_path = checkpoint_dir / (
            f"corrupted_{variant}_seed{seed}_epoch{epoch}.pt"
        )

        save_checkpoint(
            str(checkpoint_path),
            model,
            optimizer,
            epoch,
            seed,
            variant,
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
            }
        )

        print(
            f"[corrupted | {variant}] "
            f"epoch {epoch}/{epochs}: "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"lr={current_lr:.6f}"
        )

        scheduler.step()

    log_path = LOG_DIR / f"corrupted_{variant}_seed{seed}.csv"
    write_log_csv(str(log_path), log_rows)

    final_checkpoint = checkpoint_dir / (
        f"corrupted_{variant}_seed{seed}_epoch{epochs}.pt"
    )

    return {
        "experiment": "corrupted_training",
        "variant": variant,
        "seed": seed,
        "epochs": epochs,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "checkpoint": str(final_checkpoint),
        "log": str(log_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a Week 4 model with corruption augmentation."
    )

    parser.add_argument(
        "--variant",
        choices=["none", "se", "bam", "cbam"],
        default="none"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    args = parser.parse_args()

    results = train_corrupted_variant(
        variant=args.variant,
        epochs=args.epochs,
        subset=args.subset,
        seed=args.seed,
        batch_size=args.batch_size,
        lr=args.lr
    )

    print(json.dumps(results, indent=2))
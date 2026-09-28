"""Batched CIFAR-10 / CIFAR-10-C evaluation.

Evaluates one checkpoint on:
- Clean CIFAR-10 test set
- 4 primary corruptions
- Severities 1-5

CIFAR-10-C is processed in batches to avoid large RAM allocations.
"""

import json
import os
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as trn

import config
from models.backbone import build_model


BATCH_SIZE = 128


def load_clean_test(subset=None):
    eval_tf = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD),
    ])

    ds = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR,
        train=False,
        download=False,
        transform=eval_tf
    )

    if subset is not None and subset < len(ds):
        ds = torch.utils.data.Subset(
            ds,
            list(range(subset))
        )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


def load_corruption_arrays(name):
    """Load CIFAR-10-C files using memory mapping."""

    arr = np.load(
        os.path.join(config.CIFAR10C_DIR, f"{name}.npy"),
        mmap_mode="r"
    )

    labels = np.load(
        os.path.join(config.CIFAR10C_DIR, "labels.npy"),
        mmap_mode="r"
    )

    assert arr.shape == (50000, 32, 32, 3)
    assert arr.dtype == np.uint8
    assert labels.shape == (50000,)

    return arr, labels


def corruption_batches(arr, labels, severity, subset=None):
    """Yield one severity in small batches."""

    n = subset if subset is not None else 10000

    start_index = (severity - 1) * 10000

    mean = torch.tensor(
        config.NORMALIZE_MEAN,
        dtype=torch.float32
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        config.NORMALIZE_STD,
        dtype=torch.float32
    ).view(1, 3, 1, 1)

    for start in range(0, n, BATCH_SIZE):

        end = min(start + BATCH_SIZE, n)

        # Copy only this small batch into RAM
        images_np = np.array(
            arr[start_index + start:start_index + end],
            copy=True
        )

        labels_np = np.array(
            labels[start_index + start:start_index + end],
            copy=True
        )

        # HWC -> CHW
        x = torch.from_numpy(images_np)
        x = x.permute(0, 3, 1, 2)

        # uint8 -> float32 [0,1]
        x = x.float().div_(255.0)

        # Normalize
        x = (x - mean) / std

        y = torch.from_numpy(labels_np).long()

        yield x, y


@torch.no_grad()
def evaluate_batches(model, batches, device):
    """Evaluate model batch-by-batch."""

    correct = 0
    total = 0

    model.eval()

    for x, y in batches:

        x = x.to(device)
        y = y.to(device)

        output = model(x)

        predictions = output.argmax(dim=1)

        correct += (predictions == y).sum().item()
        total += y.size(0)

    accuracy = correct / max(total, 1)

    return accuracy, total


def evaluate_variant(
    variant,
    seed=None,
    ckpt_path=None,
    device=None
):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    seed = config.SEED if seed is None else seed

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # Build exact same architecture
    model = build_model(
        attention_type=variant
    ).to(device)

    # Load checkpoint
    if ckpt_path and os.path.exists(ckpt_path):

        print(f"Loading checkpoint:")
        print(ckpt_path)

        checkpoint = torch.load(
            ckpt_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        loaded_epoch = checkpoint.get(
            "epoch"
        )

        print(
            f"Checkpoint epoch: {loaded_epoch}"
        )

    else:
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}"
        )

    results = {
        "variant": variant,
        "seed": seed,
        "ckpt_path": ckpt_path,
        "loaded_epoch": loaded_epoch,
        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "corruptions": {}
    }

    # -------------------------------------------------
    # CLEAN CIFAR-10
    # -------------------------------------------------

    print("\nEvaluating CLEAN CIFAR-10...")

    clean_loader = load_clean_test(
        subset=config.EVAL_CLEAN_SUBSET
    )

    clean_acc, clean_total = evaluate_batches(
        model,
        clean_loader,
        device
    )

    results["corruptions"]["clean"] = {
        "severity": 0,
        "accuracy": clean_acc,
        "num_examples": clean_total
    }

    print(
        f"Clean accuracy: {clean_acc * 100:.2f}%"
    )

    # -------------------------------------------------
    # CIFAR-10-C
    # -------------------------------------------------

    for corruption in config.PRIMARY_CORRUPTIONS:

        print(
            f"\n===== {corruption} ====="
        )

        arr, labels = load_corruption_arrays(
            corruption
        )

        for severity in config.SEVERITIES:

            print(
                f"Severity {severity}/5...",
                end=" ",
                flush=True
            )

            batches = corruption_batches(
                arr,
                labels,
                severity,
                subset=config.EVAL_CORRUPTION_SUBSET
            )

            acc, total = evaluate_batches(
                model,
                batches,
                device
            )

            results["corruptions"].setdefault(
                corruption,
                {}
            )[severity] = {
                "accuracy": acc,
                "num_examples": total
            }

            print(
                f"{acc * 100:.2f}%"
            )

    return results


def summarize(results):

    output = {
        "variant": results["variant"],
        "seed": results["seed"],
        "loaded_epoch": results["loaded_epoch"],
        "clean_acc": results["corruptions"]["clean"]["accuracy"]
    }

    for corruption in config.PRIMARY_CORRUPTIONS:

        accuracies = [
            results["corruptions"][corruption][severity]["accuracy"]
            for severity in config.SEVERITIES
        ]

        output[corruption] = {
            "sev1_5_acc": accuracies,
            "mean": float(np.mean(accuracies))
        }

    return output


if __name__ == "__main__":

    import sys

    variant = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "none"
    )

    ckpt = (
        sys.argv[2]
        if len(sys.argv) > 2
        else None
    )

    results = evaluate_variant(
        variant,
        ckpt_path=ckpt
    )

    print("\n==============================")
    print("FINAL SUMMARY")
    print("==============================")

    print(
        json.dumps(
            summarize(results),
            indent=2
        )
    )
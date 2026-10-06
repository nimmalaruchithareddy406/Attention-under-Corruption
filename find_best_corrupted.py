import csv
from pathlib import Path

base = Path(r"C:\Attention-under-Corruption\week4_corrupted")

log_dir = base / "logs"
ckpt_dir = base / "checkpoints"

runs = [
    ("none", 0),
    ("none", 1),
    ("none", 2),

    ("se", 0),
    ("se", 1),
    ("se", 2),

    ("bam", 0),
    ("bam", 1),
    ("bam", 2),

    ("cbam", 0),
    ("cbam", 1),
    ("cbam", 2),
]

results = []

for variant, seed in runs:

    log_file = log_dir / f"corrupted_{variant}_seed{seed}.csv"

    best_acc = -1
    best_epoch = None

    with open(log_file, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            epoch = int(row["epoch"])
            train_acc = float(row["train_acc"])

            if train_acc > best_acc:
                best_acc = train_acc
                best_epoch = epoch

    # Find the checkpoint folder
    if seed == 0:
        folder_name = f"week4_corr_ckpts_{variant}"
    else:
        folder_name = f"week4_corr_ckpts_{variant}_s{seed}"

    folder = ckpt_dir / folder_name

    # Checkpoint naming pattern
    checkpoint_name = f"corrupted_{variant}_seed{seed}_epoch{best_epoch}.pt"
    checkpoint_path = folder / checkpoint_name

    results.append(
        (
            variant,
            seed,
            best_epoch,
            best_acc,
            checkpoint_path
        )
    )

print()
print("=" * 110)
print("BEST CHECKPOINT FOR EACH CORRUPTED MODEL")
print("=" * 110)

for variant, seed, epoch, acc, path in results:

    exists = "FOUND" if path.exists() else "NOT FOUND"

    print(
        f"{variant.upper():6} | "
        f"Seed: {seed} | "
        f"Best Epoch: {epoch:3} | "
        f"Train Acc: {acc*100:7.2f}% | "
        f"{path.name} | {exists}"
    )

print("=" * 110)
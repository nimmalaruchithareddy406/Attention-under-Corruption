import torch
from pathlib import Path

base = Path(r"C:\cifar_data")

folders = [
    "week4_ckpts_none",
    "week4_ckpts_none_s1",
    "week4_ckpts_none_s2",

    "week4_ckpts_se",
    "week4_ckpts_se_s1",
    "week4_ckpts_se_s2",

    "week4_ckpts_bam",
    "week4_ckpts_bam_s1",
    "week4_ckpts_bam_s2",

    "week4_ckpts_cbam",
    "week4_ckpts_cbam_s1",
    "week4_ckpts_cbam_s2",
]

results = []

for folder_name in folders:
    folder = base / folder_name

    best_acc = -1
    best_file = None
    best_epoch = None

    for p in folder.glob("*.pt"):
        ckpt = torch.load(p, map_location="cpu", weights_only=False)

        val_acc = ckpt["metrics"]["val_acc"]
        epoch = ckpt["epoch"]

        if val_acc > best_acc:
            best_acc = val_acc
            best_file = p
            best_epoch = epoch

    results.append((folder_name, best_epoch, best_acc, best_file))

print()
print("=" * 90)
print("BEST CHECKPOINT FOR EACH CLEAN MODEL")
print("=" * 90)

for folder_name, epoch, acc, path in results:
    print(f"{folder_name:30} | Epoch: {epoch:3} | "
          f"Val Acc: {acc*100:7.2f}% | {path.name}")

print("=" * 90)
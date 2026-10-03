
import os
import sys
import torch

# Add the project root so config and train can be imported.
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, PROJECT_ROOT)

import config
import train

CHECKPOINT_ROOT = r"C:\cifar_data"

runs = [
    ("none", os.path.join(CHECKPOINT_ROOT, "week4_ckpts_none_s2")),
    ("se",   os.path.join(CHECKPOINT_ROOT, "week4_ckpts_se_s2")),
    ("bam",  os.path.join(CHECKPOINT_ROOT, "week4_ckpts_bam_s2")),
    ("cbam", os.path.join(CHECKPOINT_ROOT, "week4_ckpts_cbam_s2")),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(config.NUM_THREADS)

for variant, checkpoint_dir in runs:
    log_dir = os.path.join(
        CHECKPOINT_ROOT, f"week4_logs_{variant}_s2"
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n===== STARTING SEED 2: {variant.upper()} =====", flush=True)

    model, metrics = train.train_variant(
        variant=variant,
        epochs=100,
        subset=None,
        seed=2,
        batch_size=config.BATCH_SIZE,
        ckpt_dir=checkpoint_dir,
        log_dir=log_dir,
        device=device,
    )

    print(f"===== FINISHED SEED 2: {variant.upper()} =====", flush=True)
    print(f"Checkpoint: {metrics['ckpt_path']}", flush=True)

print("\nALL FOUR SEED 2 CLEAN TRAINING RUNS FINISHED.", flush=True)
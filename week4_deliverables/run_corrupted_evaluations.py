import os
import json
import time

from evaluate_batched import evaluate_variant


# ============================================================
# SETTINGS
# ============================================================

CHECKPOINT_DIR = r"C:\best_checkpoints_for_testing\corrupted"

RESULTS_DIR = r".\evaluation_results"

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "corrupted_results.json"
)

VARIANTS = [
    "none",
    "se",
    "bam",
    "cbam"
]

SEEDS = [
    0,
    1,
    2
]


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

if os.path.exists(RESULTS_FILE):

    print("Existing results file found.")

    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

else:

    print("No existing results file found.")
    print("Creating a new one.")

    results = []


# ============================================================
# HELPER FUNCTION
# ============================================================

def experiment_key(condition, variant, seed):

    return (
        condition,
        variant,
        int(seed)
    )


# ============================================================
# FIND ALREADY COMPLETED EXPERIMENTS
# ============================================================

completed = set()

for result in results:

    key = experiment_key(
        "corrupted",
        result["variant"],
        result["seed"]
    )

    completed.add(key)


print()
print("==============================================")
print("CORRUPTED EVALUATION SETUP")
print("==============================================")

print(
    f"Already completed: "
    f"{len(completed)}/12"
)

print(
    f"Remaining: "
    f"{12 - len(completed)}/12"
)

print("==============================================")


# ============================================================
# RUN ALL 12 CHECKPOINTS
# ============================================================

for variant in VARIANTS:

    for seed in SEEDS:

        key = experiment_key(
            "corrupted",
            variant,
            seed
        )

        checkpoint = os.path.join(
            CHECKPOINT_DIR,
            f"{variant}_seed{seed}.pt"
        )

        # ----------------------------------------------------
        # SKIP IF ALREADY COMPLETED
        # ----------------------------------------------------

        if key in completed:

            print()
            print(
                f"SKIPPING: "
                f"{variant}_seed{seed}"
            )

            continue


        # ----------------------------------------------------
        # CHECK CHECKPOINT EXISTS
        # ----------------------------------------------------

        if not os.path.exists(checkpoint):

            print()
            print(
                "ERROR: checkpoint not found:"
            )

            print(checkpoint)

            continue


        # ----------------------------------------------------
        # START EXPERIMENT
        # ----------------------------------------------------

        print()
        print()
        print("##############################################")
        print("STARTING CORRUPTED CHECKPOINT")
        print(
            f"Variant : {variant}"
        )
        print(
            f"Seed    : {seed}"
        )
        print(
            f"Checkpoint: {checkpoint}"
        )
        print("##############################################")


        start_time = time.time()


        try:

            result = evaluate_variant(
                variant=variant,
                seed=seed,
                ckpt_path=checkpoint
            )


            # Add experiment metadata

            result["condition"] = "corrupted"

            result["variant"] = variant

            result["seed"] = seed

            result["checkpoint"] = checkpoint


            # ------------------------------------------------
            # ADD RESULT
            # ------------------------------------------------

            results.append(result)


            # ------------------------------------------------
            # SAVE IMMEDIATELY
            # ------------------------------------------------

            with open(
                RESULTS_FILE,
                "w"
            ) as f:

                json.dump(
                    results,
                    f,
                    indent=2
                )


            elapsed = (
                time.time() - start_time
            )


            print()
            print("==============================================")
            print("CHECKPOINT COMPLETED")
            print(
                f"Variant : {variant}"
            )
            print(
                f"Seed    : {seed}"
            )
            print(
                f"Time    : {elapsed / 60:.2f} minutes"
            )
            print()
            print(
                "RESULT SAVED IMMEDIATELY TO:"
            )
            print(
                os.path.abspath(
                    RESULTS_FILE
                )
            )
            print("==============================================")


            # Mark completed

            completed.add(key)


        except Exception as e:

            print()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("ERROR")
            print(
                f"Variant : {variant}"
            )
            print(
                f"Seed    : {seed}"
            )
            print()
            print(e)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            print()
            print(
                "Continuing to the next checkpoint..."
            )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("==============================================")
print("CORRUPTED EVALUATION RUN FINISHED")
print("==============================================")

print(
    f"Completed: {len(results)}/12"
)

print(
    f"Remaining: {12 - len(results)}/12"
)

print()
print(
    "Results file:"
)

print(
    os.path.abspath(
        RESULTS_FILE
    )
)

print("==============================================")
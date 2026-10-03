import csv
from pathlib import Path

rows = list(csv.DictReader(open("robustness_detailed_summary.csv")))

conditions = [
    r for r in rows
    if r["training_condition"] == "corrupted_trained"
    and r["corruption"] != "ALL_CORRUPTIONS"
]

out = []

for corruption in ["brightness", "contrast", "defocus_blur", "elastic_transform"]:
    for severity in range(1, 6):

        base = next(
            r for r in conditions
            if r["variant"] == "none"
            and r["corruption"] == corruption
            and int(r["severity"]) == severity
        )

        base_acc = float(base["mean_accuracy"])

        for variant in ["se", "bam", "cbam"]:
            r = next(
                r for r in conditions
                if r["variant"] == variant
                and r["corruption"] == corruption
                and int(r["severity"]) == severity
            )

            acc = float(r["mean_accuracy"])
            gain = acc - base_acc

            out.append([
                variant,
                corruption,
                severity,
                acc,
                base_acc,
                gain
            ])

path = Path("attention_gain_vs_none.csv")

with path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "attention_variant",
        "corruption",
        "severity",
        "attention_accuracy",
        "none_accuracy",
        "gain_vs_none"
    ])
    w.writerows(out)

print(f"Wrote {path}")
print(f"Rows: {len(out)}")

print("\nMean gain vs None:")
for variant in ["se", "bam", "cbam"]:
    values = [
        float(r["gain_vs_none"])
        for r in out
        if r[0] == variant
    ]

    mean_gain = sum(values) / len(values)

    print(
        f"{variant.upper():4s} | "
        f"{mean_gain*100:+.3f} percentage points"
    )

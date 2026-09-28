import json
import csv
import statistics
from pathlib import Path

VARIANTS = ["none", "se", "bam", "cbam"]
CORRUPTIONS = ["brightness", "contrast", "defocus_blur", "elastic_transform"]

clean_data = json.loads(Path("evaluation_results/clean_results.json").read_text())
corrupt_data = json.loads(Path("evaluation_results/corrupted_results.json").read_text())

def mean_sd(values):
    return statistics.mean(values), statistics.pstdev(values)

def get_acc(entry, corruption, severity):
    if corruption == "clean":
        return entry["corruptions"]["clean"]["accuracy"]
    return entry["corruptions"][corruption][str(severity)]["accuracy"]

rows = []

for variant in VARIANTS:
    clean_entries = [x for x in clean_data if x["variant"] == variant]
    corrupt_entries = [x for x in corrupt_data if x["variant"] == variant]

    clean_baseline_clean = [get_acc(x, "clean", 0) for x in clean_entries]
    corrupt_baseline_clean = [get_acc(x, "clean", 0) for x in corrupt_entries]

    clean_base_mean, clean_base_sd = mean_sd(clean_baseline_clean)
    corrupt_base_mean, corrupt_base_sd = mean_sd(corrupt_baseline_clean)

    for training_condition, entries, baseline in [
        ("clean_trained", clean_entries, clean_base_mean),
        ("corrupted_trained", corrupt_entries, corrupt_base_mean)
    ]:
        corruption_accs = []

        for corruption in CORRUPTIONS:
            for severity in range(1, 6):
                accs = [
                    get_acc(x, corruption, severity)
                    for x in entries
                ]

                m, sd = mean_sd(accs)
                degradation = baseline - m

                rows.append([
                    variant,
                    training_condition,
                    corruption,
                    severity,
                    m,
                    sd,
                    degradation
                ])

                corruption_accs.extend(accs)

        overall_mean, overall_sd = mean_sd(corruption_accs)
        overall_degradation = baseline - overall_mean

        rows.append([
            variant,
            training_condition,
            "ALL_CORRUPTIONS",
            0,
            overall_mean,
            overall_sd,
            overall_degradation
        ])

summary = Path("robustness_detailed_summary.csv")

with summary.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "variant",
        "training_condition",
        "corruption",
        "severity",
        "mean_accuracy",
        "sd",
        "degradation_from_condition_clean"
    ])
    w.writerows(rows)

print(f"Wrote {summary}")
print(f"Rows: {len(rows)}")

print("\nOverall corruption performance:")
for variant in VARIANTS:
    for condition in ["clean_trained", "corrupted_trained"]:
        r = [
            x for x in rows
            if x[0] == variant
            and x[1] == condition
            and x[2] == "ALL_CORRUPTIONS"
        ][0]

        print(
            f"{variant.upper():5s} | "
            f"{condition:17s} | "
            f"accuracy={r[4]*100:.3f}% | "
            f"SD={r[5]*100:.3f}% | "
            f"degradation={r[6]*100:.3f} pp"
        )

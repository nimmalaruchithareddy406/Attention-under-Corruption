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

    clean_accs = [get_acc(x, "clean", 0) for x in clean_entries]
    cm, cs = mean_sd(clean_accs)

    rows.append([variant, "clean", 0, cm, cs, 0.0])

    for corruption in CORRUPTIONS:
        for severity in range(1, 6):
            accs = [get_acc(x, corruption, severity) for x in corrupt_entries]
            m, sd = mean_sd(accs)
            degradation = cm - m
            rows.append([variant, corruption, severity, m, sd, degradation])

out = Path("robustness_summary.csv")

with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "variant",
        "condition",
        "severity",
        "mean_accuracy",
        "sd",
        "degradation_from_clean"
    ])
    w.writerows(rows)

print(f"Wrote {out}")
print(f"Rows: {len(rows)}")

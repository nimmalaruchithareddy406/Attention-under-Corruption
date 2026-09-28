import csv

rows = list(csv.reader(open("attention_gain_vs_none.csv")))

header = rows[0]
data = rows[1:]

variant_index = header.index("attention_variant")
gain_index = header.index("gain_vs_none")

print("\nMean gain vs None:")

for variant in ["se", "bam", "cbam"]:
    values = [
        float(r[gain_index])
        for r in data
        if r[variant_index] == variant
    ]

    mean_gain = sum(values) / len(values)

    print(
        f"{variant.upper():4s} | "
        f"{mean_gain*100:+.3f} percentage points"
    )

import json
import statistics

VARIANTS = ["none", "se", "bam", "cbam"]

clean_data = json.loads(
    open("evaluation_results/clean_results.json").read()
)

corrupt_data = json.loads(
    open("evaluation_results/corrupted_results.json").read()
)

def mean(values):
    return statistics.mean(values)

print("Attention gain over None")
print()

none_clean = mean([
    x["corruptions"]["clean"]["accuracy"]
    for x in clean_data
    if x["variant"] == "none"
])

none_corrupt = mean([
    x["corruptions"][c][str(s)]["accuracy"]
    for x in corrupt_data
    if x["variant"] == "none"
    for c in ["brightness", "contrast", "defocus_blur", "elastic_transform"]
    for s in range(1, 6)
])

for variant in ["se", "bam", "cbam"]:

    attention_clean = mean([
        x["corruptions"]["clean"]["accuracy"]
        for x in clean_data
        if x["variant"] == variant
    ])

    attention_corrupt = mean([
        x["corruptions"][c][str(s)]["accuracy"]
        for x in corrupt_data
        if x["variant"] == variant
        for c in ["brightness", "contrast", "defocus_blur", "elastic_transform"]
        for s in range(1, 6)
    ])

    clean_gain = attention_clean - none_clean
    corrupt_gain = attention_corrupt - none_corrupt

    print(
        f"{variant.upper():4s} | "
        f"clean gain = {clean_gain*100:+.3f} pp | "
        f"corruption gain = {corrupt_gain*100:+.3f} pp"
    )

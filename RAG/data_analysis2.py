import json
import pandas as pd
import numpy as np
#this script creates a barplot summary
# ---------------------------
# 1. Load data
# ---------------------------
with open("generation_eval_results_v2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# ---------------------------
# 2. Ensure engine exists
# ---------------------------
print("Total rows loaded:", len(df))
print("Engine distribution (raw):")
print(df["engine"].value_counts(dropna=False))
print()

# ---------------------------
# 3. Define metrics
# ---------------------------
metrics = [
    "faithfulness",
    "correctness",
    "answer_relevancy",
    "security_specificity",
    "context_utilization_score",
    "hallucinated_security_reasoning"
]

# ---------------------------
# 4. Force numeric conversion (NO DROPPING)
# ---------------------------
for m in metrics:
    df[m] = pd.to_numeric(df[m], errors="coerce")

# ---------------------------
# 5. Group stats (THIS is the important part)
# ---------------------------
grouped = df.groupby("engine")

summary = []

for engine, g in grouped:
    row = {"engine": engine, "n": len(g)}

    for m in metrics:
        values = g[m]

        row[m + "_mean"] = values.mean(skipna=True)
        row[m + "_std"] = values.std(skipna=True)
        row[m + "_var"] = values.var(skipna=True)

    summary.append(row)

summary_df = pd.DataFrame(summary)

# ---------------------------
# 6. Sort engines nicely (optional)
# ---------------------------
preferred_order = ["none", "bm25", "vector"]
summary_df["engine"] = pd.Categorical(summary_df["engine"], categories=preferred_order, ordered=True)
summary_df = summary_df.sort_values("engine")

# ---------------------------
# 7. Output
# ---------------------------
print("FINAL SUMMARY:")
print(summary_df.to_string(index=False))

print("\nSanity check:")
print("Total rows accounted for:", summary_df["n"].sum())

summary_df.to_csv("stat_results_summary.csv", index=False)
summary_df.to_json("stat_results_summary.json", orient="records", indent=2)

import matplotlib.pyplot as plt

metrics_base = [
    "faithfulness",
    "correctness",
    "answer_relevancy",
    "security_specificity",
    "context_utilization_score",
    "hallucinated_security_reasoning"
]

engines = summary_df["engine"].tolist()

for m in metrics_base:
    means = summary_df[m + "_mean"]
    stds = summary_df[m + "_std"]

    plt.figure(figsize=(6,4))
    plt.bar(engines, means, yerr=stds, capsize=5)
    plt.title(f"{m} (mean ± std)")
    plt.ylim(0, 2)
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(f"{m}_comparison.png", dpi=300)
    plt.close()
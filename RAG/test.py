import pandas as pd

# Load data
df = pd.read_csv("nlg_evaluation_results.csv")

# Metrics to summarize
metrics = [
    "rouge1",
    "rouge2",
    "rougeL",
    "bert_precision",
    "bert_recall",
    "bert_f1"
]

# Group by engine and compute statistics
summary = df.groupby("engine")[metrics].agg(["mean", "std", "var"])

# Flatten multi-index columns (makes CSV clean)
summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary.columns
]

summary = summary.reset_index()

# Save
summary.to_csv("nlg_summary_stats.csv", index=False)

print("Saved: nlg_summary_stats.csv")
print(summary)
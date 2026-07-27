import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open("generation_eval_results_v2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

metrics = [
    "faithfulness",
    "correctness",
    "answer_relevancy",
    "security_specificity",
    "context_utilization_score",
    "hallucinated_security_reasoning",
]

# Create row labels
df["row_label"] = df["query_id"] + " | " + df["engine"]

# Sort by query then engine
df = df.sort_values(["query_id", "engine"])

heatmap_df = df.set_index("row_label")[metrics]

# Plot
plt.figure(figsize=(10, max(6, len(heatmap_df) * 0.4)))

sns.heatmap(
    heatmap_df,
    annot=True,
    cmap="RdYlGn",
    vmin=0,
    vmax=2,
    cbar_kws={"ticks": [0, 1, 2]}
)

plt.title("Generation Evaluation Results")
plt.xlabel("Metric")
plt.ylabel("Query | Engine")

plt.tight_layout()
plt.show()
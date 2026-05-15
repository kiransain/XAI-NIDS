import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("ir_metrics_detailed.csv")

# Drop empty rows
df = df.dropna(subset=["precision@3", "recall@3", "hit@3", "mrr", "ndcg@3"])

# Create combined label
df["query_engine"] = df["query_id"] + " | " + df["engine"]

# Set index
df = df.set_index("query_engine")

metrics = ["precision@3", "recall@3", "hit@3", "mrr", "ndcg@3"]

plt.figure(figsize=(10, 12))
sns.heatmap(df[metrics], annot=True, cmap="viridis", linewidths=0.5)

plt.title("IR Metrics Heatmap")
plt.xlabel("Metrics")
plt.ylabel("Query | Engine")
plt.tight_layout()
plt.show()
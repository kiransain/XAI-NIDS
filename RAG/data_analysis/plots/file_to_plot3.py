import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
# violin plot code
script_dir = os.path.dirname(os.path.abspath(__file__))
rag_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
os.chdir(rag_root)


plt.style.use("seaborn-v0_8-white")
df = pd.read_csv("evaluation_results.csv")
df.columns = df.columns.str.strip()
sns.set_theme(style= "whitegrid", font="DejaVu Sans")

metrics = [
    "faithfulness",
    "correctness",
    "answer_relevancy",
    "security_specificity",
    "context_utilization_score",
    "hallucinated_security_reasoning",
]

titles = [
    "Faithfulness",
    "Correctness",
    "Answer Relevancy",
    "Security Specificity",
    "Context Utilization",
    "Hallucination Resistance"
]


strategy_order = ["none", "bm25", "vector"]
labels = ["None", "BM25", "Vector"]

fig, axes = plt.subplots(2, 3, figsize=(8, 4.5), sharey=True)
axes = axes.flatten()

for ax, metric, title in zip(axes, metrics, titles):

    sns.violinplot(
        data=df,
        x="engine",
        y=metric,
        order=strategy_order,
        ax=ax,
        color="lightgray",
        inner="quart",
        bw_adjust=0.6,
    )

    ax.set(
        xlabel="",
        ylabel="",
        ylim=(-0.5, 2.5),
        yticks=[0, 1, 2]
    )

    ax.set_title(title, fontsize=10)
    ax.set_xticklabels(labels)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Score")
axes[3].set_ylabel("Score")

plt.tight_layout()

# plt.savefig("paper_metrics_violin.png", dpi=300, bbox_inches="tight")

plt.show()
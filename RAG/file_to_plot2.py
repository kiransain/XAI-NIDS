import pandas as pd
import matplotlib.pyplot as plt
#plots bar charts for the summary statistics of the evaluation results

# Load summary statistics
summary_df = pd.read_csv("stat_results_summary.csv")

metrics = [
    "faithfulness",
    "correctness",
    "answer_relevancy",
    "security_specificity",
    "context_utilization_score",
    "hallucinated_security_reasoning"
]

titles = [
    "Faithfulness",
    "Correctness",
    "Answer Relevancy",
    "Security Specificity",
    "Context Utilization",
    "Hallucination (inv.)"
]

labels = ["None", "BM25", "Vector"]

fig, axes = plt.subplots(
    2, 3,
    figsize=(9, 5),
    sharey=True
)

axes = axes.flatten()

for ax, metric, title in zip(axes, metrics, titles):

    ax.bar(
        labels,
        summary_df[f"{metric}_mean"],
        yerr=summary_df[f"{metric}_std"],
        capsize=3,
        color="lightgray",
        edgecolor="black"
    )

    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 2.25)

# Only left column gets y-axis labels
axes[0].set_ylabel("Score")
axes[3].set_ylabel("Score")

plt.tight_layout()

plt.savefig(
    "llm_judge_metrics.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "llm_judge_metrics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
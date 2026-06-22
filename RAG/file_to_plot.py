import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# boxplot code

df = pd.read_csv("evaluation_results.csv")
df.columns = df.columns.str.strip()

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

fig, axes = plt.subplots(2, 3, figsize=(9, 5), sharey=True)
axes = axes.flatten()

for ax, metric, title in zip(axes, metrics, titles):

    sns.boxplot(
        data=df,
        x="engine",
        y=metric,
        ax=ax,
        order=["none", "bm25", "vector"],
        color="lightgray",
        linecolor="black",
        medianprops={"color": "black"}
        
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylim(-0.2, 2.2)
    ax.set_xticklabels(["None", "BM25", "Vector"])
    axes[0].set_ylabel("Score")
    axes[3].set_ylabel("Score")

plt.tight_layout()
plt.show()
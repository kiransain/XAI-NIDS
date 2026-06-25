
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations

df = pd.read_csv("evaluation_results.csv")

metrics = [
    "faithfulness",
    "correctness",
    "answer_relevancy",
    "security_specificity",
    "context_utilization_score",
    "hallucinated_security_reasoning"
]

metric_labels = {
    "faithfulness": "Faithfulness",
    "correctness": "Correctness",
    "answer_relevancy": "Answer Relevancy",
    "security_specificity": "Security Specificity",
    "context_utilization_score": "Context Utilization (CUS)",
    "hallucinated_security_reasoning": "Hallucinated Security Reasoning"
}

engines = ["none", "bm25", "vector"]
engine_labels = {"none": "No RAG", "bm25": "BM25", "vector": "Vector"}
colors = {"none": "#e06666", "bm25": "#6fa8dc", "vector": "#8e7cc3"}

# ── 1. Descriptive Statistics (Keeps all valid rows per engine group) ────────
print("=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)
rows = []
for metric in metrics:
    for engine in engines:
        vals = df[df["engine"] == engine][metric].dropna()
        rows.append({
            "metric": metric_labels[metric],
            "engine": engine_labels[engine],
            "n": len(vals),
            "mean": round(vals.mean(), 3),
            "median": round(vals.median(), 3),
            "std": round(vals.std(), 3),
            "min": vals.min(),
            "max": vals.max()
        })
desc = pd.DataFrame(rows)
print(desc.to_string(index=False))
desc.to_csv("descriptive_stats2.csv", index=False)
print("\nSaved: descriptive_stats2.csv\n")




# ── 2. Effect Sizes Logic (Pratt Matched Framework) ──────────────────────────
def rank_biserial_pratt(x, y):
    """Calculates rank-biserial correlation preserving ties via Pratt approach."""
    diffs = x - y
    if np.all(diffs == 0):
        return 0.0
    ranks = pd.Series(np.abs(diffs)).rank(method="average")
    
    pos_sum = ranks[diffs > 0].sum()
    neg_sum = ranks[diffs < 0].sum()
    total_sum = ranks.sum()
    
    r = (pos_sum - neg_sum) / total_sum
    return round(r, 3)


# ── 3. Aligned Hypothesis Testing (Friedman + Pratt-Wilcoxon) ────────────────
print("=" * 70)
print("FRIEDMAN TEST + PAIRWISE WILCOXON (Bonferroni corrected)")
print("=" * 70)

stat_rows = []
pairs = list(combinations(engines, 2))
n_comparisons = len(pairs)  # 3 comparisons per metric -> Alpha = 0.05/3 = 0.0167

for metric in metrics:
    # Crucial Pivot: Automatically discards queries missing any engine row
    pivoted = df.pivot(index="query_id", columns="engine", values=metric).dropna()
    
    g_none = pivoted["none"].values
    g_bm25 = pivoted["bm25"].values
    g_vector = pivoted["vector"].values
    
    # Run Friedman across aligned observations
    stat, p_friedman = friedmanchisquare(g_none, g_bm25, g_vector)
    print(f"\n{metric_labels[metric]} (N Perfectly Aligned Pairs = {len(pivoted)})")
    print(f"  Friedman: chi2={stat:.3f}, p={p_friedman:.4f}", "***" if p_friedman < 0.05 else "(n.s.)")

    groups_aligned = {"none": g_none, "bm25": g_bm25, "vector": g_vector}

    # Pairwise Tests
    for e1, e2 in pairs:
        v1 = groups_aligned[e1]
        v2 = groups_aligned[e2]
        diff = v1 - v2
        
        if np.all(diff == 0):
            print(f"  {engine_labels[e1]} vs {engine_labels[e2]}: all ties, skip")
            stat_rows.append({
                "metric": metric_labels[metric],
                "comparison": f"{engine_labels[e1]} vs {engine_labels[e2]}",
                "friedman_chi2": round(stat, 3),
                "friedman_p": round(p_friedman, 4),
                "wilcoxon_stat": None,
                "wilcoxon_p_raw": None,
                "wilcoxon_p_bonferroni": None,
                "significant_bonferroni": False,
                "effect_size_r": 0.0
            })
            continue
            
        # zero_method="pratt" preserves rank ties for evaluation values (0, 1, 2)
        w_stat, p_wilcoxon = wilcoxon(v1, v2, alternative="two-sided", zero_method="pratt")
        p_corrected = min(p_wilcoxon * n_comparisons, 1.0)
        sig = p_corrected < 0.05
        
        # Calculate matching effect size
        r_val = rank_biserial_pratt(v1, v2)
        
        print(f"  {engine_labels[e1]} vs {engine_labels[e2]}: W={w_stat:.1f}, "
              f"p_raw={p_wilcoxon:.4f}, p_bonf={p_corrected:.4f} | r = {r_val:+.3f}", "***" if sig else "")
              
        stat_rows.append({
            "metric": metric_labels[metric],
            "comparison": f"{engine_labels[e1]} vs {engine_labels[e2]}",
            "friedman_chi2": round(stat, 3),
            "friedman_p": round(p_friedman, 4),
            "wilcoxon_stat": round(w_stat, 1),
            "wilcoxon_p_raw": round(p_wilcoxon, 4),
            "wilcoxon_p_bonferroni": round(p_corrected, 4),
            "significant_bonferroni": sig,
            "effect_size_r": r_val
        })

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv("statistical_results2.csv", index=False)
print("\nSaved: statistical_results2.csv\n")


# ── 4. Box Plots (Reflects Raw Structural Ranges) ───────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Score Distributions by Retrieval Engine", fontsize=16, fontweight="bold", y=1.01)

for ax, metric in zip(axes.flatten(), metrics):
    data = [df[df["engine"] == e][metric].dropna().values for e in engines]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=[engine_labels[e] for e in engines],
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5)
    )
    for patch, engine in zip(bp["boxes"], engines):
        patch.set_facecolor(colors[engine])
        patch.set_alpha(0.75)

    ax.set_title(metric_labels[metric], fontsize=11, fontweight="bold")
    ax.set_ylim(-0.2, 2.4)
    ax.set_ylabel("Score (0–2)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("boxplots_by_engine2.png", dpi=300, bbox_inches="tight")
print("Saved: boxplots_by_engine2.png")


fig, ax = plt.subplots(figsize=(13, 7))

x = np.arange(len(metrics))
width = 0.25

for i, engine in enumerate(engines):
    means = [df[df["engine"] == engine][m].mean() for m in metrics]
    stds = [df[df["engine"] == engine][m].std() for m in metrics]
    
    offset = (i - 1) * width
    bars = ax.bar(x + offset, means, width, label=engine_labels[engine],
                  color=colors[engine], alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                color="black", capsize=4, linewidth=1.2)

ax.set_xticks(x)
ax.set_xticklabels([metric_labels[m].replace(" (CUS)", "\n(CUS)") for m in metrics],
                   fontsize=9, rotation=15, ha="right")
ax.set_ylabel("Mean Score (0–2)")
ax.set_ylim(0, 2.5)
ax.set_title("Mean Evaluation Scores by Retrieval Engine (± 1 STD)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("barplot_with_errorbars2.png", dpi=300, bbox_inches="tight")
print("Saved: barplot_with_errorbars2.png")

print("\nStatistical suite executed successfully with explicit structural grouping.")
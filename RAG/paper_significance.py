import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

from paper_holm import holm  

# ----------------------------
# CONFIG
# ----------------------------
JUDGE_CSV = "paper_judge_scores.csv"  
IR_CSV = "paper_ir_metrics_detailed.csv"        # TODO: your IR results file (query_id, engine, ..., ndcg@3)
ALPHA = 0.05
METRICS = [
    "faithfulness", "correctness", "answer_relevancy", "security_specificity",
    "context_utilization_score", "hallucinated_security_reasoning",
]
CONTEXT_METRICS = {"faithfulness", "context_utilization_score"}  # not defined without retrieval
PAIRS = [("bm25", "vector"), ("bm25", "none"), ("vector", "none")]


def rank_biserial(d):
    """Effect size for paired data: -1 .. +1, 0 = no effect."""
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < ALPHA else "ns"


j = pd.read_csv(JUDGE_CSV)
wide = {m: j.pivot(index="query_id", columns="engine", values=m) for m in METRICS}

# ---- 1. Paired Wilcoxon signed-rank test, Holm-corrected within each pair ----
rows = []
for a, b in PAIRS:
    pair_rows = []
    for m in METRICS:
        if b == "none" and m in CONTEXT_METRICS:
            continue
        d = (wide[m][a] - wide[m][b]).dropna().values
        p = 1.0 if (d == 0).all() else wilcoxon(d).pvalue  # no differences at all -> p = 1
        pair_rows.append({"comparison": f"{a} vs {b}", "metric": m, "n": len(d),
                          "mean_diff": d.mean(), "effect_size": rank_biserial(d), "p_raw": p})
    adj = holm([r["p_raw"] for r in pair_rows])
    for r, ap in zip(pair_rows, adj):
        r["p_holm"] = ap
        r["sig_holm"] = stars(ap)
    rows += pair_rows
wil = pd.DataFrame(rows).round(4)
wil.to_csv("significance_wilcoxon.csv", index=False)
print("=== Paired Wilcoxon (mean_diff = first minus second; effect_size = rank-biserial) ===")
print(wil.to_string(index=False))

# ---- 2. Spearman correlation: nDCG@3 vs judge score, per engine, Holm-corrected ----
ir = pd.read_csv(IR_CSV)
ir = ir[ir["engine"].isin(["bm25", "vector"])].pivot(index="query_id", columns="engine", values="ndcg@3")
rows = []
for e in ["bm25", "vector"]:
    for m in ["correctness", "faithfulness", "context_utilization_score"]:
        x, y = ir[e], wide[m][e]
        rho, p = spearmanr(x, y)
        rows.append({"engine": e, "judge_metric": m, "n": len(x), "spearman_rho": rho, "p_raw": p})
adj = holm([r["p_raw"] for r in rows])
for r, ap in zip(rows, adj):
    r["p_holm"] = ap
    r["sig_holm"] = stars(ap)
sp = pd.DataFrame(rows).round(4)
sp.to_csv("significance_spearman.csv", index=False)
print("\n=== Spearman correlation: nDCG@3 vs judge score ===")
print(sp.to_string(index=False))
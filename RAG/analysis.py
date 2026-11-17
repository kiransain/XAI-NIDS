import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

# ----------------------------
# CONFIG
# ----------------------------
JUDGE_CSV = "paper_judge_scores.csv"   # from judge_stats.py
IR_CSV = "paper_ir_metrics_detailed.csv"        # TODO: your IR results file (query_id, engine, ..., ndcg@3)
N_BOOT = 10000
SEED = 42

METRICS = [
    "faithfulness",
    "correctness",
    "answer_relevancy",
    "security_specificity",
    "context_utilization_score",
    "hallucinated_security_reasoning",
]
# Defined only with retrieved context, so not compared against the no-retrieval run.
CONTEXT_METRICS = {"faithfulness", "context_utilization_score"}

# Query categories. Everything not listed below counts as SHAP/LIME interpretation.
# TODO: check these two lists against your own judgement.
KB_QUERIES = [
    "Q_PFCP_retrieval_1", "Q_PFCP_retrieval_3", "Q_PFCP_retrieval_5",
    "Q_NIDD_ret_01", "Q_NIDD_ret_02", "Q_NIDD_ret_09",
    "Q_5GAD_ret_03", "Q_5GAD_ret_04", "Q_5GAD_ret_10", "Q_5GAD_ret_11", "Q_5GAD_ret_13",
    "Q_NIDD_use_05", "Q_NIDD_use_06", "Q_NIDD_use_07", "Q_NIDD_use_08", "Q_NIDD_use_12",
    "Q_NIDD_control_new1",
]
OTHER_QUERIES = [  # no answer in the KB by design (general knowledge / arithmetic)
    "Q_PFCP_control_1", "Q_PFCP_control_2", "Q_PFCP_control_3",
    "Q_NIDD_control_new2", "Q_NIDD_control_new3", "Q_NIDD_retadj_new",
    "Q_PFCP_usefulness_3",
]


def category(q):
    if q in KB_QUERIES:
        return "knowledge-base"
    if q in OTHER_QUERIES:
        return "other (no KB answer)"
    return "SHAP/LIME interpretation"


rng = np.random.default_rng(SEED)


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation: -1 .. +1, 0 = no effect."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    return (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()


def boot_ci(x):
    """95% bootstrap CI of the mean, resampling queries."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    means = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    j = pd.read_csv(JUDGE_CSV)
    wide = {m: j.pivot(index="query_id", columns="engine", values=m) for m in METRICS}
    queries = wide["correctness"].index
    cat = pd.Series({q: category(q) for q in queries})

    # ---- 1. paired tests, Holm-corrected within each comparison ----
    rows = []
    for a, b in [("bm25", "none"), ("vector", "none"), ("bm25", "vector")]:
        res = []
        for m in METRICS:
            if b == "none" and m in CONTEXT_METRICS:
                continue
            d = (wide[m][a] - wide[m][b]).dropna()
            p = 1.0 if (d == 0).all() else wilcoxon(d).pvalue  # no differences at all -> p = 1
            lo, hi = boot_ci(d.values)
            res.append({
                "comparison": f"{a} vs {b}", "metric": m, "n": len(d),
                "mean_diff": d.mean(), "ci95_low": lo, "ci95_high": hi,
                "effect_size_rank_biserial": rank_biserial(d.values),
                "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
                "p_raw": p,
            })
        for r, adj in zip(res, holm([r["p_raw"] for r in res])):
            r["p_holm"] = adj
        rows += res
    tests = pd.DataFrame(rows).round(4)
    tests.to_csv("stats_tests.csv", index=False)

    # ---- 2. results per query category ----
    mean_rows, diff_rows = [], []
    for c in sorted(cat.unique()):
        qs = cat[cat == c].index
        for m in METRICS:
            for e in ["none", "bm25", "vector"]:
                if e == "none" and m in CONTEXT_METRICS:
                    continue
                x = wide[m].loc[qs, e].dropna().values
                lo, hi = boot_ci(x)
                mean_rows.append({"category": c, "n_queries": len(x), "engine": e, "metric": m,
                                  "mean": x.mean(), "ci95_low": lo, "ci95_high": hi})
            for a, b in [("bm25", "none"), ("vector", "none"), ("bm25", "vector")]:
                if b == "none" and m in CONTEXT_METRICS:
                    continue
                d = (wide[m].loc[qs, a] - wide[m].loc[qs, b]).dropna().values
                lo, hi = boot_ci(d)
                diff_rows.append({"category": c, "n_queries": len(d), "comparison": f"{a} vs {b}",
                                  "metric": m, "mean_diff": d.mean(), "ci95_low": lo, "ci95_high": hi})
    pd.DataFrame(mean_rows).round(3).to_csv("category_means.csv", index=False)
    pd.DataFrame(diff_rows).round(3).to_csv("category_diffs.csv", index=False)

    # ---- 3. retrieval quality (nDCG@3) vs answer quality ----
    ir = pd.read_csv(IR_CSV)
    ir = ir[ir["engine"].isin(["bm25", "vector"])].pivot(index="query_id", columns="engine", values="ndcg@3")
    corr_rows = []
    for subset_name, qs in [("all queries", queries), ("excluding 'other'", cat[cat != "other (no KB answer)"].index)]:
        for e in ["bm25", "vector"]:
            for m in ["correctness", "faithfulness", "context_utilization_score"]:
                r, p = spearmanr(ir.loc[qs, e], wide[m].loc[qs, e])
                corr_rows.append({"subset": subset_name, "engine": e, "judge_metric": m,
                                  "n": len(qs), "spearman_rho": r, "p": p})
    pd.DataFrame(corr_rows).round(3).to_csv("ir_vs_quality.csv", index=False)

    # ---- 4. queries where vector has the higher nDCG but BM25 gives the better answer ----
    t = pd.DataFrame({
        "ndcg_bm25": ir["bm25"], "ndcg_vector": ir["vector"],
        "correctness_bm25": wide["correctness"]["bm25"], "correctness_vector": wide["correctness"]["vector"],
    }).dropna()
    t["category"] = cat
    disc = t[(t.ndcg_vector > t.ndcg_bm25) & (t.correctness_vector < t.correctness_bm25)].copy()
    if "notes" in j.columns:
        notes = j.set_index(["query_id", "engine"])["notes"]
        disc["judge_note_bm25"] = [notes.get((q, "bm25"), "") for q in disc.index]
        disc["judge_note_vector"] = [notes.get((q, "vector"), "") for q in disc.index]
    disc.round(3).to_csv("vector_higher_ndcg_but_worse_answer.csv")

    print("Saved: stats_tests.csv, category_means.csv, category_diffs.csv, ir_vs_quality.csv, "
          "vector_higher_ndcg_but_worse_answer.csv")
    print(f"Queries with higher nDCG for vector but better answer for BM25: {len(disc)}")
    print(tests[["comparison", "metric", "mean_diff", "p_raw", "p_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
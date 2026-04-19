"""
IR Metrics Evaluation ========================================
Computes classical information retrieval metrics for retrieval evaluation.

Metrics computed per query per engine:
  - Precision@K  : fraction of top-K retrieved that are relevant
  - Recall@K     : fraction of all relevant docs found in top-K
  - Hit@K        : 1 if any relevant doc in top-K, else 0
  - MRR          : mean reciprocal rank of first relevant doc
  - nDCG@K       : normalized discounted cumulative gain

Input:  thesis_evaluation_results.json  (from rag_pipeline.py)
        QRELS dict below — fill in manually after first pipeline run

Output: ir_metrics_detailed.csv   (per-query scores)
        ir_metrics_summary.csv    (mean per engine)

Requirements:
    pip install pandas
"""

import json
import math
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

RESULTS_FILE = "thesis_evaluation_results.json"
K = 3

# ── Relevance labels (QRELS) ───────────────────────────────────────────────────
QRELS = {
    # "Q1": ["chunk-of-doc1", "chunk-of-PFCP-doc2", etc.],
    "Q_PFCP_control_1":[],
    "Q_PFCP_control_2": [], 
    "Q_PFCP_control_3": [],
    "Q_PFCP_retrieval_1": [], # chp 4.2 3GPP TS 29.244
    # 
}

# ── Metric functions ───────────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list, relevant_ids: list, k: int = 3) -> float:
    """Fraction of top-K retrieved documents that are relevant."""
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int = 3) -> float:
    """Fraction of all relevant documents that appear in top-K."""
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant)
    return hits / len(relevant) if relevant else 0.0


def hit_at_k(retrieved_ids: list, relevant_ids: list, k: int = 3) -> float:
    """Binary: 1.0 if at least one relevant document is in top-K, else 0.0."""
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    return 1.0 if any(rid in relevant for rid in top_k) else 0.0


def mrr(retrieved_ids: list, relevant_ids: list) -> float:
    """Mean Reciprocal Rank: 1 / rank of first relevant document."""
    relevant = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list, relevant_ids: list, k: int = 3) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    relevant = set(relevant_ids)

    def dcg(ids: list) -> float:
        score = 0.0
        for i, rid in enumerate(ids[:k], start=1):
            rel = 1.0 if rid in relevant else 0.0
            score += rel / math.log2(i + 1)
        return score

    # Ideal ranking: all relevant docs at top positions
    ideal_ids = list(relevant_ids)[:k]
    idcg = dcg(ideal_ids)

    return dcg(retrieved_ids) / idcg if idcg > 0 else 0.0


# ── Load results ───────────────────────────────────────────────────────────────

print(f"Loading '{RESULTS_FILE}'...")
with open(RESULTS_FILE) as fh:
    results = json.load(fh)

print(f"Loaded {len(results)} entries.\n")

# ── Compute metrics per entry ──────────────────────────────────────────────────

rows = []

for entry in results:
    qid      = entry["query_id"]
    engine   = entry["engine"]
    retrieved = entry.get("retrieved_chunk_ids", [])
    relevant  = QRELS.get(qid, [])

    base = {"query_id": qid, "engine": engine}

    # No retrieval for 'none' engine — context metrics are N/A
    if engine == "none" or not retrieved:
        rows.append({
            **base,
            "precision@3": None,
            "recall@3":    None,
            "hit@3":       None,
            "mrr":         None,
            "ndcg@3":      None,
        })
        continue

    # if no chunks exist for query, all metrics = 0
    if not relevant:
        print(f"  WARNING: No QRELS defined for query '{qid}' — scores will be 0.")

    rows.append({
        **base,
        "precision@3": precision_at_k(retrieved, relevant, K),
        "recall@3":    recall_at_k(retrieved, relevant, K),
        "hit@3":       hit_at_k(retrieved, relevant, K),
        "mrr":         mrr(retrieved, relevant),
        "ndcg@3":      ndcg_at_k(retrieved, relevant, K),
    })


df = pd.DataFrame(rows)
# output (detailed)
print("Per-query scores:")
print(df.to_string(index=False))
df.to_csv("ir_metrics_detailed.csv", index=False)
print("\nDetailed scores saved to 'ir_metrics_detailed.csv'")

# Engine Summary ─────────────────────────────────────────────────

METRIC_COLS = ["precision@3", "recall@3", "hit@3", "mrr", "ndcg@3"]
summary = df.groupby("engine")[METRIC_COLS].mean().round(4)

print("\n" + "="*55)
print("SUMMARY — Mean IR Metrics per Retrieval Strategy")
print("="*55)
print(summary.to_string())

summary.to_csv("ir_metrics_summary.csv")
print("\nSummary saved to 'ir_metrics_summary.csv'")

print("""
── Expected pattern if retrieval improves ───────────────
  precision@3 : vector vs bm25  ← key comparison
  recall@3    : vector vs bm25  ← key comparison
  hit@3       : should be high for both (>0.5)
  mrr         : higher = relevant doc ranked earlier
  ndcg@3      : combines precision and ranking quality

Note: 'none' engine shows None for all retrieval metrics.
This is correct — there is no retrieval to evaluate.
""")

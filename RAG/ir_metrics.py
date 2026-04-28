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
    "Q_PFCP_control_1": [],
    "Q_PFCP_control_2": [],
    "Q_PFCP_control_3": [],
    "Q_PFCP_retrieval_1": [
        "c9080f8396ef6463438186a6b67ad8f3a0662e01ccd40c12180b9e6c42364d63",
        "5d6ec593de49447ac580b73a181ffafe6e76034ca6e63a1183ea276db332ebdf",
        "5e66761a14e76dca8b0c3e46fd894c2f744548460753a05585450437eb415542",
        "0a338daf1a4a0313bf7e8e8e5148d1b5f682890dde9d147cbbb6846c23ea787a"
    ], # chp 4.2 3GPP TS 29.244
    "Q_PFCP_retrieval_3":[],
    "Q_PFCP_retrieval_5": [
        "f0c1cedb2da76996194b878266c9c4161d3b52155c47b6342e514fea66f56654"
    ],
    "Q_PFCP_faithfulness_1": [
        "fc9bdf3d84a5f0676afdf4208b43ab9504f63a97b61039bab4dcc93994e7b688"
    ],
    "Q_PFCP_faithfulness_2": [],
    "Q_PFCP_faithfulness_3": [
        "fc9bdf3d84a5f0676afdf4208b43ab9504f63a97b61039bab4dcc93994e7b688"
    ],
    "Q_PFCP_faithfulness_4":[],
    "Q_NIDD_faith_1":[],
    "Q_NIDD_faith_2":[],
    "Q_NIDD_faith_3":[],
    "Q_PFCP_usefulness_1": [],
    "Q_PFCP_usefulness_2": [],
    "Q_PFCP_usefulness_3": [],
    "Q_PFCP_usefulness_4": [],
    "Q_PFCP_usefulness_5": [
        "7a82e14fb747795c4a1e09cba1c7c3cda4269bb7efbc2031d58d0cd5697b875b",
        "905eb06d31565e18d4729b239b3590f23d118303dde85faecbf044065ba833ba",
        "9293e1a8d508de144d45a7cd48386df2199ab351053c750e7d91e5a3761767eb",
        "d68755988e51274c3361c4cb59bfd7a5aa644074a6f80fa69b3e163a4d5b032a",
        "ded47291676490160ecd444d26f4e483cc062718453347326f3dfeea8e4099bd",
        "b06bb89b0d11d600522d2b49a74fe449fc3667e7e530960a2a5d243143e9c1df",
        "097f09c10130e1c98796805875cfcc17ace87d5d6b32274c53d53f6df86abe3e"
    ],
    "Q_PFCP_usefulness_6": [
        "fc9bdf3d84a5f0676afdf4208b43ab9504f63a97b61039bab4dcc93994e7b688"
    ],
    "Q_PFCP_usefulness_7": [
        "fc9bdf3d84a5f0676afdf4208b43ab9504f63a97b61039bab4dcc93994e7b688",
        "d68755988e51274c3361c4cb59bfd7a5aa644074a6f80fa69b3e163a4d5b032a",
        "097f09c10130e1c98796805875cfcc17ace87d5d6b32274c53d53f6df86abe3e"
    ],
    "Q_PFCP_usefulness_8": [
        "d68755988e51274c3361c4cb59bfd7a5aa644074a6f80fa69b3e163a4d5b032a",
        "097f09c10130e1c98796805875cfcc17ace87d5d6b32274c53d53f6df86abe3e",
        "fc9bdf3d84a5f0676afdf4208b43ab9504f63a97b61039bab4dcc93994e7b688"
    ],
    "Q_PFCP_usefulness_9": [
        "fc9bdf3d84a5f0676afdf4208b43ab9504f63a97b61039bab4dcc93994e7b688"
    ],
    "Q_5GAD_usefulness_1": [
        "ded47291676490160ecd444d26f4e483cc062718453347326f3dfeea8e4099bd",
        "80a6fb787e4c8215c07f43e7b533ee9ef0eb67e0471066c7a7797e5523396809",
        "9293e1a8d508de144d45a7cd48386df2199ab351053c750e7d91e5a3761767eb",
        "c9080f8396ef6463438186a6b67ad8f3a0662e01ccd40c12180b9e6c42364d63",
        "fc9bdf3d84a5f0676afdf4208b43ab9504f63a97b61039bab4dcc93994e7b688"
    ],
    "Q_5GAD_usefulness_2": [
        "c9080f8396ef6463438186a6b67ad8f3a0662e01ccd40c12180b9e6c42364d63",
        "3ef121622617ee80674df772a61153dfe048d2a9246e239820d1740f1fae1266",
        "80a6fb787e4c8215c07f43e7b533ee9ef0eb67e0471066c7a7797e5523396809",
        "fc9bdf3d84a5f0676afdf4208b43ab9504f63a97b61039bab4dcc93994e7b688"
    ],
    "Q_5GAD_usefulness_3": [],
    "Q_NIDD_use_1": [
        "3ef121622617ee80674df772a61153dfe048d2a9246e239820d1740f1fae1266",
        "0a338daf1a4a0313bf7e8e8e5148d1b5f682890dde9d147cbbb6846c23ea787a",
        "fa3f2f145d05f188d48ea5155fcd400aa825d8d65fcc15f1c6dd64a021f430b9",
        "73c46dcb54d0707905f5ef5829b35491bfa11102193dd5972018b688973b0613"
    ],
    "Q_NIDD_use_2": [
        "fa3f2f145d05f188d48ea5155fcd400aa825d8d65fcc15f1c6dd64a021f430b9",
        "7cb68838492a84200bb164206c258c6a81b6dc92f42550af4d222b0f1bd5c7ba",
        "6867404c74e8effb6d55265d43afef38f4e2a6a143ca3f97f11ffea0115f41f0"
    ]

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

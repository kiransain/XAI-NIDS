"""
RAGAS Evaluation Script — XAI-RAG Thesis
==========================================
Evaluates three retrieval configurations:
  - none  : LLM only, no retrieval
  - bm25  : BM25 lexical retrieval
  - vector: Dense vector retrieval

Metrics:
  - Faithfulness       (all engines)
  - Answer Relevancy   (all engines)
  - Context Precision  (bm25, vector only)
  - Context Recall     (bm25, vector only)

Input:  thesis_evaluation_results.json
Output: ragas_scores_detailed.csv  (per-query scores)
        ragas_scores_summary.csv   (mean per engine)

Requirements:
    pip install ragas langchain-community langchain-ollama ollama datasets pandas
"""

import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

# ── Configuration ──────────────────────────────────────────────────────────────

RESULTS_FILE     = "thesis_evaluation_results.json"
OUTPUT_DETAILED  = "ragas_scores_detailed.csv"
OUTPUT_SUMMARY   = "ragas_scores_summary.csv"
OLLAMA_MODEL     = "llama3.2:1b" #deepseek-r1:8b
EMBED_MODEL      = "nomic-embed-text"

# ── Judge LLM + Embeddings ─────────────────────────────────────────────────────

print("Initialising Ollama judge...")
evaluator_llm        = ChatOllama(model=OLLAMA_MODEL)
evaluator_embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# ── Load results ───────────────────────────────────────────────────────────────

print(f"Loading '{RESULTS_FILE}'...")
with open(RESULTS_FILE, "r") as f:
    results_data = json.load(f)

# ── Build evaluation dataframe ─────────────────────────────────────────────────

rows = []
for entry in results_data:
    raw_context = entry.get("retrieved_context", "") or ""

    # Split joined chunks back into a list for RAGAS
    # (chunks were joined with \n\n in the pipeline)
    contexts = [c.strip() for c in raw_context.split("\n\n") if c.strip()]
    if not contexts:
        contexts = [""]  # RAGAS requires at least one context string

    rows.append({
        "question":     entry["query"],
        "answer":       entry["answer"],
        "contexts":     contexts,
        "ground_truth": entry["ground_truth"],
        "engine":       entry["engine"],
        "query_id":     entry.get("query_id", ""),
    })

df = pd.DataFrame(rows)

# ── Evaluate per engine ────────────────────────────────────────────────────────

ENGINES = ["none", "bm25", "vector"]
all_results = {}

for engine in ENGINES:
    print(f"\n{'='*60}")
    print(f"Engine: {engine.upper()}")
    print(f"{'='*60}")

    engine_df = df[df["engine"] == engine].copy()

    if engine_df.empty:
        print(f"  No data found for engine '{engine}', skipping.")
        continue

    print(f"  Queries: {len(engine_df)}")

    # Context metrics require actual retrieved chunks
    # Skip them for the no-RAG baseline
    if engine == "none":
        active_metrics = [faithfulness, answer_relevancy]
        print("  Metrics: faithfulness, answer_relevancy")
        print("  (context metrics skipped — no retrieval in baseline)")
    else:
        active_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        print("  Metrics: faithfulness, answer_relevancy, context_precision, context_recall")

    # RAGAS needs these exact columns, drop extras before converting
    ragas_df = engine_df[["question", "answer", "contexts", "ground_truth"]]
    dataset  = Dataset.from_pandas(ragas_df)

    try:
        result    = evaluate(
            dataset,
            metrics=active_metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        scores_df = result.to_pandas()

        # Re-attach metadata columns
        scores_df["engine"]   = engine
        scores_df["query_id"] = engine_df["query_id"].values

        all_results[engine] = scores_df

        # Print mean scores
        metric_cols = [
            c for c in scores_df.columns
            if c not in ("question", "answer", "contexts", "ground_truth",
                         "engine", "query_id")
        ]
        print(f"\n  {'Metric':<25} {'Mean':>8}")
        print(f"  {'-'*35}")
        for col in metric_cols:
            try:
                mean_val = pd.to_numeric(scores_df[col], errors='coerce').mean()
                print(f"  {col:<25} {mean_val:>8.4f}")
            except Exception:
                print(f"  {col:<25} {'ERROR':>8}")

    except Exception as e:
        print(f"  ERROR: {e}")

# ── Save detailed per-query CSV ────────────────────────────────────────────────

if all_results:
    detailed_df = pd.concat(all_results.values(), ignore_index=True)
    detailed_df.to_csv(OUTPUT_DETAILED, index=False)
    print(f"\nPer-query scores saved to '{OUTPUT_DETAILED}'")
else:
    print("\nNo results to save.")

# ── Summary table ──────────────────────────────────────────────────────────────

METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

summary_rows = []
for engine, scores_df in all_results.items():
    row = {"engine": engine}
    for col in METRIC_COLS:
        if col in scores_df.columns:
            row[col] = round(pd.to_numeric(scores_df[col], errors='coerce').mean(), 4)
        else:
            row[col] = None
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("engine")

print("\n" + "="*60)
print("SUMMARY — Mean RAGAS Scores per Retrieval Strategy")
print("="*60)
print(summary_df.to_string())
print()

summary_df.to_csv(OUTPUT_SUMMARY)
print(f"Summary saved to '{OUTPUT_SUMMARY}'")

# ── Expected pattern reminder ──────────────────────────────────────────────────

print("""
── Expected pattern if RAG improves explanations ────────────
  faithfulness:      vector >= bm25 > none
  answer_relevancy:  vector >= bm25 > none
  context_precision: vector vs bm25  ← key thesis finding
  context_recall:    vector vs bm25  ← key thesis finding

If scores are all 0 or NaN:
  1. Run: ollama serve
  2. Run: ollama list  (check both models are pulled)
  3. Check retrieved_context is non-empty in your JSON
""")
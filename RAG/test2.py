"""
RAGAS Evaluation Script for XAI-RAG Thesis
============================================
Evaluates three retrieval configurations:
  - none  : LLM only (no RAG)
  - bm25  : BM25 lexical retrieval
  - vector: Dense vector retrieval

Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
Judge LLM: Local Ollama (deepseek-r1:8b)
Embeddings: nomic-embed-text via Ollama

Input:  thesis_evaluation_results.json  (produced by main RAG pipeline)
Output: ragas_scores.json + ragas_scores_summary.csv
"""

import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_FILE   = "thesis_evaluation_results.json"   # output from main pipeline
OUTPUT_JSON    = "ragas_scores.json"
OUTPUT_CSV     = "ragas_scores_summary.csv"

OLLAMA_MODEL   = "deepseek-r1:8b"   # judge LLM (same model, kept constant) #deepseek-r1:8b
EMBED_MODEL    = "nomic-embed-text"  # embedding model for Answer Relevancy

# ── Load Ollama judge LLM + embeddings ────────────────────────────────────────

print("Loading Ollama judge LLM and embeddings...")

judge_llm = LangchainLLMWrapper(
    ChatOllama(model=OLLAMA_MODEL, timeout=600)
)

judge_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model=EMBED_MODEL)
)

# ── Instantiate RAGAS metrics ──────────────────────────────────────────────────

metrics = [
    Faithfulness(llm=judge_llm),
    AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
    ContextPrecision(llm=judge_llm),
    ContextRecall(llm=judge_llm),
]

# ── Load pipeline results ──────────────────────────────────────────────────────

print(f"Loading results from '{RESULTS_FILE}'...")

with open(RESULTS_FILE, "r") as f:
    raw_results = json.load(f)

# ── Helper: build RAGAS dataset for one engine mode ───────────────────────────

def build_ragas_dataset(results: list, engine: str) -> Dataset:
    """
    Filter results by engine type and convert to RAGAS Dataset format.

    RAGAS expects these columns:
      - user_input      : the query sent to the LLM
      - response        : the LLM-generated answer
      - retrieved_contexts: list of retrieved chunks (list of strings)
      - reference       : ground truth answer
    """
    subset = [r for r in results if r["engine"] == engine]

    if not subset:
        raise ValueError(f"No results found for engine='{engine}'")

    data = {
        "user_input":          [r["query"]           for r in subset],
        "response":            [r["answer"]           for r in subset],
        "retrieved_contexts":  [
            # retrieved_context is a single string in the pipeline output;
            # RAGAS expects a list of strings (one per retrieved chunk).
            # Split on double-newline which is how chunks are joined.
            r["retrieved_context"].split("\n\n") if r["retrieved_context"] else [""]
            for r in subset
        ],
        "reference":           [r["ground_truth"]     for r in subset],
    }

    return Dataset.from_dict(data)


# ── Run evaluation per engine ──────────────────────────────────────────────────

ENGINE_MODES = ["none", "bm25", "vector"]
all_scores   = {}

for engine in ENGINE_MODES:
    print(f"\n{'='*60}")
    print(f"Evaluating engine: '{engine}'")
    print(f"{'='*60}")

    try:
        dataset = build_ragas_dataset(raw_results, engine)
        print(f"  Queries loaded: {len(dataset)}")

        # Note: ContextPrecision + ContextRecall are skipped for 'none' mode
        # because there are no retrieved contexts to evaluate.
        if engine == "none":
            active_metrics = [
                Faithfulness(llm=judge_llm),
                AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
            ]
            print("  (Skipping context metrics for no-RAG baseline)")
        else:
            active_metrics = metrics

        result = evaluate(
            dataset=dataset,
            metrics=active_metrics,
        )

        scores_df = result.to_pandas()
        all_scores[engine] = scores_df.to_dict(orient="records")

        print(f"\n  Results for '{engine}':")
        print(f"  {'Metric':<30} {'Mean Score':>10}")
        print(f"  {'-'*42}")
        for col in scores_df.columns:
            if col not in ("user_input", "response", "retrieved_contexts", "reference"):
                print(f"  {col:<30} {scores_df[col].mean():>10.4f}")

    except Exception as e:
        print(f"  ERROR evaluating '{engine}': {e}")
        all_scores[engine] = {"error": str(e)}


# ── Save per-query scores to JSON ─────────────────────────────────────────────

with open(OUTPUT_JSON, "w") as f:
    json.dump(all_scores, f, indent=4)

print(f"\nPer-query scores saved to '{OUTPUT_JSON}'")


# ── Build summary comparison table ────────────────────────────────────────────

METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

summary_rows = []

for engine in ENGINE_MODES:
    if "error" in all_scores.get(engine, {}):
        continue

    records = all_scores[engine]
    if not records:
        continue

    df = pd.DataFrame(records)
    row = {"engine": engine}

    for col in METRIC_COLS:
        if col in df.columns:
            row[col] = round(df[col].mean(), 4)
        else:
            row[col] = None   # not applicable (e.g. context metrics for 'none')

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("engine")

print("\n" + "="*60)
print("SUMMARY: Mean RAGAS Scores per Retrieval Strategy")
print("="*60)
print(summary_df.to_string())

summary_df.to_csv(OUTPUT_CSV)
print(f"\nSummary table saved to '{OUTPUT_CSV}'")


# ── Quick sanity check ────────────────────────────────────────────────────────

print("\n── Sanity Check ──────────────────────────────────────────")
print("Expected pattern if RAG helps:")
print("  faithfulness:       vector >= bm25 > none")
print("  answer_relevancy:   vector >= bm25 > none")
print("  context_precision:  vector vs bm25 (key comparison)")
print("  context_recall:     vector vs bm25 (key comparison)")
print("")
print("If scores are unexpectedly uniform or all ~0, check:")
print("  1. Ollama is running:  `ollama serve`")
print("  2. Model is pulled:    `ollama pull deepseek-r1:8b`")
print("  3. Embed model pulled: `ollama pull nomic-embed-text`")
print("  4. retrieved_context in JSON is not empty for bm25/vector rows")
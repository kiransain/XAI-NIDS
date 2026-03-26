"""
Manual RAG Evaluation Script — XAI-RAG Thesis
===============================================
Uses DeepSeek-R1:8b as judge directly via ollama (no RAGAS framework).
Evaluates three retrieval configurations: none, bm25, vector.

Metrics scored 0.0-1.0:
  - Faithfulness       : Is the answer grounded in the retrieved context?
  - Answer Relevancy   : Does the answer address the query?
  - Context Precision  : Is the retrieved context relevant to the query?
  - Context Recall     : Does the retrieved context cover the ground truth?

Input:  thesis_evaluation_results.json
Output: manual_scores_detailed.csv
        manual_scores_summary.csv

Requirements:
    pip install ollama pandas
"""

import json
import re
import time
import pandas as pd
import ollama

# ── Configuration ──────────────────────────────────────────────────────────────

RESULTS_FILE    = "thesis_evaluation_results.json"
OUTPUT_DETAILED = "manual_scores_detailed.csv"
OUTPUT_SUMMARY  = "manual_scores_summary.csv"
JUDGE_MODEL     = "deepseek-r1:8b"
ENGINES         = ["none", "bm25", "vector"]

# ── Judge helpers ──────────────────────────────────────────────────────────────

def ask_judge(prompt: str, retries: int = 3) -> str:
    """Call local DeepSeek judge and return raw text response."""
    for attempt in range(retries):
        try:
            response = ollama.chat(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_predict": 100},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"    Attempt {attempt+1}/{retries} failed: {e}")
            time.sleep(5)
    return "ERROR"


def parse_score(raw: str) -> float:
    """Extract a 0.0-1.0 float from judge response."""
    # Strip DeepSeek <think>...</think> tags if present
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    matches = re.findall(r"\b(1\.0+|0\.\d+|[01])\b", raw)
    if matches:
        try:
            return max(0.0, min(1.0, float(matches[0])))
        except ValueError:
            pass
    print(f"    Could not parse score from: {raw[:120]}")
    return float("nan")


# ── Scoring functions ──────────────────────────────────────────────────────────

def score_faithfulness(answer: str, context: str) -> float:
    """Is every claim in the answer supported by the retrieved context?"""
    prompt = f"""You are an expert judge evaluating a RAG system for 5G network security.

Task: Score FAITHFULNESS — are the claims in the ANSWER supported by the CONTEXT?
- 1.0 = every claim is grounded in the context
- 0.5 = some claims are grounded, some are not
- 0.0 = the answer ignores or contradicts the context

CONTEXT:
{context[:1500]}

ANSWER:
{answer[:1000]}

Reply with ONLY a single decimal number between 0.0 and 1.0. Nothing else."""
    return parse_score(ask_judge(prompt))


def score_answer_relevancy(query: str, answer: str) -> float:
    """Does the answer directly address the question?"""
    prompt = f"""You are an expert judge evaluating a RAG system for 5G network security.

Task: Score ANSWER RELEVANCY — does the ANSWER directly address the QUESTION?
- 1.0 = fully addresses the question
- 0.5 = partially addresses the question
- 0.0 = off-topic or does not address the question

QUESTION:
{query}

ANSWER:
{answer[:1000]}

Reply with ONLY a single decimal number between 0.0 and 1.0. Nothing else."""
    return parse_score(ask_judge(prompt))


def score_context_precision(query: str, context: str) -> float:
    """Is the retrieved context relevant and useful for answering the query?"""
    prompt = f"""You are an expert judge evaluating a RAG system for 5G network security.

Task: Score CONTEXT PRECISION — is the RETRIEVED CONTEXT relevant and useful for answering the QUESTION?
- 1.0 = highly relevant, directly useful
- 0.5 = partially relevant
- 0.0 = irrelevant or not useful

QUESTION:
{query}

RETRIEVED CONTEXT:
{context[:1500]}

Reply with ONLY a single decimal number between 0.0 and 1.0. Nothing else."""
    return parse_score(ask_judge(prompt))


def score_context_recall(context: str, ground_truth: str) -> float:
    """Does the retrieved context contain the information needed to reach the ground truth?"""
    prompt = f"""You are an expert judge evaluating a RAG system for 5G network security.

Task: Score CONTEXT RECALL — does the RETRIEVED CONTEXT contain the information needed to produce the GROUND TRUTH answer?
- 1.0 = context fully covers what is needed
- 0.5 = context partially covers what is needed
- 0.0 = context is missing the key information

GROUND TRUTH ANSWER:
{ground_truth[:800]}

RETRIEVED CONTEXT:
{context[:1500]}

Reply with ONLY a single decimal number between 0.0 and 1.0. Nothing else."""
    return parse_score(ask_judge(prompt))


# ── Load results ───────────────────────────────────────────────────────────────

print(f"Loading '{RESULTS_FILE}'...")
with open(RESULTS_FILE, "r") as f:
    results_data = json.load(f)

print(f"Loaded {len(results_data)} entries.\n")

# ── Score each entry ───────────────────────────────────────────────────────────

scored_rows = []

for entry in results_data:
    engine      = entry["engine"]
    query_id    = entry.get("query_id", "?")
    query       = entry["query"]
    answer      = entry["answer"]
    ground_truth = entry["ground_truth"]
    raw_context = entry.get("retrieved_context", "") or ""

    print(f"Scoring {query_id} | engine={engine}")

    row = {
        "query_id": query_id,
        "engine":   engine,
        "query":    query[:80] + "...",
    }

    # Faithfulness — meaningful for bm25/vector; for 'none' context is empty
    if engine == "none":
        row["faithfulness"]      = float("nan")  # no context to be faithful to
        row["context_precision"] = float("nan")
        row["context_recall"]    = float("nan")
        print(f"  faithfulness     : N/A (no retrieval)")
        print(f"  context_precision: N/A (no retrieval)")
        print(f"  context_recall   : N/A (no retrieval)")
    else:
        print(f"  faithfulness     : ", end="", flush=True)
        row["faithfulness"] = score_faithfulness(answer, raw_context)
        print(row["faithfulness"])

        print(f"  context_precision: ", end="", flush=True)
        row["context_precision"] = score_context_precision(query, raw_context)
        print(row["context_precision"])

        print(f"  context_recall   : ", end="", flush=True)
        row["context_recall"] = score_context_recall(raw_context, ground_truth)
        print(row["context_recall"])

    # Answer relevancy — meaningful for all engines
    print(f"  answer_relevancy : ", end="", flush=True)
    row["answer_relevancy"] = score_answer_relevancy(query, answer)
    print(row["answer_relevancy"])

    scored_rows.append(row)
    print()

# ── Save detailed results ──────────────────────────────────────────────────────

detailed_df = pd.DataFrame(scored_rows)
detailed_df.to_csv(OUTPUT_DETAILED, index=False)
print(f"Per-query scores saved to '{OUTPUT_DETAILED}'")

# ── Summary table ──────────────────────────────────────────────────────────────

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

summary_rows = []
for engine in ENGINES:
    engine_df = detailed_df[detailed_df["engine"] == engine]
    if engine_df.empty:
        continue
    row = {"engine": engine}
    for col in METRIC_COLS:
        if col in engine_df.columns:
            row[col] = round(pd.to_numeric(engine_df[col], errors="coerce").mean(), 4)
        else:
            row[col] = None
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("engine")

print("\n" + "="*60)
print("SUMMARY — Mean Scores per Retrieval Strategy")
print("="*60)
print(summary_df.to_string())
print()

summary_df.to_csv(OUTPUT_SUMMARY)
print(f"Summary saved to '{OUTPUT_SUMMARY}'")

print("""
── Expected pattern if RAG improves explanations ────────────
  faithfulness:      vector >= bm25 > none (N/A)
  answer_relevancy:  vector >= bm25 > none
  context_precision: vector vs bm25  ← key thesis finding
  context_recall:    vector vs bm25  ← key thesis finding
""")
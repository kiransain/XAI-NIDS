import json

import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
INPUT_FILE = "paper_results_llm_as_judge.json"
SCORES_CSV = "paper_judge_scores.csv"
STATS_CSV = "paper_judge_stats.csv"
EXPECTED_QUERIES = 53
ENGINES = ["none", "bm25", "vector"]
METRICS = [
    "faithfulness",
    "correctness",
    "answer_relevancy",
    "security_specificity",
    "context_utilization_score",
    "hallucinated_security_reasoning",
]


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Accept a plain list or the {"meta": ..., "judgments": [...]} format of judge.py
    return data["judgments"] if isinstance(data, dict) else data


def main():
    df = pd.DataFrame(load(INPUT_FILE))
    if "notes" not in df.columns:
        df["notes"] = ""
    df["notes"] = df["notes"].fillna("")
    df = df[["query_id", "engine"] + METRICS + ["notes"]]
    print(f"Entries in file: {len(df)}")

    # --- duplicates: keep the first entry per (query_id, engine), ignore the rest ---
    dup_all = df[df.duplicated(["query_id", "engine"], keep=False)]
    if len(dup_all):
        differing = (
            dup_all.groupby(["query_id", "engine"])[METRICS].nunique().max(axis=1) > 1
        )
        print(f"Duplicate (query_id, engine) entries ignored: {df.duplicated(['query_id', 'engine']).sum()}")
        if differing.any():
            print("  WARNING: duplicates with different scores for:")
            for qid, eng in differing[differing].index:
                print(f"    {qid} / {eng}")
    df = df.drop_duplicates(["query_id", "engine"], keep="first").reset_index(drop=True)

    # --- completeness check ---
    n_queries = df["query_id"].nunique()
    print(f"Unique query_ids: {n_queries} (expected {EXPECTED_QUERIES})")
    present = set(zip(df["query_id"], df["engine"]))
    missing = [(q, e) for q in df["query_id"].unique() for e in ENGINES if (q, e) not in present]
    print(f"Rows after de-duplication: {len(df)} (expected {EXPECTED_QUERIES * len(ENGINES)})")
    if missing:
        print(f"Missing (query_id, engine) combinations among the query_ids found: {len(missing)}")
        for q, e in missing:
            print(f"  {q} / {e}")
    other = sorted(set(df["engine"]) - set(ENGINES))
    if other:
        print(f"WARNING: unexpected engine names: {other}")

    # --- CSV 1: all judgments ---
    df.to_csv(SCORES_CSV, index=False)

    # --- CSV 2: statistics per engine and metric ---
    rows = []
    for engine in [e for e in ENGINES if e in set(df["engine"])] + other:
        sub = df[df["engine"] == engine]
        for metric in METRICS:
            s = sub[metric].astype(float)
            rows.append(
                {
                    "engine": engine,
                    "metric": metric,
                    "n": int(s.count()),
                    "mean": s.mean(),
                    "std": s.std(),
                    "min": s.min(),
                    "q1": s.quantile(0.25),
                    "median": s.median(),
                    "q3": s.quantile(0.75),
                    "max": s.max(),
                }
            )
    stats = pd.DataFrame(rows).round(3)
    stats.to_csv(STATS_CSV, index=False)
    print(f"Saved {SCORES_CSV} and {STATS_CSV}")


if __name__ == "__main__":
    main()
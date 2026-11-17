import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
RAW_JSON = "paper_raw_results.json"   # query, answer, ground_truth per query_id/engine
JUDGE_CSV = "paper_judge_scores.csv"        # from judge_stats.py
SHEET_CSV = "human_eval_sheet.csv"    # send this to Nasim
KEY_CSV = "human_eval_key.csv"        # keep this private (engine + judge scores)
RUBRIC_TXT = "human_eval_rubric.txt"  # send this to Nasim
META_JSON = "human_eval_meta.json"    # seed, settings and input hashes (for the paper / reproducibility)
SEED = 42

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


# Number of answers per (category, engine) cell: 20 answers, engines 7 / 7 / 6.
QUOTA = {
    ("knowledge-base", "none"): 2, ("knowledge-base", "bm25"): 3, ("knowledge-base", "vector"): 2,
    ("SHAP/LIME interpretation", "none"): 3, ("SHAP/LIME interpretation", "bm25"): 3,
    ("SHAP/LIME interpretation", "vector"): 3,
    ("other (no KB answer)", "none"): 2, ("other (no KB answer)", "bm25"): 1,
    ("other (no KB answer)", "vector"): 1,
}
MIN_PER_LEVEL = 3  # at least this many answers with judge correctness 0, 1 and 2

RUBRIC = """HUMAN EVALUATION RUBRIC (same 0-2 scale for every criterion)

You see a question, a reference answer and one answer. Score each criterion separately.
When unsure between two scores, choose the lower one if the mistake could mislead a 5G network operator.

1. CORRECTNESS (0-2)
Does the answer capture the main security/technical conclusion and key reasoning of the reference answer?
2 = captures the main conclusion and key reasoning
1 = partly captures the conclusion, or misses important implications or supporting reasoning
0 = incorrect, misleading, or contradicts the reference answer

2. FACTUAL GROUNDING (0-2)
Are the factual claims in the answer accurate, and free of invented facts, numbers or mechanisms?
(Judge the claims against the question, the reference answer and your own knowledge.)
2 = no significant unsupported or invented claims
1 = some speculative or unsupported claims that do not change the main conclusion
0 = multiple invented or false claims

3. SECURITY-SPECIFIC REASONING (0-2)
Does the answer give reasoning specific to the scenario (protocol, attack, traffic pattern),
or only generic ML/XAI commentary?
2 = concrete attack- or protocol-specific interpretation
1 = mix of specific and generic reasoning
0 = mostly generic ML/XAI explanation

Fill in the columns correctness_0_2, factual_grounding_0_2 and security_reasoning_0_2 in the sheet.
Use the comments column for anything unclear.
"""


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def draw(rng, pool_by_cell):
    chosen, used = [], set()
    cells = list(QUOTA)
    rng.shuffle(cells)
    for cell in cells:
        pool = [q for q in pool_by_cell[cell] if q not in used]
        if len(pool) < QUOTA[cell]:
            raise RuntimeError(f"not enough distinct queries for {cell}")
        take = [str(q) for q in rng.choice(pool, size=QUOTA[cell], replace=False)]
        used.update(take)
        chosen += [(q, cell[1]) for q in take]
    return chosen


def main():
    with open(RAW_JSON, encoding="utf-8") as f:
        raw = pd.DataFrame(json.load(f))[["query_id", "engine", "query", "answer", "ground_truth"]]
    judge = pd.read_csv(JUDGE_CSV).drop(columns=["notes"], errors="ignore")
    df = raw.merge(judge, on=["query_id", "engine"])
    df["category"] = df["query_id"].map(category)

    pool_by_cell = {
        cell: sorted(df[(df.category == cell[0]) & (df.engine == cell[1])]["query_id"])
        for cell in QUOTA
    }
    rng = np.random.default_rng(SEED)
    for attempt in range(1000):
        chosen = draw(rng, pool_by_cell)
        sel = pd.DataFrame(chosen, columns=["query_id", "engine"]).merge(df, on=["query_id", "engine"])
        levels = sel["correctness"].value_counts()
        if all(levels.get(k, 0) >= MIN_PER_LEVEL for k in (0, 1, 2)):
            break
    else:
        raise RuntimeError("could not find a sample with enough spread in judge correctness")

    sel = sel.sample(frac=1, random_state=SEED).reset_index(drop=True)  # random order
    sel["id"] = [f"A{i + 1:02d}" for i in range(len(sel))]

    sheet = sel[["id", "query", "ground_truth", "answer"]].rename(
        columns={"query": "question", "ground_truth": "reference_answer"}
    )
    for col in ["correctness_0_2", "factual_grounding_0_2", "security_reasoning_0_2", "comments"]:
        sheet[col] = ""
    sheet.to_csv(SHEET_CSV, index=False)

    key_cols = ["id", "query_id", "engine", "category", "faithfulness", "correctness", "answer_relevancy",
                "security_specificity", "context_utilization_score", "hallucinated_security_reasoning"]
    sel[key_cols].to_csv(KEY_CSV, index=False)
    with open(RUBRIC_TXT, "w", encoding="utf-8") as f:
        f.write(RUBRIC)

    meta = {
        "seed": SEED,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "quota_per_category_and_engine": {f"{c} | {e}": n for (c, e), n in QUOTA.items()},
        "min_answers_per_judge_correctness_level": MIN_PER_LEVEL,
        "sampling_attempts_needed": attempt + 1,
        "input_sha256": {RAW_JSON: sha256(RAW_JSON), JUDGE_CSV: sha256(JUDGE_CSV)},
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {SHEET_CSV} (send), {RUBRIC_TXT} (send), {KEY_CSV} (keep private), {META_JSON}")
    print(sel.groupby("engine").size().to_dict(), sel.groupby("category").size().to_dict())
    print("judge correctness levels in sample:", sel["correctness"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
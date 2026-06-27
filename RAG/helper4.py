import pandas as pd

# -------------------------
# File paths
# -------------------------
PRE_FILE = "evaluation_results.csv"
POST_FILE = "llm_change_eval_results.csv"

# -------------------------
# Load data
# -------------------------
pre = pd.read_csv(PRE_FILE)
post = pd.read_csv(POST_FILE)

# -------------------------
# CRITICAL CLEANING: Strip hidden spaces & normalize casing
# -------------------------
for df in [pre, post]:
    df["query_id"] = df["query_id"].astype(str).str.strip()
    df["engine"] = df["engine"].astype(str).str.strip().str.lower()

# -------------------------
# Keep only the matching queries evaluated in both files
# -------------------------
queries = post["query_id"].unique()

pre_filtered = pre[pre["query_id"].isin(queries)].copy()
post_filtered = post[post["query_id"].isin(queries)].copy()

# -------------------------
# Metrics to summarize
# -------------------------
metrics = [
    "faithfulness",
    "correctness",
    "answer_relevancy",
    "security_specificity",
    "context_utilization_score",
    "hallucinated_security_reasoning"
]

# -------------------------
# Function for descriptive statistics
# -------------------------
def summarize(df, name):
    print(f"\n{'='*70}")
    print(name)
    print("="*70)

    # Reorder display columns to match your exact format
    stats_columns = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    print("\nOverall:")
    print(df[metrics].describe().T[stats_columns])

    print("\nBy retrieval engine:")
    for engine in ["none", "bm25", "vector"]:
        print(f"\n--- {engine.upper()} ---")
        subset = df[df["engine"] == engine]
        
        if subset.empty:
            print(f"Warning: No rows found for engine '{engine}'. Check your CSV values.")
            continue

        print(subset[metrics].describe().T[stats_columns])

# -------------------------
# Print summaries
# -------------------------
summarize(pre_filtered, "PRE-ABLATION (FILTERED TO MATCHING QUERIES)")
summarize(post_filtered, "POST-ABLATION")
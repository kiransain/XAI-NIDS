# This script computes the summary statistics for ablation studies.
# output: CSV file with stats including mean, std, min, max, and quartiles for each metric, 
# grouped by dataset and retrieval engine.

import pandas as pd
import os
import csv

PRE_FILE = "evaluation_results.csv"
POST_FILE = "llm_change_eval_results.csv" # llm_change_eval_results.csv

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
grand_parent_dir = os.path.dirname(parent_dir)
PRE_FILE = os.path.join(grand_parent_dir, PRE_FILE)
POST_FILE = os.path.join(grand_parent_dir, POST_FILE)

pre = pd.read_csv(PRE_FILE)
post = pd.read_csv(POST_FILE)


for df in [pre, post]:
    df["query_id"] = df["query_id"].astype(str).str.strip()
    df["engine"] = df["engine"].astype(str).str.strip().str.lower()


queries = post["query_id"].unique()
print(sorted(queries))
print("Number of queries:", len(queries))

pre_filtered = pre[pre["query_id"].isin(queries)].copy()
post_filtered = post[post["query_id"].isin(queries)].copy()


metrics = [
    "faithfulness",
    "correctness",
    "answer_relevancy",
    "security_specificity",
    "context_utilization_score",
    "hallucinated_security_reasoning"
]

stats_columns = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

# collect all summary tables here
all_summaries = []

def summarize(df, name):
    print(f"\n{'='*70}")
    print(name)
    print("="*70)

    print("\nOverall:")
    overall = df[metrics].describe().T[stats_columns].round(2)
    print(overall)

    overall = overall.reset_index().rename(columns={"index": "metric"})
    overall.insert(0, "group", "overall")
    overall.insert(0, "dataset", name)
    all_summaries.append(overall)

    print("\nBy retrieval engine:")
    for engine in ["none", "bm25", "vector"]:
        print(f"\n--- {engine.upper()} ---")
        subset = df[df["engine"] == engine]

        if subset.empty:
            print(f"Warning: No rows found for engine '{engine}'. Check your CSV values.")
            continue

        engine_summary = subset[metrics].describe().T[stats_columns].round(2)
        print(engine_summary)

        engine_summary = engine_summary.reset_index().rename(columns={"index": "metric"})
        engine_summary.insert(0, "group", engine)
        engine_summary.insert(0, "dataset", name)
        all_summaries.append(engine_summary)


summarize(pre_filtered, "PRE-ABLATION (FILTERED TO MATCHING QUERIES)")
summarize(post_filtered, "POST-ABLATION")


final = pd.concat(all_summaries, ignore_index=True)
output_path = os.path.join(grand_parent_dir, "llm_ablation_change_res_stats.csv") #prompt_ablation_change_res_stats.csv
final.to_csv(output_path, index=False)
print(f"\nSaved summary stats to {output_path}")

# CSV output note: 27 stands for 9 queries * 3 engines, and not all 28 original queries.
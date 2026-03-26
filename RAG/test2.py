import json
import pandas as pd
import time
from deepeval.models import OllamaModel
from deepeval.metrics import (
    FaithfulnessMetric, 
    AnswerRelevancyMetric, 
    ContextualPrecisionMetric, 
    ContextualRecallMetric
)
from deepeval.test_case import LLMTestCase

# --- CONFIGURATION ---
MODEL_NAME = "llama3.1:8b" 
INPUT_FILE = "thesis_evaluation_results.json"
OUTPUT_CSV = "thesis_final_deepeval_scores.csv"
MAX_CONTEXT_CHARS = 4000  # Cap context to ~1000 tokens to prevent timeouts

# 1. Setup the Judge
print(f"--- Initializing {MODEL_NAME} Judge ---")
# Lowering timeout slightly; if it takes > 10 mins for one metric, something is wrong
judge_model = OllamaModel(
    model="llama3.1:8b", 
    timeout=3600,
    num_predict=512 
)

# 2. Initialize Metrics
# Note: async_mode=False is CRITICAL for local hardware stability
common_params = {"model": judge_model, "async_mode": False, "verbose_mode": True, "include_reason": True}

faithfulness = FaithfulnessMetric(**common_params)
relevancy = AnswerRelevancyMetric(**common_params)
precision = ContextualPrecisionMetric(**common_params)
recall = ContextualRecallMetric(**common_params)

# 3. Load Results
print(f"--- Loading pre-generated results from {INPUT_FILE} ---")
with open(INPUT_FILE, "r") as f:
    results_data = json.load(f)

evaluation_output = []

# 4. Loop through and Score
for i, entry in enumerate(results_data):
    query_id = entry.get("query_id", f"Q{i}")
    engine = entry.get("engine", "unknown")
    
    print(f"\n{'='*50}")
    print(f"[{i+1}/{len(results_data)}] Processing {query_id} | Engine: {engine}")
    print(f"{'='*50}")

    # --- DEFENSIVE CONTEXT HANDLING ---
    raw_context = entry.get("retrieved_context", "")
    
    if engine == "none" or not raw_context or "No additional technical documentation" in raw_context:
        context_list = ["No technical documentation available for baseline."]
    else:
        # Split by double newline as intended
        chunks = [c.strip() for c in raw_context.split("\n\n") if c.strip()]
        # Truncate each chunk if it's too massive, and limit total chunks
        context_list = [c[:2000] for c in chunks[:3]]

    # Create Test Case
    test_case = LLMTestCase(
        input=entry["query"],
        actual_output=entry["answer"],
        retrieval_context=context_list,
        expected_output=entry["ground_truth"]
    )

    row_results = {"query_id": query_id, "engine": engine}

    # Decide which metrics to run
    if engine == "none":
        metrics_to_run = [faithfulness, relevancy]
    else:
        metrics_to_run = [faithfulness, relevancy, precision, recall]

    for metric in metrics_to_run:
        metric_name = metric.__class__.__name__
        print(f"   > Measuring {metric_name}...")
        
        start_time = time.time()
        try:
            metric.measure(test_case)
            row_results[metric_name] = metric.score
            row_results[f"{metric_name}_reason"] = metric.reason
            print(f"     Success! ({round(time.time() - start_time, 2)}s)")
        except Exception as e:
            print(f"     FAILED scoring {metric_name}: {e}")
            row_results[metric_name] = None
            row_results[f"{metric_name}_reason"] = f"Error: {str(e)}"

    evaluation_output.append(row_results)
    
    # Save progress after every row (so you don't lose data if it crashes at 90%)
    pd.DataFrame(evaluation_output).to_csv(OUTPUT_CSV, index=False)

print(f"\n--- DONE! Final scores saved to {OUTPUT_CSV} ---")
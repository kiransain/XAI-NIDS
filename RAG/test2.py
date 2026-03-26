import json
import pandas as pd
from deepeval.models import OllamaModel
from deepeval.metrics import (
    FaithfulnessMetric, 
    AnswerRelevancyMetric, 
    ContextualPrecisionMetric, 
    ContextualRecallMetric
)
from deepeval.test_case import LLMTestCase

# 1. Setup the Judge (Using 8B as discussed for accuracy)
print("--- Initializing DeepSeek-R1:8B Judge ---")
judge_model = OllamaModel(model="llama3.1:8b", timeout=3600)

# 2. Initialize Metrics
# async_mode=False is safer for local Ollama to prevent "hanging"
faithfulness = FaithfulnessMetric(model=judge_model, async_mode=False)
relevancy = AnswerRelevancyMetric(model=judge_model, async_mode=False)
precision = ContextualPrecisionMetric(model=judge_model, async_mode=False)
recall = ContextualRecallMetric(model=judge_model, async_mode=False)

# 3. Load your PRE-GENERATED results
INPUT_FILE = "thesis_evaluation_results.json"
print(f"--- Loading pre-generated results from {INPUT_FILE} ---")

with open(INPUT_FILE, "r") as f:
    results_data = json.load(f)

evaluation_output = []

# 4. Loop through the JSON entries and Score them
for i, entry in enumerate(results_data):
    query_id = entry.get("query_id", f"Q{i}")
    engine = entry.get("engine", "unknown")
    
    print(f"[{i+1}/{len(results_data)}] Scoring {query_id} | Engine: {engine}")

    # DeepEval needs a list for context. 
    # If it's a string, we split it. If it's "No documentation...", we pass a placeholder.
    raw_context = entry.get("retrieved_context", "")
    if engine == "none" or "No additional technical documentation" in raw_context:
        context_list = ["No technical documentation provided for this baseline."]
    else:
        context_list = [c.strip() for c in raw_context.split("\n\n") if c.strip()]

    # Create the Test Case
    test_case = LLMTestCase(
        input=entry["query"],
        actual_output=entry["answer"],
        retrieval_context=context_list,
        expected_output=entry["ground_truth"]
    )

    # Prepare score storage for this row
    row_results = {
        "query_id": query_id,
        "engine": engine
    }

    # Execute Metrics
    # Note: We skip retrieval metrics for the 'none' engine
    metrics_to_run = [faithfulness, relevancy]
    if engine != "none":
        metrics_to_run.extend([precision, recall])

    for metric in metrics_to_run:
        metric_name = metric.__class__.__name__
        try:
            metric.measure(test_case)
            row_results[metric_name] = metric.score
            row_results[f"{metric_name}_reason"] = metric.reason
        except Exception as e:
            print(f"   Error scoring {metric_name}: {e}")
            row_results[metric_name] = None

    evaluation_output.append(row_results)

# 5. Save everything to a structured CSV
df = pd.DataFrame(evaluation_output)
df.to_csv("thesis_final_deepeval_scores.csv", index=False)

print("\n--- DONE! ---")
print(f"Scores saved to 'thesis_final_deepeval_scores.csv'")

# Quick Summary for your terminal
summary = df.groupby('engine')[[
    'FaithfulnessMetric', 
    'AnswerRelevancyMetric', 
    'ContextualPrecisionMetric', 
    'ContextualRecallMetric'
]].mean()

print("\n--- MEAN SCORES PER ENGINE ---")
print(summary)

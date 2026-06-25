import json
import csv
from pathlib import Path

INPUT_FILE = "eval_prompt_change_generation_res.json"
OUTPUT_FILE = "prompt_change_eval_results.csv"

def load_json_objects(filepath):
    text = Path(filepath).read_text(encoding="utf-8").strip()

    try:
        # Case 1: Proper JSON array
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    try:
        # Case 2: Comma-separated JSON objects
        wrapped = f"[{text.rstrip(',')}]"
        return json.loads(wrapped)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON file: {e}")

data = load_json_objects(INPUT_FILE)

fields = [
    "query_id",
    "engine",
    "faithfulness",
    "correctness",
    "answer_relevancy",
    "security_specificity",
    "context_utilization_score",
    "hallucinated_security_reasoning",
]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

    for row in data:
        writer.writerow({field: row.get(field, "") for field in fields})

print(f"Saved {len(data)} records to {OUTPUT_FILE}")
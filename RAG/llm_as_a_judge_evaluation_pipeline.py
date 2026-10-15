import os
import json
import time
from itertools import cycle
from google import genai
from google.genai import types
import time 

# Import evaluation prompt template
from prompt import PROMPT

# ----------------------------
# CONFIG
# ----------------------------
INPUT_FILE = "paper_raw_results_helper.json"
OUTPUT_FILE = "paper_results_llm_as_judge2.json"
GEMINI_MODEL = "gemini-3.8-flash"

# Key setup - Reads from environment variables or explicitly defined keys
API_KEYS = [
    # os.environ.get("GEMINI_API_KEY_2"),
]
# Filter out empty strings/None
API_KEYS = [k for k in API_KEYS if k]

# Fallback to default GEMINI_API_KEY environment variable if list is empty
if not API_KEYS and os.environ.get("GEMINI_API_KEY"):
    API_KEYS.append(os.environ.get("GEMINI_API_KEY"))

if not API_KEYS:
    raise ValueError("No Gemini API keys found. Please set GEMINI_API_KEY or populate API_KEYS.")

api_key_cycle = cycle(API_KEYS)

# Load raw evaluation data
with open(INPUT_FILE, "r") as f:
    results = json.load(f)

judged_results = []

for result in results:
    query_id = result.get("query_id", "")
    engine = result.get("engine", "")
    query = result.get("query", "")
    retrieved_context = result.get("retrieved_context", "")
    answer = result.get("answer", "")
    ground_truth = result.get("ground_truth", "")

    # Cycle through API keys
    current_key = next(api_key_cycle)
    client = genai.Client(api_key=current_key)

    # Format structured content for prompt
    formatted_prompt = f"""{PROMPT}

QUERY_ID: {query_id}
ENGINE: {engine}

[QUERY]
{query}

[CONTEXT]
{retrieved_context}

[ANSWER]
{answer}

[GROUND_TRUTH]
{ground_truth}
"""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=formatted_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    seed = 42,
                    response_mime_type="application/json"  # Enforce strictly valid JSON response
                )
            )

            # Parse response into dictionary
            eval_data = json.loads(response.text)
            
            # Ensure query_id and engine match current record
            eval_data["query_id"] = query_id
            eval_data["engine"] = engine

            judged_results.append(eval_data)
            print(f"Evaluated {query_id} successfully.")
            break

        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3
                print(f"Server busy for {query_id} (503). Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Error evaluating {query_id}: {e}")
                break

# Save all structured results into a single JSON file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(judged_results, f, indent=2)

print(f"Results saved! Check {OUTPUT_FILE}")
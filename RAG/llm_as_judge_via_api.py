'''
this file is for using an LLM as a judge to evaluate the generated answers from the benchmark.
'''
import json
import time
import re
from google import genai
from prompt import PROMPT as prompt

client = genai.Client(api_key="AQ.Ab8RN6K38onl_l0Gn93uIxvK-Hpom_4KFAnnf-iQI8dXCWaUhw")

with open("thesis_evaluation_results.json", "r") as f:
    data = json.load(f)

output_file = "llm_as_judge_generation_eval.json"
def extract_json(text):
    """
    Extract JSON from ```json ... ``` or fallback to raw parsing.
    """
    try:
        # try fenced block first
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # fallback: sometimes model returns raw JSON
        return json.loads(text)

    except Exception as e:
        return {
            "parse_error": str(e),
            "raw_response": text
        }


results = []

for idx, item in enumerate(data):

    to_be_eval = (
        f"{item['query_id']},"
        f"{item['engine']},"
        f"{item['query']},"
        f"{item['retrieved_context']},"
        f"{item['answer']},"
        f"{item['ground_truth']}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt + "\n" + to_be_eval
        )

        parsed = extract_json(response.text)

        # optional safety check
        parsed["query_id"] = item.get("query_id", parsed.get("query_id"))
        parsed["engine"] = item.get("engine", parsed.get("engine"))

        results.append(parsed)

        # checkpoint save EVERY iteration (non-negotiable for your sanity)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"done {idx+1}/{len(data)}")

        # rate limiting safety
        time.sleep(13)

    except Exception as e:
        print(f"failed at {idx}: {e}")
        time.sleep(30)
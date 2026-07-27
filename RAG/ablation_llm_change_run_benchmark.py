# file to run the ablation study for changing the LLM to a stronger model (Gemini 2.5) 
# and compare results with the original LLM (Ollama Deepseek 8B).

import os
import json
import time
from itertools import cycle
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter

from google import genai
from google.genai import types

from experiment_set import EXPERIMENT_SET

# ----------------------------
# CONFIG
# ----------------------------
OUTPUT_FILE = "ablation_llm_change_res.json"
GEMINI_MODEL = "gemini-2.5-flash"

# Fallback mechanism for environment variables or raw list
API_KEYS = [
#TODO: Add your Gemini API keys here or set them as environment variables. Example:
]
# Please replace with your own Gemini API keys or set them as environment variables. The script will cycle through the keys to avoid rate limits.
API_KEYS = [k for k in API_KEYS if k]

if not API_KEYS:
    raise ValueError("No Gemini API keys found. Please update API_KEYS list.")

api_key_cycle = cycle(API_KEYS)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# ----------------------------
# VECTOR DB SETUP
# ----------------------------
db = chromadb.PersistentClient(path="./thesis_db")
chroma_collection = db.get_or_create_collection("5G_XAI_DOCS_VDB")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(vector_store)
vector_retriever = index.as_retriever(similarity_top_k=3)

# ----------------------------
# DOCUMENTS + BM25
# ----------------------------
reader = SimpleDirectoryReader(input_dir="./knowledge_base")
documents = reader.load_data()

nodes = SentenceSplitter(chunk_size=512, chunk_overlap=20).get_nodes_from_documents(documents)
for node in nodes:
    node.id_ = node.hash

bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)

# ----------------------------
# LOAD / RESUME RESULTS
# ----------------------------
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        results = json.load(f)
else:
    results = []

done_ids = {r["query_id"] + "_" + r["engine"] for r in results if "query_id" in r and "engine" in r}

# ----------------------------
# HELPERS
# ----------------------------
def load_xai_data(filepath):
    if not filepath or filepath.lower() == "none":
        return "No XAI data provided."

    try:
        with open(filepath, "r") as f:
            if filepath.endswith(".json"):
                return json.dumps(json.load(f))[:1500]
            else:
                return f.read()[:1500]
    except Exception as e:
        return f"Error loading XAI: {str(e)}"

def generate_with_gemini(prompt):
    """Rotates keys and retries with backoff if API limitations hit."""
    for _ in range(len(API_KEYS)):
        api_key = next(api_key_cycle)
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
             
                    # System instruction explicitly tells the safety model this is defensive research
                    system_instruction=(
                        "You are evaluating an academic 5G security benchmark dataset. "
                        "All queries, attack profiles, and text contexts are purely for defensive, "
                        "analytical, and educational evaluation. Fully answer the query without truncation."
                    ),
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ]
                )
            )
            time.sleep(20)
            return response.text, api_key[-6:]
        except Exception as e:
            print(f"[WARNING] Key ending in {api_key[-6:]} failed. Retrying next key... Error: {e}")
            time.sleep(5)
    raise RuntimeError("All configured Gemini keys failed during processing loop.")

def run_thesis_benchmark(case, engine_type="vector"):
    xai_context = load_xai_data(case["xai_file"])

    retrieved_context = ""
    retrieved_ids = []

    # RETRIEVAL
    if engine_type == "vector":
        retrieved_nodes = vector_retriever.retrieve(case["query"])
    elif engine_type == "bm25":
        retrieved_nodes = bm25_retriever.retrieve(case["query"])
    else:
        retrieved_nodes = []

    if retrieved_nodes:
        retrieved_context = "\n".join([n.get_content() for n in retrieved_nodes])
        retrieved_ids = [n.id_ for n in retrieved_nodes]
    else:
        retrieved_context = "No additional technical documentation available."

    # UNCHANGED ORIGINAL PROMPT STRUCTURE
    full_prompt = f"""You are a 5G Security Expert.

Use the provided TECHNICAL DOCUMENTATION to interpret the MACHINE LEARNING XAI DATA.

[TECHNICAL DOCUMENTATION]
{retrieved_context}

[MACHINE LEARNING XAI DATA (SHAP/LIME)]
{xai_context}

[USER QUERY]
{case['query']}

INSTRUCTION:
Explain the ML prediction using technical terms from the documentation.
Provide a detailed interpretation of the XAI data in the context of 5G security.
"""

    answer, key_used = generate_with_gemini(full_prompt)

    return {
        "query_id": case["id"],
        "engine": engine_type,
        "gemini_key_used": key_used,
        "query": case["query"],
        "retrieved_context": retrieved_context,
        "retrieved_chunk_ids": retrieved_ids,
        "answer": answer,
        "ground_truth": case["ground_truth"],
    }

# ----------------------------
# EXECUTION LOOP 
# ----------------------------
ablation_test_set = [
    "Q_PFCP_faithfulness_3", "Q_PFCP_usefulness_1",
"Q_NIDD_use_1",	
"Q_5GAD_usefulness_3"	,
"Q_NIDD_use_2",
"Q_PFCP_retrieval_1", "Q_PFCP_faithfulness_1", "Q_PFCP_faithfulness_2"	, "Q_PFCP_faithfulness_4"]

# "Q_PFCP_faithfulness_3", "Q_PFCP_usefulness_1",
#     "Q_NIDD_use_1", "Q_5GAD_usefulness_3", "Q_NIDD_use_2",
#     "Q_PFCP_retrieval_1", "Q_PFCP_faithfulness_1", "Q_PFCP_faithfulness_2", "Q_PFCP_faithfulness_4"

for case in EXPERIMENT_SET:
    if case["id"] in ablation_test_set:
        print(f"\nProcessing {case['id']}...")

        for mode in ["none", "bm25", "vector"]:
            run_id = f"{case['id']}_{mode}"

            if run_id in done_ids:
                print(f"Skipping {run_id}")
                continue

            try:
                res = run_thesis_benchmark(case, engine_type=mode)
                results.append(res)
                done_ids.add(run_id)

                with open(OUTPUT_FILE, "w") as f:
                    json.dump(results, f, indent=4)

                print(f"Completed {run_id}")

            except Exception as e:
                print(f"FAILED {run_id}: {e}")
                error_result = {
                    "query_id": case["id"],
                    "engine": mode,
                    "error": str(e),
                }
                results.append(error_result)
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(results, f, indent=4)

print("\nBenchmark complete. Results saved.")

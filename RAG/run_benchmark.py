'''
main file to run the thesis benchmark evaluation.
'''

import os
import json
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine

from experiment_set import EXPERIMENT_SET


# ----------------------------
# CONFIG
# ----------------------------

OUTPUT_FILE = "main_thesis_evaluation_results.json"

Settings.llm = Ollama(model="deepseek-r1:8b", request_timeout=1000.0)
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

done_ids = {r["query_id"] + "_" + r["engine"] for r in results}


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


def run_thesis_benchmark(case, engine_type="vector"):
    xai_context = load_xai_data(case["xai_file"])

    retrieved_context = ""
    retrieved_ids = []

    # ----------------------------
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

    # PROMPT
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

    response = Settings.llm.complete(full_prompt)

    return {
        "query_id": case["id"],
        "engine": engine_type,
        "query": case["query"],
        "retrieved_context": retrieved_context,
        "retrieved_chunk_ids": retrieved_ids,
        "answer": str(response),
        "ground_truth": case["ground_truth"],
    }


# ----------------------------
# EXECUTION LOOP 
# ----------------------------
for case in EXPERIMENT_SET:
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

            # SAVE AFTER EACH RUN (checkpointing)
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
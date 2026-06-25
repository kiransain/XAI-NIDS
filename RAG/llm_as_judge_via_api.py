"""
Main file to run the thesis benchmark evaluation using Gemini.
"""

import os
import json
import time
from itertools import cycle

import chromadb

from google import genai
from google.genai import types

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter

from experiment_set import EXPERIMENT_SET
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)

# =====================================================
# CONFIG
# =====================================================

OUTPUT_FILE = "gemini_results.json"

GEMINI_MODEL = "gemini-2.5-flash"

API_KEYS = [
    "AQ.Ab8RN6LyPohWu2Ot8jdMt7uzbYiELdTbAz0IfeE8jHnVB8azsg",
    "AQ.Ab8RN6JqbsVVo7mXafjxhg1ZrVm7_cFUsLXrT2rSroqt6Ja7dQ",
"AQ.Ab8RN6KIFGYFZA-KvqJey1bM0W4OITm69fN8OC7khOO69nLmrQ",
]

API_KEYS = [k for k in API_KEYS if k]

if len(API_KEYS) == 0:
    raise ValueError(
        "No Gemini API keys found. "
        "Set GEMINI_KEY_1, GEMINI_KEY_2, GEMINI_KEY_3."
    )

api_key_cycle = cycle(API_KEYS)


# =====================================================
# GEMINI HELPER
# =====================================================

def generate_with_gemini(prompt):
    """
    Rotates through Gemini API keys.
    If one key hits quota, automatically tries the next.
    """

    last_error = None

    for _ in range(len(API_KEYS)):

        api_key = next(api_key_cycle)

        try:
            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1500,
                )
            )

            # Respect rate limits
            time.sleep(13)

            return response.text, api_key[-6:]

        except Exception as e:
            print(
                f"[WARNING] API key ending "
                f"{api_key[-6:]} failed: {e}"
            )
            last_error = e

    raise RuntimeError(
        f"All Gemini API keys exhausted. Last error: {last_error}"
    )


# =====================================================
# VECTOR DB SETUP
# =====================================================

db = chromadb.PersistentClient(path="./thesis_db")

chroma_collection = db.get_or_create_collection(
    "5G_XAI_DOCS_VDB"
)

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

index = VectorStoreIndex.from_vector_store(
    vector_store
)

vector_retriever = index.as_retriever(
    similarity_top_k=3
)


# =====================================================
# DOCUMENTS + BM25
# =====================================================

reader = SimpleDirectoryReader(
    input_dir="./knowledge_base"
)

documents = reader.load_data()

nodes = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=20
).get_nodes_from_documents(documents)

for node in nodes:
    node.id_ = node.hash

bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=3
)


# =====================================================
# LOAD / RESUME RESULTS
# =====================================================

if os.path.exists(OUTPUT_FILE):

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

else:
    results = []

done_ids = {
    f"{r['query_id']}_{r['engine']}"
    for r in results
    if "query_id" in r and "engine" in r
}


# =====================================================
# HELPERS
# =====================================================

def load_xai_data(filepath):

    if not filepath or filepath.lower() == "none":
        return "No XAI data provided."

    try:

        with open(filepath, "r", encoding="utf-8") as f:

            if filepath.endswith(".json"):
                return json.dumps(
                    json.load(f)
                )[:1500]

            return f.read()[:1500]

    except Exception as e:
        return f"Error loading XAI: {str(e)}"


# =====================================================
# BENCHMARK RUNNER
# =====================================================

def run_thesis_benchmark(
    case,
    engine_type="vector"
):

    xai_context = load_xai_data(
        case["xai_file"]
    )

    retrieved_context = ""
    retrieved_ids = []

    # -----------------------------------------
    # RETRIEVAL
    # -----------------------------------------

    if engine_type == "none":

        retrieved_nodes = []

    elif engine_type == "bm25":

        retrieved_nodes = bm25_retriever.retrieve(
            case["query"]
        )

    elif engine_type == "vector":

        retrieved_nodes = vector_retriever.retrieve(
            case["query"]
        )

    else:

        raise ValueError(
            f"Unknown engine: {engine_type}"
        )

    # -----------------------------------------
    # CONTEXT BUILDING
    # -----------------------------------------

    if retrieved_nodes:

        retrieved_context = "\n\n".join(
            node.get_content()
            for node in retrieved_nodes
        )

        retrieved_ids = [
            node.id_
            for node in retrieved_nodes
        ]

    else:

        retrieved_context = (
            "No additional technical "
            "documentation available."
        )

    # -----------------------------------------
    # PROMPT
    # -----------------------------------------

    full_prompt = f"""
You are a 5G Security Expert.

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

    answer, key_used = generate_with_gemini(
        full_prompt
    )

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


# =====================================================
# EXECUTION LOOP
# =====================================================

ablation_test_set= [
"Q_PFCP_retrieval_3",	"Q_PFCP_faithfulness_3", "Q_PFCP_usefulness_1",
"Q_NIDD_use_1",	
"Q_5GAD_usefulness_3"	,
"Q_NIDD_use_2",
"Q_PFCP_retrieval_1", "Q_PFCP_faithfulness_1", "Q_PFCP_faithfulness_2"	, "Q_PFCP_faithfulness_4"]



for case in EXPERIMENT_SET:
    if case["id"] in ablation_test_set:
        print(
            f"\nProcessing {case['id']}..."
        )

        for mode in [
            "none",
            "bm25",
            "vector"
        ]:

            run_id = f"{case['id']}_{mode}"

            if run_id in done_ids:

                print(
                    f"Skipping {run_id}"
                )
                continue

            try:

                result = run_thesis_benchmark(
                    case,
                    engine_type=mode
                )

                results.append(result)

                done_ids.add(run_id)

                with open(
                    OUTPUT_FILE,
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        results,
                        f,
                        indent=4,
                        ensure_ascii=False
                    )

                print(
                    f"Completed {run_id}"
                )

            except Exception as e:

                print(
                    f"FAILED {run_id}: {e}"
                )

                error_result = {
                    "query_id": case["id"],
                    "engine": mode,
                    "error": str(e),
                }

                results.append(
                    error_result
                )

                with open(
                    OUTPUT_FILE,
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        results,
                        f,
                        indent=4,
                        ensure_ascii=False
                    )

print(
    "\nBenchmark complete. "
    "Results saved."
)

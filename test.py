import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine

# configuration
Settings.llm = Ollama(model="deepseek-coder:6.7b ", request_timeout=1000.0) # old llama3.2:1b
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
db = chromadb.PersistentClient(path="./thesis_db")
chroma_collection = db.get_or_create_collection("5g_security")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# folder of PDFs converted to text
reader = SimpleDirectoryReader(input_dir="./knowledge_base")
documents = reader.load_data()

# splitter
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes, storage_context=storage_context)
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes, 
    similarity_top_k=3
)
vector_retriever = index.as_retriever(similarity_top_k=3)

print(f"RAG System Ready! Processed {len(nodes)} text chunks from your 5G PDFs.")


test_query = "What features are used for PFCP intrusion detection?"
print(f"\nTesting Query: {test_query}")

v_nodes = index.as_retriever(similarity_top_k=2).retrieve(test_query)
print(f"Vector retrieved {len(v_nodes)} nodes.")

b_nodes = bm25_retriever.retrieve(test_query)
print(f"BM25 retrieved {len(b_nodes)} nodes.")

print("\n--- VECTOR RETRIEVAL CONTENT ---")
for i, node in enumerate(v_nodes):
    print(f"\nNode {i+1} (Score: {node.score:.4f}):")
    print(node.get_content()[:300] + "...") # Print first 300 chars

print("\n--- BM25 RETRIEVAL CONTENT ---")
for i, node in enumerate(b_nodes):
    print(f"\nNode {i+1}:")
    print(node.get_content()[:300] + "...")

vector_query_engine = RetrieverQueryEngine.from_args(
    retriever=vector_retriever, 
    response_mode="compact"
)

bm25_query_engine = RetrieverQueryEngine.from_args(
    retriever=bm25_retriever,
    response_mode="compact"
)


# # responses

# print("\n" + "="*30)
# print("GENERATING LLM ANSWERS")
# print("="*30)

# print("\nThinking... (Vector RAG)")
# v_response = vector_query_engine.query(test_query)
# print(f"\n[VECTOR RAG RESPONSE]:\n{v_response}")

# print("\nThinking... (BM25 RAG)")
# b_response = bm25_query_engine.query(test_query)
# print(f"\n[BM25 RAG RESPONSE]:\n{b_response}")

import json
import os
from llama_index.core import Response

# 1.EXPERIMENT SET


EXPERIMENT_SET = [
    {
        "id": "Q1",
        "sample_id": "145",
        "xai_file": "results/5G-NIDD/binary/DecisionTree/shap_individual_145.json",
        "query": "Interpret the SHAP values for this file",
        "ground_truth": "Answer X"
    },
    # {
    #     "id": "Q2",
    #     "sample_id": "17363",
    #     "xai_file": "results/5G-NIDD/binary/DecisionTree/lime_individual_17363.txt",
    #     "query": "Compare the LIME importance of src_bytes with the security baseline for 5G UPF interfaces.",
    #     "ground_truth": "The UPF (User Plane Function) should see a burst in src_bytes during an exfiltration event..."
    # }
    # and more
]

def load_xai_data(filepath):
    """Helper to read either JSON or TXT XAI outputs"""
    try:
        with open(filepath, 'r') as f:
            if filepath.endswith('.json'):
                data = json.load(f)
                return json.dumps(data)[:1500] 
            else:
                return f.read()[:1500]
    except Exception as e:
        return f"Error loading XAI: {str(e)}"

def run_thesis_benchmark(case, engine_type="vector"):
    xai_context = load_xai_data(case['xai_file'])
    retrieved_context = ""
    if engine_type == "vector":
        nodes = vector_retriever.retrieve(case['query'])
        retrieved_context = "\n".join([n.get_content() for n in nodes])
    elif engine_type == "bm25":
        nodes = bm25_retriever.retrieve(case['query'])
        retrieved_context = "\n".join([n.get_content() for n in nodes])
    else:
        retrieved_context = "No additional technical documentation available."

    # LLM PROMPT (inspired by msc thesis)
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
                    Explain the ML prediction using the technical terms found in the documentation. 
                    If the data shows high 'Bwd Pkt Len Std', look for what that means in the 5G context provided.
                    """
  
    response = Settings.llm.complete(full_prompt)

    return {
        "query_id": case['id'],
        "engine": engine_type,
        "query": case['query'],
        "retrieved_context": retrieved_context,
        "answer": str(response),
        "ground_truth": case['ground_truth']
    }

#execution
results = []
print(f"Starting Benchmark for {len(EXPERIMENT_SET)} queries...")

for case in EXPERIMENT_SET:
    print(f"Processing {case['id']}...")
    for mode in ["none", "bm25", "vector"]: # no RAG, BM25 RAG, Vector RAG
        res = run_thesis_benchmark(case, engine_type=mode)
        results.append(res)

# results saved as JSON file for RAGAS
with open("thesis_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Benchmark complete! Data saved to thesis_evaluation_results.json")
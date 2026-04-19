import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
import json
from llama_index.core import Response
from experiment_set import EXPERIMENT_SET


Settings.llm = Ollama(model="deepseek-r1:8b", request_timeout=1000.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

db = chromadb.PersistentClient(path="./thesis_db")
chroma_collection = db.get_or_create_collection("5G_XAI_DOCS_VDB")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load index from existing DB — no re-embedding!
index = VectorStoreIndex.from_vector_store(vector_store)
vector_retriever = index.as_retriever(similarity_top_k=3)

reader = SimpleDirectoryReader(input_dir="./knowledge_base")
documents = reader.load_data()
nodes = SentenceSplitter(chunk_size=512, chunk_overlap=20).get_nodes_from_documents(documents)
for node in nodes: node.id_ = node.hash

bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)

test_query = "What features are used for PFCP intrusion detection?"
print(f"\nTesting Query: {test_query}")

v_nodes = index.as_retriever(similarity_top_k=3).retrieve(test_query)
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

# 1.EXPERIMENT SET
def load_xai_data(filepath):
    """Helper to read either JSON or TXT XAI outputs safely"""
    # Safe check for "none"
    if filepath.lower() == "none" or not filepath:
        return "No XAI data provided for this query."
        
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
    retrieved_ids = []
    if engine_type == "vector":
        nodes = vector_retriever.retrieve(case['query'])
        retrieved_context = "\n".join([n.get_content() for n in nodes])
        retrieved_ids = [n.id_ for n in nodes]
    elif engine_type == "bm25":
        nodes = bm25_retriever.retrieve(case['query'])
        retrieved_context = "\n".join([n.get_content() for n in nodes])
        retrieved_ids = [n.id_ for n in nodes]
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
                    Provide a detailed interpretation of the XAI data in the context of 5G security.
                    """
  
    response = Settings.llm.complete(full_prompt)

    return {
        "query_id": case['id'],
        "engine": engine_type,
        "query": case['query'],
        "retrieved_context": retrieved_context,
        "retrieved_chunk_ids": retrieved_ids, 
        "answer": str(response),
        "ground_truth": case['ground_truth']
    }

#execution
results = []

for case in EXPERIMENT_SET:
    print(f"Processing {case['id']}...")
    for mode in ["none", "bm25", "vector"]: # 3 cases: no RAG, BM25 RAG, Vector RAG
        res = run_thesis_benchmark(case, engine_type=mode)
        results.append(res)

# results saved as JSON file for later RAGAS evaluation
with open("thesis_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Benchmark complete! Data saved to thesis_evaluation_results.json")

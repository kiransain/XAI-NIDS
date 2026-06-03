# generate_qrels.py
'''
this script is for generating qrels (query relevance judgments) for the benchmark evaluation.
it pools candidate chunks together and asks the annotator to label them as relevant or not for each query in the experiment set.
'''
import json
import random
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb
from experiment_set import EXPERIMENT_SET

# --- Setup (same as run_benchmark.py) ---
Settings.llm = Ollama(model="deepseek-r1:8b", request_timeout=1000.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

db = chromadb.PersistentClient(path="./thesis_db")
chroma_collection = db.get_or_create_collection("5G_XAI_DOCS_VDB")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store)
vector_retriever = index.as_retriever(similarity_top_k=3)

reader = SimpleDirectoryReader(input_dir="./knowledge_base")
documents = reader.load_data()
nodes = SentenceSplitter(chunk_size=512, chunk_overlap=20).get_nodes_from_documents(documents)
for node in nodes:
    node.id_ = node.hash

node_map = {}
for node in nodes:
    node_map[node.id_] = node # for random sampling later

bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)

# --- Annotation loop ---
def generate_qrels(n_random=2):
    """
    n_random: how many random corpus chunks to inject per query
              to reduce pooling bias
    """
    qrels = {}

    for case in EXPERIMENT_SET:
        qid   = case["id"]
        query = case["query"]

        print("\n" + "="*70)
        print(f"Query: {qid}")
        print(f"  {query}")
        print("="*70)

        v_nodes = vector_retriever.retrieve(query)
        b_nodes = bm25_retriever.retrieve(query)

        # pool retrieved + random chunks to create a more balanced annotation set
        pooled = {n.id_: n for n in (v_nodes + b_nodes)}

        random_ids = random.sample(
            [nid for nid in node_map if nid not in pooled],
            min(n_random, len(node_map) - len(pooled))
        )
        for rid in random_ids:
            pooled[rid] = node_map[rid]

        # random shuffling in hope to remove bias
        items = list(pooled.items())
        random.shuffle(items)

        selected = []
        for i, (nid, node) in enumerate(items):
            print(f"\n--- Chunk {i+1}/{len(items)} ---")
            print(node.get_content())
            decision = input("\nRelevant? (y / n / s to skip query): ").strip().lower()
            if decision == "s":
                break
            if decision == "y":
                selected.append(nid)

        qrels[qid] = selected
        print(f"  --> Marked {len(selected)} relevant chunk(s) for {qid}")

        with open("qrels.json", "w") as f:
            json.dump(qrels, f, indent=4)

    print("\nDone! Saved to qrels.json")

if __name__ == "__main__":
    generate_qrels(n_random=2)
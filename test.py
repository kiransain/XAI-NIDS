import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine

# configuration
Settings.llm = Ollama(model="llama3.2:1b", request_timeout=300.0)
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


# responses

print("\n" + "="*30)
print("GENERATING LLM ANSWERS")
print("="*30)

print("\nThinking... (Vector RAG)")
v_response = vector_query_engine.query(test_query)
print(f"\n[VECTOR RAG RESPONSE]:\n{v_response}")

print("\nThinking... (BM25 RAG)")
b_response = bm25_query_engine.query(test_query)
print(f"\n[BM25 RAG RESPONSE]:\n{b_response}")
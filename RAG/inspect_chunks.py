# inspect_chunks.py — Lazy version: reads directly from ChromaDB!
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

# 1. Setup the same embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# 2. Connect to your EXISTING database
db = chromadb.PersistentClient(path="./thesis_db")
chroma_collection = db.get_or_create_collection("5G_XAI_DOCS_VDB")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 3. Load the index directly from the DB (We skip reading PDFs entirely!)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

# 4. Extract the exact nodes currently sitting in your DB
# (We ask LlamaIndex to fetch the documents stored in the index)
nodes = list(index.docstore.docs.values())

print(f"Total chunks in DB: {len(nodes)}\n")

# 5. Print the exact IDs and content that your main script is seeing!
for node in nodes:
    print(f"ID: {node.node_id}")
    print(f"Source: {node.metadata.get('file_name', 'unknown')}")
    print(f"Content: {node.get_content()[:300]}")
    print("---")
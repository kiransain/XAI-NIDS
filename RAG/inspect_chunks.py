# inspect_chunks.py — run this once, no LLM needed
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

db = chromadb.PersistentClient(path="./thesis_db")
chroma_collection = db.get_or_create_collection("5G_XAI_DOCS_VDB")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

reader = SimpleDirectoryReader(input_dir="./knowledge_base")
documents = reader.load_data()
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents)

print(f"Total chunks: {len(nodes)}\n")
for node in nodes:
    print(f"ID: {node.node_id}")
    print(f"Source: {node.metadata.get('file_name', 'unknown')}")
    print(f"Content: {node.get_content()[:300]}")
    print("---")
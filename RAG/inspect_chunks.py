# inspect_chunks.py — Lazy version: reads directly from ChromaDB!
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

# inspect_chunks.py — Bulletproof version (Skips DB entirely)
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# 1. Read PDFs straight from the folder
reader = SimpleDirectoryReader(input_dir="./knowledge_base")
documents = reader.load_data()

# 2. Split them exactly how your main pipeline does
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents)

print(f"Total chunks processed: {len(nodes)}\n")

# 3. Apply the content hash and print
# for node in nodes:
#     # Use the static hash property
#     node.id_ = node.hash  
#     print(f"ID: {node.node_id}")
#     print(f"Source: {node.metadata.get('file_name', 'unknown')}")
#     print(f"Content: {node.get_content()[:300]}")
#     print("---")

search_terms = ["DoS", "vital threat"]  # change per query

print(f"\nSearching for: {search_terms}")
for node in nodes:
    node.id_ = node.hash
    content = node.get_content().lower()
    if all(term.lower() in content for term in search_terms):
        print(f"\n✓ RELEVANT CHUNK FOUND")
        print(f"  ID: {node.node_id}")
        print(f"  Preview: {node.get_content()[:300]}")
        print("---")

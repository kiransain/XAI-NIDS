'''
this file is for indexing the PDF documents in the knowledge base into ChromaDB.
'''
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
for node in nodes:
    node.id_ = node.hash

# Guard condition
if chroma_collection.count() > 0:
    print(f"Collection already has {chroma_collection.count()} chunks. Skipping indexing.")
    print("Delete ./thesis_db to re-index from scratch.")
else:
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    print(f"Indexed {len(nodes)} chunks into ChromaDB. Done!")

index = VectorStoreIndex(nodes, storage_context=storage_context)
print(f"Indexed {len(nodes)} chunks into ChromaDB. Done!")
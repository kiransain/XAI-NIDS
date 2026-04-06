# inspect_chunks.py 
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# 1. Read PDFs straight from the folder
reader = SimpleDirectoryReader(input_dir="./knowledge_base")
documents = reader.load_data()

# 2. Split them exactly how your main pipeline does
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents)

print(f"Total chunks processed: {len(nodes)}\n")

# content hash and print applied
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

# inspect_chunks.py 
'''
a simple script to inspect the chunks created from the PDF documents in the knowledge base. 
this is useful for debugging and understanding how the documents are being processed and split into chunks, 
'''
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
search_terms = []  # change per query e.g. ["DoS", "vital threat"]

print(f"\nSearching for: {search_terms}")
for node in nodes:
    node.id_ = node.hash
    content = node.get_content().lower()
    if all(term.lower() in content for term in search_terms):
        print(f"\n✓ RELEVANT CHUNK FOUND")
        print(f"  ID: {node.node_id}")
        print(f"  Preview: {node.get_content()[:300]}")
        print("---")

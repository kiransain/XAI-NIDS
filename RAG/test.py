import chromadb

db = chromadb.PersistentClient(path="./thesis_db")
collection = db.get_collection("5G_XAI_DOCS_VDB")

print(collection.metadata)
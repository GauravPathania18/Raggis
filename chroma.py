import chromadb

# 1. Create a client (using a persistent local folder here)
client = chromadb.PersistentClient(path="./my_chroma_data")

# 2. Create or get a collection
collection = client.get_or_create_collection(
    name="my_knowledge_base"
)

# 3. Add your documents
documents = [
    "Alexandra is a computer science student at the University of Washington.",
    "The university chess club meets every Tuesday to play and analyze matches.",
    "Seattle, where the University of Washington is located, is known for its coffee culture."
]
metadatas = [
    {"source": "student_profile", "category": "person"},
    {"source": "club_info", "category": "club"},
    {"source": "city_info", "category": "location"}
]
ids = ["doc1", "doc2", "doc3"]

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
print(f"Added {collection.count()} documents to the collection.")

# 4. Query by meaning (semantic search)
results = collection.query(
    query_texts=["What does the chess club do?"],
    n_results=1
)

print("\nQuery Results:")
print(results['documents'][0])  # Shows the most relevant documents
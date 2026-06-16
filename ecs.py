from sentence_transformers import SentenceTransformer
from fixed_chunk import process_document


def run_embedding_pipeline(file_path):
    # 1. Get chunks (THIS runs fixed_chunk logic)
    chunks = process_document(file_path)

    # 2. Sort (safety)
    chunks = sorted(chunks, key=lambda x: x["chunk_id"])

    # 3. Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 4. Embed
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)

    # 5. Print mapping
    for chunk, embedding in zip(chunks, embeddings):
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Embedding: {embedding}")
        print("-" * 50)


if __name__ == "__main__":
    run_embedding_pipeline("Cinderella.sty")
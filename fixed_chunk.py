# ================================
# DOCUMENT CHUNKING PIPELINE (NO EMBEDDINGS)
# ================================


# ----------------
# 1. READ DOCUMENT
# ----------------
def read_document(file_path):
    """
    Reads a text file and returns content as string.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""


# ----------------
# 2. CLEAN TEXT
# ----------------
def clean_text(text):
    """
    Normalize whitespace and remove unnecessary noise.
    """
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


# ----------------
# 3. CHUNK TEXT (RAG-OPTIMIZED)
# ----------------
def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits text into overlapping chunks.

    Args:
        text (str): Input text
        chunk_size (int): Characters per chunk
        overlap (int): Overlap between chunks

    Returns:
        list: List of chunk dictionaries
    """
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]

        if not chunk:
            continue

        chunks.append({
            "chunk_id": len(chunks),
            "text": chunk,
            "start_index": i,
            "end_index": i + len(chunk)
        })

    return chunks


# ----------------
# 4. FULL PIPELINE
# ----------------
def process_document(file_path, chunk_size=300, overlap=50):
    """
    Full pipeline:
    read -> clean -> chunk
    """
    print("📄 Reading document...")
    text = read_document(file_path)

    if not text:
        print("❌ Empty or unreadable document.")
        return []

    print("🧹 Cleaning text...")
    text = clean_text(text)

    print("✂️ Chunking text...")
    chunks = chunk_text(text, chunk_size, overlap)

    print(f"✅ Created {len(chunks)} chunks")

    return chunks


# ----------------
# 5. RUN
# ----------------
if __name__ == "__main__":
    file_path = "Cinderella.sty"  # change to your file

    chunks = process_document(file_path, chunk_size=300, overlap=50)

    # Preview chunks
    for chunk in chunks:
        print("\n----------------------------")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Start: {chunk['start_index']} | End: {chunk['end_index']}")
        print(f"Text Preview: {chunk['text']}")
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import re

def semantic_chunk_ordered(text, n_chunks=8):
    """Semantic chunking that preserves story order"""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Get embeddings
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(sentences)
    
    # Cluster sentences
    n_clusters = min(n_chunks, len(sentences))
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)
    
    # Group by cluster while tracking original indices
    indexed_chunks = {}
    for idx, (sent, label) in enumerate(zip(sentences, labels)):
        if label not in indexed_chunks:
            indexed_chunks[label] = []
        indexed_chunks[label].append((idx, sent))
    
    # Sort by original position and create chunks
    chunks = []
    for label in sorted(indexed_chunks.keys()):
        sorted_sents = sorted(indexed_chunks[label], key=lambda x: x[0])
        chunk_text = ' '.join([s[1] for s in sorted_sents])
        chunks.append(chunk_text)
    
    return chunks, embeddings

# Run on Cinderella
with open('Cinderella.sty', 'r', encoding='utf-8') as f:
    text = f.read()

chunks, embeddings = semantic_chunk_ordered(text, n_chunks=8)

print(f"\n{'='*60}")
print(f"Created {len(chunks)} semantic chunks")
print(f"{'='*60}")

for i, chunk in enumerate(chunks, 1):
    preview = chunk
    print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
    print(preview)
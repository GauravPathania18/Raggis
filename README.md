# 🧠 Hierarchical RAG System

A sophisticated Retrieval-Augmented Generation system that builds a **semantic pyramid** over your documents — enabling multi-level understanding and high-quality Q&A with source attribution.

---

## 🚀 Core Innovation

Traditional RAG retrieves chunks from a flat vector store. This system adds a **hierarchical summarization layer**:

```
Level 0 → Original chunks (most granular)
Level 1 → LLM summaries of semantically similar clusters
Level 2 → Summaries of Level 1 summaries
Level N → Top-level document themes (most abstract)
```

Each query benefits from context at multiple granularities simultaneously.

---

## 🏗️ Architecture

```
PDF Files
   │
   ▼
Text Extraction → Chunking (size=500, overlap=50)
   │
   ▼
Embedding Model (nomic-embed-text, 768-dim)
   │
   ▼
ChromaDB (Level 0 chunks + HNSW index)
   │
   ▼
┌─────────────────────────────────────────────┐
│          Recursive Clustering Loop           │
│                                             │
│  Level N embeddings                         │
│      → PCA (768-dim → 50-dim)               │
│      → StandardScaler                       │
│      → GMM (2–8 clusters)                   │
│      → LLM summary per cluster              │
│      → Store as Level N+1                   │
│      → Check silhouette convergence         │
└─────────────────────────────────────────────┘
   │
   ▼
Query → HNSW Search → Fetch chunks + summaries → LLM → Answer
```

---

## 🧮 Key Algorithms

### 1. Chunking

Overlapping chunks prevent information loss at boundaries:

```
chunk_size  = 500 characters
overlap     = 50 characters
step        = chunk_size − overlap = 450

Chunk i: text[i·step : i·step + chunk_size]
```

Paragraph-aware chunking is also supported, which preserves semantic boundaries over fixed-size splits.

---

### 2. Gaussian Mixture Model (GMM) Clustering

Unlike K-Means (hard assignment), GMM uses **soft clustering** — each point belongs to all clusters with a probability. This is more suitable for high-dimensional embedding spaces.

**Probability Density Function:**

$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

| Symbol | Meaning |
|--------|---------|
| $K$ | Number of clusters |
| $\pi_k$ | Mixing coefficient (weight of cluster $k$) |
| $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ | Gaussian with mean $\boldsymbol{\mu}_k$ and covariance $\boldsymbol{\Sigma}_k$ |

Fitted via **EM Algorithm**:
- **E-step**: Compute responsibilities — $r_{ik} = P(\text{cluster } k \mid \mathbf{x}_i)$
- **M-step**: Update $\pi_k$, $\boldsymbol{\mu}_k$, $\boldsymbol{\Sigma}_k$ to maximize log-likelihood

**Why GMM over K-Means?**
- Soft assignments (chunks can partially belong to multiple clusters)
- Handles elliptical cluster shapes
- Provides confidence scores — useful for borderline chunks

---

### 3. PCA Dimension Reduction

Embeddings are 768-dimensional; PCA reduces to 50 dimensions before clustering.

**Covariance matrix and eigenvector decomposition:**

$$\mathbf{C} = \frac{1}{n} \mathbf{X}^\top \mathbf{X}, \qquad \mathbf{C}\mathbf{v} = \lambda \mathbf{v}$$

**Explained variance for choosing $k$ components:**

$$\text{EVR}_k = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

Target: retain **85–95% of total variance**. This reduces the curse of dimensionality and speeds up GMM fitting dramatically.

---

### 4. Silhouette Score (Convergence Criterion)

Controls when to stop recursing:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\ b(i))}$$

| Term | Meaning |
|------|---------|
| $a(i)$ | Mean intra-cluster distance for point $i$ |
| $b(i)$ | Mean distance to the nearest other cluster |

| Score | Interpretation |
|-------|---------------|
| $> 0.5$ | Good clustering |
| $0.2$–$0.5$ | Reasonable |
| $< 0.2$ | Poor — stop recursing |

Recursion halts when silhouette improvement falls below **5%** or only one cluster is found.

---

### 5. Cosine Similarity (Retrieval)

ChromaDB retrieves chunks using cosine distance.

Range: $0$ (identical) to $2$ (opposite). ChromaDB stores normalized vectors ($\|\mathbf{v}\| = 1$), so the dot product alone suffices.

---

### 6. HNSW Index (Vector Store)

ChromaDB uses HNSW as its default index — a multi-layer graph for approximate nearest neighbor search:

```
Build:  O(N log N)
Query:  O(log N)          ← vs O(N) for brute force
Memory: O(N × M)          where M = avg connections per node
```

Higher layers contain exponentially sparser subsets of vectors; search starts at the top and descends.

---

## 📊 Data Structures

### `document_chunks` Collection

```python
{
    'ids':       'chunk_<timestamp>_<uuid>',
    'documents': '<chunk text>',
    'embeddings': [768-dim float array],
    'metadatas': {
        'source_file': 'filename.pdf',
        'chunk_id':    'chunk_000001',
        'level':        0,            # 0 = original, 1+ = summaries
        'type':        'chunk',       # or 'summary'
        'cluster_id':   int,
        'chunks_count': int,
    }
}
```

### `hierarchical_summaries` Collection

```python
{
    'ids':       'level_X_cluster_Y_<uuid>',
    'documents': '<cluster summary text>',
    'metadatas': {
        'level':            int,
        'cluster_id':       int,
        'chunks_count':     int,
        'silhouette_score': float,
        'source_files':     [list],
    }
}
```

---

## 🔄 Phase-by-Phase Flow

### Phase 1 — Ingestion

```
PDF → Text Extraction → Overlapping Chunks → Embeddings → ChromaDB (Level 0)
```

### Phase 2 — Hierarchical Clustering

```python
for level in range(max_levels):
    embeddings = get_by_level(level)
    pca_emb    = PCA(n_components=50).fit_transform(embeddings)
    scaled     = StandardScaler().fit_transform(pca_emb)
    labels, probs = GMM(n_components=k).fit(scaled)

    for cluster_id in unique(labels):
        chunks  = get_cluster_chunks(cluster_id)
        summary = ollama.generate(chunks)           # gemma3:1b
        store(summary, metadata={'level': level+1})

    score = silhouette_score(scaled, labels)
    if improvement(score) < 0.05 or n_clusters == 1:
        break
```

### Phase 3 — Retrieval

```
User Query
   → Embed query
   → HNSW top-k search (Level 0 + summaries)
   → Extract cluster IDs from results
   → Fetch cluster summaries (higher context)
   → Build prompt: [system] + [summaries] + [chunks] + [history] + [query]
   → LLM (gemma3:1b via Ollama)
   → Answer + source attribution
```

---

## ⚙️ Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `chunk_size` | 500 chars | Before overlap |
| `overlap` | 50 chars | Step = 450 |
| `embedding_model` | `nomic-embed-text` | 768-dim output |
| `llm_model` | `gemma3:1b` | Via Ollama |
| `pca_components` | 50 | Targets 85–95% EVR |
| `gmm_clusters` | 2–8 | Auto-selected |
| `silhouette_threshold` | 0.05 | Convergence delta |
| `max_levels` | Configurable | Safety cap |
| `retrieval_top_k` | Configurable | HNSW neighbors |

---

## 🗂️ Collections

| Collection | Contents |
|-----------|----------|
| `document_chunks` | Raw chunks (Level 0) + summary chunks (Level 1+) |
| `hierarchical_summaries` | Cluster-level summaries with metadata |

---

## 📦 Stack

- **Vector Store**: ChromaDB (HNSW index)
- **Embeddings**: Ollama (`nomic-embed-text`)
- **LLM**: Ollama (`gemma3:1b`)
- **Clustering**: scikit-learn GMM + PCA
- **Document Parsing**: PyMuPDF / pdfplumber

---

## 🧪 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Token F1 | Overlap between generated and reference answer |
| Hallucination Rate | Fraction of unsupported claims |
| Context Relevance | Semantic similarity of retrieved chunks to query |
| Query Latency | End-to-end response time |

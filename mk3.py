import chromadb
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
from typing import List, Dict, Tuple, Optional
import uuid
import pandas as pd
import re
import json
import requests
import time
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class HierarchicalClusterSummarizer:
    def __init__(self, chroma_path: str = "./chroma_db", 
                 ollama_model: str = "gemma3:1b",
                 ollama_url: str = "http://localhost:11434/api/generate"):
        """
        Initialize the hierarchical clustering system
        
        Args:
            chroma_path: Path to ChromaDB storage
            ollama_model: Ollama model name (e.g., gemma3:1b)
            ollama_url: Ollama API endpoint
        """
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # Main collection for chunks
        self.chunks_collection = self.client.get_or_create_collection(
            name="document_chunks",
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_search": 200,
                    "max_neighbors": 32
                }
            }
        )
        
        # Collection for hierarchical summaries (each level stored separately)
        self.summaries_collection = self.client.get_or_create_collection(
            name="hierarchical_summaries",
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_search": 100,
                    "max_neighbors": 16
                }
            }
        )
        
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.batch_size = 1000
        self.clustering_history = []  # Track clustering iterations
        
    def read_pdf(self, file_path: str) -> str:
        """Read PDF file and extract text"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            import PyPDF2
            print(f"Reading PDF: {file_path}")
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                print(f"PDF has {len(reader.pages)} pages")
                
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                        if (page_num + 1) % 50 == 0:
                            print(f"  Processed {page_num + 1}/{len(reader.pages)} pages")
            
            print(f"Extracted {len(text):,} characters")
            return text
            
        except ImportError:
            print("PyPDF2 not installed. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'PyPDF2'])
            import PyPDF2
            return self.read_pdf(file_path)
    
    def create_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Create overlapping chunks from text"""
        chunks = []
        step = chunk_size - overlap
        
        # Split by paragraphs for better context
        paragraphs = text.split('\n\n')
        
        if len(paragraphs) > 1 and len(text) > 10000:
            current_chunk = ""
            chunk_index = 0
            
            for para in paragraphs:
                if len(current_chunk) + len(para) > chunk_size and current_chunk:
                    chunk_id = f"chunk_{chunk_index:06d}"
                    
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_id': chunk_id,
                        'chunk_index': chunk_index,
                        'level': 0  # Base level
                    })
                    chunk_index += 1
                    current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            
            # Add last chunk
            if current_chunk:
                chunk_id = f"chunk_{chunk_index:06d}"
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_index,
                    'level': 0
                })
        else:
            # Simple chunking
            for i in range(0, len(text), step):
                chunk_text = text[i:i + chunk_size]
                if chunk_text.strip():
                    chunk_id = f"chunk_{i:06d}"
                    chunks.append({
                        'text': chunk_text,
                        'chunk_id': chunk_id,
                        'position': i,
                        'chunk_index': len(chunks),
                        'level': 0
                    })
        
        print(f"Created {len(chunks)} base chunks")
        return chunks
    
    def add_document(self, file_path: str, document_name: str = None) -> int:
        """Add a document to ChromaDB"""
        if document_name is None:
            document_name = os.path.basename(file_path)
        
        # Read and chunk the document
        text = self.read_pdf(file_path)
        chunks = self.create_chunks(text)
        
        if not chunks:
            print("No chunks created")
            return 0
        
        # Add chunks to ChromaDB
        print(f"\nAdding {len(chunks)} chunks to ChromaDB...")
        all_ids = []
        
        for batch_start in range(0, len(chunks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            batch_ids = []
            batch_documents = []
            batch_metadatas = []
            
            for chunk in batch_chunks:
                chroma_id = f"{document_name}_{chunk['chunk_id']}_{uuid.uuid4().hex[:8]}"
                batch_ids.append(chroma_id)
                batch_documents.append(chunk['text'])
                
                metadata = {
                    'source_file': document_name,
                    'chunk_id': chunk['chunk_id'],
                    'chunk_index': chunk['chunk_index'],
                    'level': 0,  # Base level
                    'chunk_preview': chunk['text'][:50].replace('\n', ' '),
                    'added_timestamp': pd.Timestamp.now().isoformat()
                }
                
                batch_metadatas.append(metadata)
            
            try:
                self.chunks_collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                all_ids.extend(batch_ids)
                print(f"  Added batch {batch_start//self.batch_size + 1}: {len(batch_ids)} chunks")
            except Exception as e:
                print(f"  Error: {e}")
        
        print(f"Successfully added {len(all_ids)} chunks")
        return len(all_ids)
    
    def get_embeddings_by_level(self, level: int = 0) -> Tuple[np.ndarray, List[str], List[Dict], List[str]]:
        """Get all embeddings and metadata for a specific level"""
        # Get all items from chunks collection
        total_count = self.chunks_collection.count()
        print(f"Total items in collection: {total_count}")
        
        all_embeddings = []
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        batch_size = 1000
        for offset in range(0, total_count, batch_size):
            try:
                result = self.chunks_collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["embeddings", "documents", "metadatas"]
                )
                
                # FIXED: Properly check if embeddings exist
                # Check if embeddings key exists and is not None, and has length > 0
                if result.get('embeddings') is not None and len(result['embeddings']) > 0:
                    # Filter by level if specified
                    for i, (emb, doc, meta, id_) in enumerate(zip(
                        result['embeddings'], 
                        result['documents'], 
                        result['metadatas'],
                        result['ids']
                    )):
                        if meta.get('level', 0) == level:
                            all_embeddings.append(emb)
                            all_documents.append(doc)
                            all_metadatas.append(meta)
                            all_ids.append(id_)
                            
            except Exception as e:
                print(f"  Error retrieving batch at offset {offset}: {e}")
                continue
        
        if len(all_embeddings) == 0:
            print(f"No embeddings found at level {level}")
            return None, None, None, None
        
        print(f"Found {len(all_embeddings)} items at level {level}")
        return np.array(all_embeddings), all_documents, all_metadatas, all_ids
    
    def perform_clustering(self, embeddings: np.ndarray, n_clusters: int = None) -> Dict:
        """Perform GMM clustering on embeddings"""
        if len(embeddings) < 3:
            print("Not enough embeddings for clustering (need at least 3)")
            return None
        
        print(f"Clustering {len(embeddings)} embeddings...")
        
        # Reduce dimensions with PCA
        if embeddings.shape[1] > 100 and len(embeddings) > 50:
            print(f"Reducing dimensions from {embeddings.shape[1]}...")
            n_components = min(50, len(embeddings) // 2)
            pca = PCA(n_components=n_components)
            embeddings_reduced = pca.fit_transform(embeddings)
            print(f"Reduced to {embeddings_reduced.shape[1]} dimensions")
            explained_variance = pca.explained_variance_ratio_.sum()
            print(f"Explained variance: {explained_variance:.2%}")
        else:
            embeddings_reduced = embeddings
        
        # Standardize
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_reduced)
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = max(2, min(8, len(embeddings) // 100))
        
        print(f"Running GMM with {n_clusters} clusters...")
        
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='diag',
            random_state=42,
            n_init=5,
            max_iter=200
        )
        
        cluster_labels = gmm.fit_predict(embeddings_scaled)
        cluster_probs = gmm.predict_proba(embeddings_scaled)
        
        # Calculate silhouette score
        if len(set(cluster_labels)) > 1:
            try:
                silhouette_avg = silhouette_score(embeddings_scaled, cluster_labels)
                print(f"Silhouette Score: {silhouette_avg:.3f}")
            except:
                silhouette_avg = 0
        else:
            silhouette_avg = 0
        
        return {
            'labels': cluster_labels,
            'probabilities': cluster_probs,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'embeddings_scaled': embeddings_scaled
        }
    
    def generate_summary_with_ollama(self, texts: List[str], max_chunks: int = 15) -> str:
        """Generate summary using Ollama local LLM"""
        # Take sample of chunks (avoid token limits)
        sample_size = min(max_chunks, len(texts))
        sampled_texts = texts[:sample_size]
        
        # Combine texts with separators
        combined_text = "\n\n---\n\n".join(sampled_texts)
        
        # Truncate if too long (Gemma3:1b has ~8k context)
        if len(combined_text) > 4000:
            combined_text = combined_text[:4000] + "..."
        
        # Create prompt for summarization
        prompt = f"""You are a document summarizer. Summarize the following content into a concise, informative summary that captures the main themes and key points.

Content to summarize:
{combined_text}

Provide a clear summary (3-5 paragraphs) that:
1. Identifies the main topic or theme
2. Highlights key concepts discussed
3. Mentions important details
4. Uses clear, concise language

Summary:"""
        
        try:
            # Call Ollama API
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get('response', '').strip()
                return summary if summary else self._generate_fallback_summary(texts)
            else:
                print(f"Ollama API error: {response.status_code}")
                return self._generate_fallback_summary(texts)
                
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return self._generate_fallback_summary(texts)
    
    def _generate_fallback_summary(self, texts: List[str]) -> str:
        """Fallback summary generation if Ollama fails"""
        # Extract keywords
        all_text = ' '.join(texts)
        words = all_text.lower().split()
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'to', 'for', 'is', 'are', 'was', 'were', 'of', 'with'}
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Get top keywords
        keyword_counts = Counter(keywords)
        top_keywords = keyword_counts.most_common(10)
        
        summary = f"""
Cluster Summary:
- Total chunks: {len(texts)}
- Key themes: {', '.join([f"{word}" for word, _ in top_keywords[:8]])}
- Sample content: {texts[0][:200]}...

This cluster contains content related to these key concepts.
"""
        return summary.strip()
    
    def store_cluster_summaries(self, clustering_results: Dict, 
                                documents: List[str],
                                metadatas: List[Dict],
                                level: int) -> Dict:
        """Store summaries for each cluster at a given level"""
        labels = clustering_results['labels']
        n_clusters = clustering_results['n_clusters']
        
        cluster_summaries = {}
        
        print(f"\n{'='*80}")
        print(f"STORING SUMMARIES FOR LEVEL {level}")
        print(f"{'='*80}")
        
        for cluster_id in range(n_clusters):
            # Get chunks in this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_texts = [documents[i] for i in cluster_indices]
            cluster_metas = [metadatas[i] for i in cluster_indices]
            
            print(f"\n📊 Cluster {cluster_id} at Level {level}:")
            print(f"   Chunks: {len(cluster_texts)}")
            
            # Generate summary using Ollama
            print(f"   Generating summary with {self.ollama_model}...")
            summary = self.generate_summary_with_ollama(cluster_texts)
            
            # Store summary in summaries collection
            summary_id = f"level_{level}_cluster_{cluster_id}_{uuid.uuid4().hex[:8]}"
            
            # Get source files in this cluster
            sources = list(set([m.get('source_file', 'unknown') for m in cluster_metas]))
            
            # Calculate average confidence
            avg_confidence = float(np.mean([max(clustering_results['probabilities'][i]) 
                                           for i in cluster_indices]))
            
            summary_metadata = {
                'type': 'cluster_summary',
                'level': level,
                'cluster_id': cluster_id,
                'chunks_count': len(cluster_texts),
                'source_files': sources,
                'avg_confidence': avg_confidence,
                'silhouette_score': clustering_results.get('silhouette_score', 0),
                'created_at': pd.Timestamp.now().isoformat(),
                'summary_preview': summary[:100]
            }
            
            # Store in summaries collection
            try:
                self.summaries_collection.add(
                    ids=[summary_id],
                    documents=[summary],
                    metadatas=[summary_metadata]
                )
                
                # Also create a summary chunk for the next level
                summary_chunk_id = f"summary_level_{level}_cluster_{cluster_id}"
                
                # Store summary as a chunk in the chunks collection for next level
                self.chunks_collection.add(
                    ids=[summary_chunk_id],
                    documents=[summary],
                    metadatas=[{
                        'type': 'cluster_summary',
                        'level': level + 1,  # Will be used in next iteration
                        'cluster_id': cluster_id,
                        'parent_cluster_id': cluster_id,
                        'chunks_count': len(cluster_texts),
                        'source_files': sources,
                        'is_summary': True
                    }]
                )
                
                cluster_summaries[cluster_id] = {
                    'summary': summary,
                    'summary_id': summary_id,
                    'chunks_count': len(cluster_texts),
                    'sources': sources,
                    'summary_chunk_id': summary_chunk_id
                }
                
                print(f"   ✓ Summary stored (ID: {summary_id[:20]}...)")
                print(f"   Summary preview: {summary[:150]}...")
                
            except Exception as e:
                print(f"   ✗ Error storing summary: {e}")
        
        return cluster_summaries
    
    def hierarchical_clustering(self, max_levels: int = 3, 
                                improvement_threshold: float = 0.05) -> Dict:
        """
        Perform hierarchical clustering recursively until no improvement
        
        Args:
            max_levels: Maximum number of hierarchical levels
            improvement_threshold: Minimum silhouette score improvement to continue
        
        Returns:
            Dictionary with clustering history
        """
        print("\n" + "="*80)
        print("STARTING HIERARCHICAL CLUSTERING")
        print("="*80)
        
        history = {
            'levels': [],
            'convergence': False,
            'final_level': 0
        }
        
        current_level = 0
        previous_silhouette = 0
        
        while current_level < max_levels:
            print(f"\n{'='*80}")
            print(f"LEVEL {current_level} CLUSTERING")
            print(f"{'='*80}")
            
            # Get embeddings for current level
            embeddings, documents, metadatas, ids = self.get_embeddings_by_level(level=current_level)
            
            if embeddings is None:
                print(f"No embeddings found at level {current_level}")
                break
            
            print(f"Found {len(embeddings)} items at level {current_level}")
            
            # Determine number of clusters based on data size
            n_clusters = max(2, min(8, len(embeddings) // 50))
            
            # Perform clustering
            clustering_results = self.perform_clustering(embeddings, n_clusters)
            
            if clustering_results is None:
                print("Clustering failed")
                break
            
            # Store summaries for this level
            cluster_summaries = self.store_cluster_summaries(
                clustering_results,
                documents,
                metadatas,
                current_level
            )
            
            # Track history
            history['levels'].append({
                'level': current_level,
                'n_clusters': clustering_results['n_clusters'],
                'silhouette_score': clustering_results['silhouette_score'],
                'n_items': len(embeddings),
                'summaries': cluster_summaries
            })
            
            current_silhouette = clustering_results['silhouette_score']
            
            # Check for improvement
            improvement = current_silhouette - previous_silhouette if current_level > 0 else 1
            
            print(f"\n📊 Level {current_level} Summary:")
            print(f"   Silhouette Score: {current_silhouette:.3f}")
            print(f"   Improvement: {improvement:.3f}")
            print(f"   Number of clusters: {clustering_results['n_clusters']}")
            
            # Stop if no improvement or only one cluster
            if current_level > 0 and improvement < improvement_threshold:
                print(f"\n✅ Convergence reached at level {current_level}")
                print(f"   Improvement ({improvement:.3f}) < threshold ({improvement_threshold})")
                history['convergence'] = True
                history['final_level'] = current_level
                break
            
            # Stop if only one cluster formed
            if clustering_results['n_clusters'] <= 1:
                print(f"\n✅ Only one cluster formed at level {current_level}")
                history['convergence'] = True
                history['final_level'] = current_level
                break
            
            previous_silhouette = current_silhouette
            current_level += 1
        
        if current_level >= max_levels:
            print(f"\n⚠️ Reached maximum levels ({max_levels})")
            history['final_level'] = max_levels - 1
        
        print("\n" + "="*80)
        print("HIERARCHICAL CLUSTERING COMPLETE")
        print("="*80)
        
        return history
    
    def get_summary_tree(self) -> pd.DataFrame:
        """Get all summaries organized as a tree"""
        result = self.summaries_collection.get(
            include=["documents", "metadatas"]
        )
        
        if not result['ids']:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'summary_id': result['ids'],
            'level': [m.get('level', -1) for m in result['metadatas']],
            'cluster_id': [m.get('cluster_id', -1) for m in result['metadatas']],
            'chunks_count': [m.get('chunks_count', 0) for m in result['metadatas']],
            'silhouette_score': [m.get('silhouette_score', 0) for m in result['metadatas']],
            'summary_preview': [doc[:100] for doc in result['documents']],
            'created_at': [m.get('created_at', '') for m in result['metadatas']]
        })
        
        return df.sort_values(['level', 'cluster_id'])
    
    def query_by_level(self, query: str, level: int = None, n_results: int = 5) -> Dict:
        """Query summaries at a specific level"""
        where_filter = {}
        if level is not None:
            where_filter = {"level": level}
        
        results = self.summaries_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        return results
    
    def visualize_hierarchy(self) -> None:
        """Print hierarchical structure of summaries"""
        df = self.get_summary_tree()
        
        if df.empty:
            print("No summaries found")
            return
        
        print("\n" + "="*80)
        print("HIERARCHICAL SUMMARY STRUCTURE")
        print("="*80)
        
        for level in sorted(df['level'].unique()):
            level_df = df[df['level'] == level]
            print(f"\n📚 Level {level} ({len(level_df)} clusters):")
            print("-" * 60)
            
            for _, row in level_df.iterrows():
                print(f"  Cluster {row['cluster_id']}:")
                print(f"    • {row['chunks_count']} chunks")
                print(f"    • Silhouette: {row['silhouette_score']:.3f}")
                print(f"    • Preview: {row['summary_preview'][:80]}...")
                print()


# ========== MAIN EXECUTION ==========

def main():
    # Initialize the processor
    processor = HierarchicalClusterSummarizer(
        chroma_path="./hierarchical_chroma_db",
        ollama_model="gemma3:1b",
        ollama_url="http://localhost:11434/api/generate"
    )
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("⚠️ Ollama doesn't seem to be running. Please start Ollama first.")
            print("   Run: ollama serve")
            return
        print("✓ Ollama is running")
    except Exception as e:
        print(f"⚠️ Cannot connect to Ollama: {e}")
        print("   Please make sure Ollama is running.")
        print("   Install and run: ollama pull gemma3:1b")
        print("   Then: ollama serve")
        return
    
    # List of files to add (modify as needed)
    files_to_add = [
        "The Odyssey.pdf",
        # Add more files here
    ]
    
    # Add files (if they exist)
    for file_path in files_to_add:
        if os.path.exists(file_path):
            print(f"\n📖 Processing: {file_path}")
            processor.add_document(file_path)
        else:
            print(f"\n⚠️ File not found: {file_path}")
    
    # If no files were added, check if there's existing data
    total_chunks = processor.chunks_collection.count()
    if total_chunks == 0:
        print("\n⚠️ No documents found. Please add PDF files to process.")
        return
    
    print(f"\n📊 Total chunks in database: {total_chunks}")
    
    # Perform hierarchical clustering
    history = processor.hierarchical_clustering(
        max_levels=3,  # Maximum depth of hierarchy
        improvement_threshold=0.05  # Stop if improvement < 5%
    )
    
    # Display results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    print(f"\n📈 Clustering History:")
    for level_info in history['levels']:
        print(f"  Level {level_info['level']}: {level_info['n_clusters']} clusters, "
              f"Silhouette: {level_info['silhouette_score']:.3f}, "
              f"Items: {level_info['n_items']}")
    
    print(f"\n✅ Convergence reached: {history['convergence']}")
    print(f"📊 Final level: {history['final_level']}")
    
    # Display summary tree
    processor.visualize_hierarchy()
    
    # Export summaries to CSV
    summaries_df = processor.get_summary_tree()
    if not summaries_df.empty:
        output_file = "hierarchical_summaries.csv"
        summaries_df.to_csv(output_file, index=False)
        print(f"\n💾 Summaries exported to: {output_file}")
    
    # Export history
    history_df = pd.DataFrame(history['levels'])
    if not history_df.empty:
        history_file = "clustering_history.csv"
        history_df.to_csv(history_file, index=False)
        print(f"📊 Clustering history exported to: {history_file}")

if __name__ == "__main__":
    main()
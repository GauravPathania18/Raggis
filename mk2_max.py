import chromadb
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import os
from typing import List, Dict
import uuid
import pandas as pd
import re

class DocumentProcessor:
    def __init__(self, chroma_path: str = "./chroma_db"):
        """Initialize ChromaDB client and collection"""
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_search": 200,
                    "max_neighbors": 32,
                    "ef_construction": 200
                }
            }
        )
        
        # Define batch size
        self.batch_size = 1000
    
    def read_document(self, file_path: str) -> str:
        """Read content from a document file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt' or ext == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif ext == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            except ImportError:
                print("PyPDF2 not installed. Please install it with: pip install PyPDF2")
                return ""
        
        elif ext == '.sty' or ext == '.tex':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return self.clean_latex_content(content)
        
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def clean_latex_content(self, content: str) -> str:
        """Clean LaTeX content to extract meaningful text"""
        # Remove LaTeX comments
        content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)
        
        # Remove LaTeX commands but keep their arguments where meaningful
        content = re.sub(r'\\begin\{[^}]+\}', '', content)
        content = re.sub(r'\\end\{[^}]+\}', '', content)
        
        # Replace common LaTeX commands
        replacements = {
            r'\\section\{([^}]+)\}': r'\1\n',
            r'\\subsection\{([^}]+)\}': r'\1\n',
            r'\\textbf\{([^}]+)\}': r'\1',
            r'\\textit\{([^}]+)\}': r'\1',
            r'\\emph\{([^}]+)\}': r'\1',
            r'\\label\{[^}]+\}': '',
            r'\\cite\{[^}]+\}': '[CITATION]',
            r'\\ref\{[^}]+\}': '[REFERENCE]',
            r'\\usepackage\{[^}]+\}': '',
            r'\\documentclass\{[^}]+\}': '',
        }
        
        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content)
        
        # Remove remaining LaTeX commands
        content = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?', '', content)
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = content.strip()
        
        return content
    
    def create_chunks(self, text: str, chunk_size: int = 50, overlap: int = 5) -> List[Dict]:
        """Create overlapping chunks from text"""
        chunks = []
        step = chunk_size - overlap
        
        for i in range(0, len(text), step):
            chunk_text = text[i:i + chunk_size]
            
            if not chunk_text.strip():
                continue
            
            # Try to cut at sentence boundaries for LaTeX files
            if i > 0 and i + chunk_size < len(text):
                for j in range(i + chunk_size, min(i + chunk_size + 10, len(text))):
                    if text[j] in '.!?':
                        chunk_text = text[i:j+1]
                        break
            
            chunk_id = f"chunk_{i:04d}_{i+len(chunk_text):04d}"
            
            chunk_info = {
                'text': chunk_text,
                'chunk_id': chunk_id,
                'start_position': i,
                'end_position': i + len(chunk_text),
                'chunk_index': len(chunks),
                'chunk_size': len(chunk_text),
                'file_type': self.detect_file_type_from_text(text)
            }
            chunks.append(chunk_info)
            
            if i + chunk_size >= len(text):
                break
        
        print(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def detect_file_type_from_text(self, text: str) -> str:
        """Detect if text is from LaTeX file"""
        if '\\documentclass' in text or '\\usepackage' in text or '\\begin{' in text:
            return 'latex'
        return 'plain_text'
    
    def add_chunks_to_chromadb(self, chunks: List[Dict], source_file: str) -> List[str]:
        """Add chunks to ChromaDB with embeddings and metadata in batches"""
        all_ids = []
        total_chunks = len(chunks)
        
        print(f"Adding {total_chunks} chunks to ChromaDB in batches of {self.batch_size}...")
        
        for batch_start in range(0, total_chunks, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]
            
            print(f"  Processing batch {batch_start//self.batch_size + 1}: chunks {batch_start} to {batch_end-1}")
            
            batch_ids = []
            batch_documents = []
            batch_metadatas = []
            
            for chunk in batch_chunks:
                chroma_id = f"{source_file}_{chunk['chunk_id']}_{uuid.uuid4().hex[:8]}"
                batch_ids.append(chroma_id)
                batch_documents.append(chunk['text'])
                
                metadata = {
                    'source_file': source_file,
                    'chunk_id': chunk['chunk_id'],
                    'chunk_index': chunk['chunk_index'],
                    'start_position': chunk['start_position'],
                    'end_position': chunk['end_position'],
                    'chunk_size': chunk['chunk_size'],
                    'chunk_text_preview': chunk['text'][:30],
                    'file_type': chunk.get('file_type', 'unknown')
                }
                batch_metadatas.append(metadata)
            
            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                all_ids.extend(batch_ids)
                print(f"    ✓ Added {len(batch_ids)} chunks")
            except Exception as e:
                print(f"    ✗ Error adding batch: {e}")
                # Try with smaller sub-batches
                if self.batch_size > 100:
                    print(f"    Retrying with smaller batch size...")
                    for sub_start in range(0, len(batch_chunks), 500):
                        sub_end = min(sub_start + 500, len(batch_chunks))
                        sub_chunks = batch_chunks[sub_start:sub_end]
                        
                        sub_ids = batch_ids[sub_start:sub_end]
                        sub_docs = batch_documents[sub_start:sub_end]
                        sub_metas = batch_metadatas[sub_start:sub_end]
                        
                        try:
                            self.collection.add(
                                ids=sub_ids,
                                documents=sub_docs,
                                metadatas=sub_metas
                            )
                            all_ids.extend(sub_ids)
                            print(f"      ✓ Added sub-batch of {len(sub_ids)} chunks")
                        except Exception as sub_e:
                            print(f"      ✗ Error adding sub-batch: {sub_e}")
        
        print(f"Successfully added {len(all_ids)} chunks to ChromaDB")
        return all_ids
    
    def get_all_embeddings_with_metadata(self) -> tuple:
        """Retrieve all embeddings, documents, and metadata from ChromaDB"""
        total_count = self.collection.count()
        print(f"Retrieving {total_count} items from ChromaDB...")
        
        all_embeddings = []
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        batch_size = 1000
        for offset in range(0, total_count, batch_size):
            try:
                result = self.collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["embeddings", "documents", "metadatas"]
                )
                
                # FIXED: Properly check if embeddings exist
                # Check if embeddings key exists and has elements
                if result.get('embeddings') is not None and len(result['embeddings']) > 0:
                    all_embeddings.extend(result['embeddings'])
                    all_documents.extend(result['documents'])
                    all_metadatas.extend(result['metadatas'])
                    all_ids.extend(result['ids'])
                    print(f"  Retrieved batch {offset//batch_size + 1}: {len(result['ids'])} items")
                else:
                    print(f"  Batch {offset//batch_size + 1} has no embeddings")
                    # Don't break - there might be more batches
                    
            except Exception as e:
                print(f"  Error retrieving batch {offset//batch_size + 1}: {e}")
                continue
        
        # FIXED: Check if list is empty using len()
        if len(all_embeddings) == 0:
            print("No embeddings found in the collection")
            return None, None, None, None
        
        # Convert to numpy array
        embeddings = np.array(all_embeddings)
        print(f"Total retrieved: {len(embeddings)} embeddings")
        
        return embeddings, all_documents, all_metadatas, all_ids
    
    def perform_gmm_clustering(self, embeddings: np.ndarray, n_clusters: int = None) -> Dict:
        """Perform GMM clustering on embeddings"""
        # Standardize the embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = max(2, min(10, int(np.sqrt(len(embeddings) / 2))))
            print(f"Auto-determined number of clusters: {n_clusters}")
        
        # Perform GMM clustering
        print(f"Running GMM clustering with {n_clusters} clusters...")
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        cluster_labels = gmm.fit_predict(embeddings_scaled)
        cluster_probs = gmm.predict_proba(embeddings_scaled)
        
        return {
            'labels': cluster_labels,
            'probabilities': cluster_probs,
            'model': gmm,
            'scaler': scaler,
            'n_clusters': n_clusters,
            'bic_score': gmm.bic(embeddings_scaled),
            'aic_score': gmm.aic(embeddings_scaled)
        }
    
    def update_chunks_with_clusters(self, clustering_results: Dict, chroma_ids: List[str], 
                                   metadatas: List[Dict], documents: List[str]) -> None:
        """Update ChromaDB with cluster labels in batches"""
        total_updates = len(chroma_ids)
        print(f"Updating {total_updates} chunks with cluster labels in batches...")
        
        batch_size = 1000
        for i in range(0, total_updates, batch_size):
            batch_end = min(i + batch_size, total_updates)
            batch_ids = chroma_ids[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_labels = clustering_results['labels'][i:batch_end]
            batch_probs = clustering_results['probabilities'][i:batch_end]
            
            for j, (chroma_id, metadata, label) in enumerate(zip(batch_ids, batch_metadatas, batch_labels)):
                metadata['cluster_id'] = int(label)
                metadata['cluster_confidence'] = float(max(batch_probs[j]))
                
                try:
                    self.collection.update(
                        ids=[chroma_id],
                        metadatas=[metadata]
                    )
                except Exception as e:
                    print(f"  Error updating chunk {chroma_id}: {e}")
            
            print(f"  Updated batch {i//batch_size + 1}: {len(batch_ids)} chunks")
        
        print(f"Updated {total_updates} chunks with cluster labels")
    
    def display_cluster_assignments(self, clustering_results: Dict, chunks_data: List[Dict], 
                                   chroma_ids: List[str], metadatas: List[Dict]) -> Dict:
        """Display which chunks belong to which clusters"""
        labels = clustering_results['labels']
        
        print("\n" + "="*80)
        print("CLUSTER ASSIGNMENTS: Chunk ID to Cluster ID Mapping")
        print("="*80)
        
        cluster_to_chunks = {}
        
        for i, (label, metadata, chroma_id) in enumerate(zip(labels, metadatas, chroma_ids)):
            cluster_id = int(label)
            chunk_id = metadata['chunk_id']
            chunk_text = metadata['chunk_text_preview']
            confidence = float(max(clustering_results['probabilities'][i]))
            
            if cluster_id not in cluster_to_chunks:
                cluster_to_chunks[cluster_id] = []
            
            cluster_to_chunks[cluster_id].append({
                'chunk_id': chunk_id,
                'chroma_id': chroma_id,
                'text_preview': chunk_text,
                'confidence': confidence,
                'chunk_index': metadata['chunk_index'],
                'full_text': chunks_data[metadata['chunk_index']]['text'] if metadata['chunk_index'] < len(chunks_data) else "N/A",
                'file_type': metadata.get('file_type', 'unknown')
            })
        
        for cluster_id in cluster_to_chunks:
            cluster_to_chunks[cluster_id].sort(key=lambda x: x['chunk_index'])
        
        print("\n📊 CHUNK TO CLUSTER MAPPING:")
        print("-" * 80)
        
        for cluster_id in sorted(cluster_to_chunks.keys()):
            chunks_in_cluster = cluster_to_chunks[cluster_id]
            print(f"\n🔷 CLUSTER {cluster_id}")
            print(f"   Total chunks: {len(chunks_in_cluster)}")
            print(f"   Average confidence: {np.mean([c['confidence'] for c in chunks_in_cluster]):.3f}")
            print("\n   Chunks in this cluster:")
            print("   " + "-" * 70)
            
            for chunk in chunks_in_cluster[:10]:
                print(f"   • Chunk ID: {chunk['chunk_id']}")
                print(f"     Type: {chunk['file_type']}")
                print(f"     Preview: \"{chunk['text_preview']}...\"")
                print(f"     Confidence: {chunk['confidence']:.3f}")
                print(f"     Full text: {chunk['full_text'][:100]}")
                print()
            
            if len(chunks_in_cluster) > 10:
                print(f"   ... and {len(chunks_in_cluster) - 10} more chunks in this cluster")
        
        return cluster_to_chunks


# ========== MAIN EXECUTION ==========

def main():
    processor = DocumentProcessor(chroma_path="./document_db")
    
    sample_sty_file = "The Book of Five Rings.pdf"
    if not os.path.exists(sample_sty_file):
        with open(sample_sty_file, 'w', encoding='utf-8') as f:
            for i in range(100):
                f.write(f"""
% LaTeX style file section {i}
\\ProvidesPackage{{sample_package_{i}}}[2024/01/01 v1.0 Sample Package]
\\RequirePackage{{geometry}}
\\RequirePackage{{amsmath}}
\\newcommand{{\\samplecommand_{i}}}[1]{{\\textbf{{Sample {i}: #1}}}}
\\newcommand{{\\R}}{{\\mathbb{{R}}}}
""")
    
    print(f"Reading .sty file from: {sample_sty_file}")
    document_text = processor.read_document(sample_sty_file)
    print(f"Document length: {len(document_text)} characters")
    
    CHUNK_SIZE = 50
    OVERLAP = 5
    
    print(f"\nCreating chunks...")
    chunks = processor.create_chunks(
        text=document_text,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )
    
    print(f"Created {len(chunks)} chunks total")
    
    print("\n" + "="*80)
    print("ADDING CHUNKS TO CHROMADB")
    print("="*80)
    chunk_ids = processor.add_chunks_to_chromadb(
        chunks=chunks,
        source_file=sample_sty_file
    )
    
    print("\nRetrieving embeddings...")
    embeddings, documents, metadatas, chroma_ids = processor.get_all_embeddings_with_metadata()
    
    if embeddings is not None:
        print(f"Retrieved {len(embeddings)} embeddings")
        
        print("\n" + "="*80)
        print("PERFORMING GMM CLUSTERING")
        print("="*80)
        clustering_results = processor.perform_gmm_clustering(
            embeddings=embeddings,
            n_clusters=3
        )
        
        print("\nUpdating ChromaDB with cluster labels...")
        processor.update_chunks_with_clusters(
            clustering_results=clustering_results,
            chroma_ids=chroma_ids,
            metadatas=metadatas,
            documents=documents
        )
        
        cluster_mapping = processor.display_cluster_assignments(
            clustering_results=clustering_results,
            chunks_data=chunks,
            chroma_ids=chroma_ids,
            metadatas=metadatas
        )
        
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        mapping_table = []
        for i, (metadata, label) in enumerate(zip(metadatas, clustering_results['labels'])):
            mapping_table.append({
                'Chunk ID': metadata['chunk_id'],
                'Cluster ID': int(label),
                'Confidence': f"{float(max(clustering_results['probabilities'][i])):.3f}",
                'Preview': metadata['chunk_text_preview'][:50]
            })
        
        df_final = pd.DataFrame(mapping_table)
        print("\n", df_final.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
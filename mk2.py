import chromadb
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
from typing import List, Dict
import uuid
import pandas as pd
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class ImprovedDocumentProcessor:
    def __init__(self, chroma_path: str = "./chroma_db"):
        """Initialize ChromaDB client and collection"""
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # Use a new collection for clean data
        self.collection = self.client.get_or_create_collection(
            name="improved_document_chunks",
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_search": 200,
                    "max_neighbors": 32,
                    "ef_construction": 200
                }
            }
        )
        
        self.batch_size = 1000
        self.tracked_chunks = set()  # Track chunk IDs to avoid duplicates
    
    def read_document(self, file_path: str) -> str:
        """Read content from a document file - handles PDFs correctly"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt' or ext == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif ext == '.pdf':
            try:
                import PyPDF2
                print(f"Reading PDF file: {file_path}")
                text = ""
                with open(file_path, 'rb') as f:  # Note: 'rb' for binary mode
                    reader = PyPDF2.PdfReader(f)
                    print(f"PDF has {len(reader.pages)} pages")
                    
                    for page_num, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                            if (page_num + 1) % 10 == 0:  # Progress indicator
                                print(f"  Processed page {page_num + 1}/{len(reader.pages)}")
                
                print(f"Extracted {len(text)} characters from PDF")
                return text
                
            except ImportError:
                print("PyPDF2 not installed. Installing...")
                import subprocess
                subprocess.check_call(['pip', 'install', 'PyPDF2'])
                import PyPDF2
                # Retry after installation
                return self.read_document(file_path)
            except Exception as e:
                print(f"Error reading PDF: {e}")
                return ""
        
        elif ext == '.sty' or ext == '.tex':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return self.clean_latex_content_improved(content)
        
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported: .txt, .md, .pdf, .sty, .tex")
    
    def clean_latex_content_improved(self, content: str) -> str:
        """Improved LaTeX cleaning that preserves structure better"""
        # Remove comments
        content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)
        
        # Preserve section headers as important markers
        content = re.sub(r'\\section\{([^}]+)\}', r'\n\n### SECTION: \1 ###\n\n', content)
        content = re.sub(r'\\subsection\{([^}]+)\}', r'\n\n### SUBSECTION: \1 ###\n\n', content)
        
        # Keep meaningful LaTeX commands that indicate content
        content = re.sub(r'\\ProvidesPackage\{([^}]+)\}', r'PACKAGE: \1', content)
        content = re.sub(r'\\RequirePackage\{([^}]+)\}', r'REQUIRES: \1', content)
        content = re.sub(r'\\newcommand\{\\([^}]+)\}', r'COMMAND: \1', content)
        
        # Remove remaining formatting commands
        content = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?', '', content)
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n', '\n', content)
        content = content.strip()
        
        return content
    
    def create_smart_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """
        Create smarter chunks with larger size to preserve meaning
        For PDF books, use larger chunks to preserve paragraphs
        """
        chunks = []
        step = chunk_size - overlap
        
        # Try to split by paragraphs first for better context
        paragraphs = text.split('\n\n')
        
        if len(paragraphs) > 1 and len(text) > 10000:
            print(f"Splitting by paragraphs: {len(paragraphs)} paragraphs")
            current_chunk = ""
            chunk_index = 0
            
            for para in paragraphs:
                # If adding this paragraph would exceed chunk size
                if len(current_chunk) + len(para) > chunk_size and current_chunk:
                    chunk_id = f"chunk_{chunk_index:06d}"
                    
                    # Skip duplicates
                    if chunk_id not in self.tracked_chunks:
                        self.tracked_chunks.add(chunk_id)
                        chunks.append({
                            'text': current_chunk.strip(),
                            'chunk_id': chunk_id,
                            'chunk_index': chunk_index
                        })
                        chunk_index += 1
                        current_chunk = para
                    else:
                        current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            
            # Add the last chunk
            if current_chunk:
                chunk_id = f"chunk_{chunk_index:06d}"
                if chunk_id not in self.tracked_chunks:
                    self.tracked_chunks.add(chunk_id)
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_id': chunk_id,
                        'chunk_index': chunk_index
                    })
        else:
            # Simple sliding window chunking for smaller documents
            for i in range(0, len(text), step):
                chunk_text = text[i:i + chunk_size]
                if chunk_text.strip():
                    chunk_id = f"chunk_{i:06d}"
                    
                    if chunk_id not in self.tracked_chunks:
                        self.tracked_chunks.add(chunk_id)
                        chunks.append({
                            'text': chunk_text,
                            'chunk_id': chunk_id,
                            'position': i,
                            'chunk_index': len(chunks)
                        })
        
        print(f"Created {len(chunks)} unique chunks")
        return chunks
    
    def add_chunks_to_chromadb(self, chunks: List[Dict], source_file: str) -> List[str]:
        """Add chunks with duplicate checking"""
        all_ids = []
        
        print(f"Adding {len(chunks)} unique chunks to ChromaDB...")
        
        for batch_start in range(0, len(chunks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            batch_ids = []
            batch_documents = []
            batch_metadatas = []
            
            for chunk in batch_chunks:
                chroma_id = f"{source_file}_{chunk['chunk_id']}_{uuid.uuid4().hex[:8]}"
                batch_ids.append(chroma_id)
                batch_documents.append(chunk['text'])
                
                metadata = {
                    'source_file': os.path.basename(source_file),
                    'chunk_id': chunk['chunk_id'],
                    'chunk_index': chunk['chunk_index'],
                    'chunk_preview': chunk['text'][:50].replace('\n', ' ')
                }
                
                batch_metadatas.append(metadata)
            
            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                all_ids.extend(batch_ids)
                print(f"  Added batch {batch_start//self.batch_size + 1}: {len(batch_ids)} chunks")
            except Exception as e:
                print(f"  Error adding batch: {e}")
                # Try smaller batch
                for chunk in batch_chunks:
                    try:
                        chroma_id = f"{source_file}_{chunk['chunk_id']}_{uuid.uuid4().hex[:8]}"
                        self.collection.add(
                            ids=[chroma_id],
                            documents=[chunk['text']],
                            metadatas=[{
                                'source_file': os.path.basename(source_file),
                                'chunk_id': chunk['chunk_id'],
                                'chunk_index': chunk['chunk_index'],
                                'chunk_preview': chunk['text'][:50].replace('\n', ' ')
                            }]
                        )
                        all_ids.append(chroma_id)
                    except Exception as sub_e:
                        print(f"    Error adding chunk {chunk['chunk_id']}: {sub_e}")
        
        return all_ids
    
    def get_all_embeddings_with_metadata(self) -> tuple:
        """Retrieve all embeddings and metadata"""
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
                
                if result.get('embeddings') is not None and len(result['embeddings']) > 0:
                    all_embeddings.extend(result['embeddings'])
                    all_documents.extend(result['documents'])
                    all_metadatas.extend(result['metadatas'])
                    all_ids.extend(result['ids'])
                    print(f"  Retrieved batch {offset//batch_size + 1}: {len(result['ids'])} items")
                    
            except Exception as e:
                print(f"  Error retrieving batch: {e}")
                continue
        
        if len(all_embeddings) == 0:
            print("No embeddings found")
            return None, None, None, None
        
        embeddings = np.array(all_embeddings)
        print(f"Total retrieved: {len(embeddings)} embeddings")
        
        return embeddings, all_documents, all_metadatas, all_ids
    
    def perform_gmm_clustering(self, embeddings: np.ndarray, n_clusters: int = 5) -> Dict:
        """Perform GMM clustering"""
        print(f"Preprocessing {len(embeddings)} embeddings...")
        
        # Reduce dimensions with PCA
        if embeddings.shape[1] > 100:
            print(f"Reducing dimensions from {embeddings.shape[1]}...")
            pca = PCA(n_components=min(100, embeddings.shape[0] // 2))
            embeddings_reduced = pca.fit_transform(embeddings)
            print(f"Reduced to {embeddings_reduced.shape[1]} dimensions")
            print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        else:
            embeddings_reduced = embeddings
        
        # Standardize
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_reduced)
        
        print(f"Running GMM with {n_clusters} clusters...")
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='diag',
            random_state=42,
            n_init=5,
            max_iter=200,
            verbose=0
        )
        
        cluster_labels = gmm.fit_predict(embeddings_scaled)
        cluster_probs = gmm.predict_proba(embeddings_scaled)
        
        return {
            'labels': cluster_labels,
            'probabilities': cluster_probs,
            'n_clusters': n_clusters,
            'model': gmm
        }
    
    def display_cluster_results(self, clustering_results: Dict, metadatas: List[Dict]) -> None:
        """Display cluster results"""
        labels = clustering_results['labels']
        
        print("\n" + "="*80)
        print("CLUSTER ASSIGNMENTS")
        print("="*80)
        
        # Group by cluster
        cluster_to_chunks = {}
        
        for i, (label, metadata) in enumerate(zip(labels, metadatas)):
            cluster_id = int(label)
            chunk_id = metadata['chunk_id']
            
            if cluster_id not in cluster_to_chunks:
                cluster_to_chunks[cluster_id] = []
            
            # Avoid duplicates in display
            if chunk_id not in [c['chunk_id'] for c in cluster_to_chunks[cluster_id]]:
                cluster_to_chunks[cluster_id].append({
                    'chunk_id': chunk_id,
                    'preview': metadata['chunk_preview'],
                    'confidence': float(max(clustering_results['probabilities'][i]))
                })
        
        # Display statistics
        for cluster_id in sorted(cluster_to_chunks.keys()):
            chunks = cluster_to_chunks[cluster_id]
            print(f"\n🔷 CLUSTER {cluster_id}")
            print(f"   Total unique chunks: {len(chunks)}")
            
            confidences = [c['confidence'] for c in chunks]
            print(f"   Avg confidence: {np.mean(confidences):.3f}")
            print(f"   Min confidence: {np.min(confidences):.3f}")
            print(f"   Max confidence: {np.max(confidences):.3f}")
            
            print("\n   Sample chunks:")
            for chunk in chunks[:3]:
                print(f"   • {chunk['chunk_id']}")
                print(f"     \"{chunk['preview']}...\"")
                print(f"     Confidence: {chunk['confidence']:.3f}")
                print()
    
    def generate_cluster_report(self, clustering_results: Dict, metadatas: List[Dict]) -> pd.DataFrame:
        """Generate cluster report"""
        labels = clustering_results['labels']
        
        data = []
        for i, (label, metadata) in enumerate(zip(labels, metadatas)):
            data.append({
                'Chunk ID': metadata['chunk_id'],
                'Cluster': int(label),
                'Confidence': float(max(clustering_results['probabilities'][i])),
                'Preview': metadata['chunk_preview'][:40]
            })
        
        df = pd.DataFrame(data)
        
        print("\n" + "="*80)
        print("CLUSTER STATISTICS")
        print("="*80)
        summary = df.groupby('Cluster').agg({
            'Chunk ID': 'count',
            'Confidence': ['mean', 'min', 'max']
        }).round(3)
        print(summary)
        
        return df

# ========== MAIN EXECUTION ==========

def main():
    # Set environment variable
    os.environ['LOKY_MAX_CPU_COUNT'] = '4'
    
    # Initialize processor
    processor = ImprovedDocumentProcessor(chroma_path="./pdf_chroma_db")
    
    # Specify your PDF file
    pdf_file = "sample.txt"
    
    # Check if file exists
    if not os.path.exists(pdf_file):
        print(f"Error: File '{pdf_file}' not found!")
        print("Please make sure the PDF file is in the current directory.")
        return
    
    # Step 1: Read the PDF document
    print("="*80)
    print("STEP 1: READING PDF DOCUMENT")
    print("="*80)
    document_text = processor.read_document(pdf_file)
    
    if not document_text:
        print("Failed to extract text from PDF. Please check the file.")
        return
    
    print(f"\nDocument length: {len(document_text):,} characters")
    print(f"Sample of extracted text:")
    print("-" * 80)
    print(document_text[:500])
    print("-" * 80)
    
    # Step 2: Create chunks
    print("\n" + "="*80)
    print("STEP 2: CREATING CHUNKS")
    print("="*80)
    CHUNK_SIZE = 500  # Characters per chunk
    OVERLAP = 100      # Overlap between chunks
    
    chunks = processor.create_smart_chunks(
        text=document_text,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )
    
    print(f"\nCreated {len(chunks)} chunks")
    
    # Step 3: Add chunks to ChromaDB
    print("\n" + "="*80)
    print("STEP 3: ADDING CHUNKS TO CHROMADB")
    print("="*80)
    chunk_ids = processor.add_chunks_to_chromadb(
        chunks=chunks,
        source_file=pdf_file
    )
    
    # Step 4: Retrieve embeddings
    print("\n" + "="*80)
    print("STEP 4: RETRIEVING EMBEDDINGS")
    print("="*80)
    embeddings, documents, metadatas, ids = processor.get_all_embeddings_with_metadata()
    
    if embeddings is not None:
        # Step 5: Perform clustering
        print("\n" + "="*80)
        print("STEP 5: GMM CLUSTERING")
        print("="*80)
        
        # Try different numbers of clusters based on document size
        n_clusters = min(10, max(3, len(embeddings) // 500))
        print(f"Using {n_clusters} clusters for {len(embeddings)} chunks")
        
        clustering_results = processor.perform_gmm_clustering(
            embeddings=embeddings,
            n_clusters=5
        )
        
        # Step 6: Display results
        processor.display_cluster_results(clustering_results, metadatas)
        
        # Step 7: Generate report
        df_report = processor.generate_cluster_report(clustering_results, metadatas)
        
        # Step 8: Save report
        report_file = "cluster_report.csv"
        df_report.to_csv(report_file, index=False)
        print(f"\n✓ Report saved to {report_file}")
        
        # Step 9: Show cluster distribution
        print("\n" + "="*80)
        print("CLUSTER DISTRIBUTION")
        print("="*80)
        cluster_counts = df_report['Cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            percentage = (count / len(df_report)) * 100
            bar = "█" * int(percentage / 2)
            print(f"Cluster {cluster}: {count:5d} chunks ({percentage:5.1f}%) {bar}")
    
    else:
        print("No embeddings retrieved!")

if __name__ == "__main__":
    main()
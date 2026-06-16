import os
import sqlite3
import pickle
import json
from pathlib import Path
import chromadb
from chromadb.config import Settings

def inspect_chroma_db(persist_directory="./hierarchical_chroma_db"):
    """Comprehensive inspection of ChromaDB structure"""
    
    print("=" * 80)
    print(f"CHROMA DB INSPECTION: {persist_directory}")
    print("=" * 80)
    
    # ============================================
    # 1. CHECK DIRECTORY STRUCTURE
    # ============================================
    print("\n📁 1. DIRECTORY STRUCTURE")
    print("-" * 80)
    
    if not os.path.exists(persist_directory):
        print(f"❌ Directory does not exist: {persist_directory}")
        return
    
    # List all items in the directory
    items = os.listdir(persist_directory)
    print(f"Found {len(items)} items in {persist_directory}:")
    
    for item in items:
        item_path = os.path.join(persist_directory, item)
        if os.path.isdir(item_path):
            print(f"  📂 {item}/")
            # Show contents of subdirectories
            subitems = os.listdir(item_path)
            for subitem in subitems:
                file_size = os.path.getsize(os.path.join(item_path, subitem))
                print(f"      📄 {subitem} ({file_size:,} bytes)")
        else:
            file_size = os.path.getsize(item_path)
            print(f"  📄 {item} ({file_size:,} bytes)")
    
    # ============================================
    # 2. EXAMINE SQLITE DATABASE
    # ============================================
    sqlite_file = os.path.join(persist_directory, "chroma.sqlite3")
    
    if os.path.exists(sqlite_file):
        print("\n🗄️  2. SQLITE DATABASE STRUCTURE")
        print("-" * 80)
        
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables found: {', '.join([t[0] for t in tables])}")
        
        # Inspect each table
        for table in tables:
            table_name = table[0]
            print(f"\n  📊 Table: {table_name}")
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print(f"     Columns: {', '.join([col[1] for col in columns])}")
            
            # Count rows
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            print(f"     Rows: {row_count}")
            
            # Show first few rows for main tables
            if row_count > 0 and table_name in ['collections', 'embeddings', 'metadata']:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample = cursor.fetchall()
                print(f"     Sample data:")
                for row in sample:
                    print(f"       {row}")
        
        # Check embeddings table specifically (most important)
        cursor.execute("SELECT name FROM sqlite_master WHERE name='embeddings'")
        if cursor.fetchone():
            print(f"\n  🔍 Detailed embeddings inspection:")
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            total_embeddings = cursor.fetchone()[0]
            print(f"     Total embeddings: {total_embeddings}")
            
            if total_embeddings > 0:
                cursor.execute("SELECT id, collection_id, metadata FROM embeddings LIMIT 3")
                for row in cursor.fetchall():
                    print(f"     ID: {row[0]}, Collection: {row[1]}, Metadata: {row[2]}")
        
        conn.close()
    else:
        print("\n❌ No chroma.sqlite3 file found")
    
    # ============================================
    # 3. INSPECT HNSW INDEX FILES
    # ============================================
    print("\n🎯 3. HNSW INDEX FILES")
    print("-" * 80)
    
    for item in os.listdir(persist_directory):
        item_path = os.path.join(persist_directory, item)
        if os.path.isdir(item_path):
            # Try to parse UUID format
            try:
                import uuid
                uuid_obj = uuid.UUID(item)
                print(f"\n  Collection UUID: {item}")
                
                # Check pickle files
                pickle_file = os.path.join(item_path, "index_metadata.pickle")
                if os.path.exists(pickle_file):
                    with open(pickle_file, 'rb') as f:
                        metadata = pickle.load(f)
                    print(f"    📦 Pickle metadata contents:")
                    for key, value in metadata.items():
                        if isinstance(value, dict):
                            print(f"       {key}: {len(value)} items")
                        else:
                            print(f"       {key}: {value}")
                
                # List all bin files with sizes
                bin_files = [f for f in os.listdir(item_path) if f.endswith('.bin')]
                if bin_files:
                    print(f"    💾 Index files:")
                    for bin_file in bin_files:
                        size = os.path.getsize(os.path.join(item_path, bin_file))
                        print(f"       {bin_file}: {size:,} bytes")
                        
            except ValueError:
                print(f"\n  Non-UUID directory: {item}/")
    
    # ============================================
    # 4. CONNECT VIA CHROMADB CLIENT
    # ============================================
    print("\n🔌 4. CHROMADB CLIENT INSPECTION")
    print("-" * 80)
    
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        
        # List all collections
        collections = client.list_collections()
        print(f"Collections found via API: {len(collections)}")
        
        for collection in collections:
            print(f"\n  📚 Collection: {collection.name}")
            print(f"     ID: {collection.id}")
            print(f"     Metadata: {collection.metadata}")
            
            # Get collection count
            count = collection.count()
            print(f"     Item count: {count}")
            
            # Sample first few items if any exist
            if count > 0:
                try:
                    # Get a sample of items
                    results = collection.get(limit=min(3, count))
                    print(f"     Sample items:")
                    
                    if results['ids']:
                        for i, doc_id in enumerate(results['ids']):
                            print(f"       ID: {doc_id}")
                            if results['documents'] and i < len(results['documents']):
                                doc_preview = results['documents'][i][:100] + "..." if len(results['documents'][i]) > 100 else results['documents'][i]
                                print(f"         Document: {doc_preview}")
                            if results['metadatas'] and i < len(results['metadatas']):
                                print(f"         Metadata: {results['metadatas'][i]}")
                except Exception as e:
                    print(f"     Could not sample items: {e}")
                    
    except Exception as e:
        print(f"❌ Could not connect via ChromaDB: {e}")
    
    # ============================================
    # 5. FILE SIZE SUMMARY
    # ============================================
    print("\n📊 5. STORAGE SUMMARY")
    print("-" * 80)
    
    def get_size(path):
        total = 0
        if os.path.isfile(path):
            return os.path.getsize(path)
        elif os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total += os.path.getsize(fp)
        return total
    
    total_size = get_size(persist_directory)
    print(f"Total storage used: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
    
    # Breakdown by type
    sqlite_size = get_size(sqlite_file) if os.path.exists(sqlite_file) else 0
    print(f"  chroma.sqlite3: {sqlite_size:,} bytes ({sqlite_size / 1024:.2f} KB)")
    
    for item in os.listdir(persist_directory):
        item_path = os.path.join(persist_directory, item)
        if os.path.isdir(item_path):
            dir_size = get_size(item_path)
            print(f"  {item}/: {dir_size:,} bytes ({dir_size / 1024:.2f} KB)")
    
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    # Default is now hierarchical_chroma_db
    CHROMA_PATH = "./hierarchical_chroma_db"
    
    # You can also pass as command line argument
    import sys
    if len(sys.argv) > 1:
        CHROMA_PATH = sys.argv[1]
    
    inspect_chroma_db(CHROMA_PATH)
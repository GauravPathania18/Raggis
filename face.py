import requests
import pandas as pd
from typing import List, Dict, Optional
import time
from datetime import datetime
import json
import os
import sys

# Import backend from mk3.py
try:
    from mk3 import HierarchicalClusterSummarizer
except ImportError:
    print("❌ Error: Could not import from mk3.py")
    print("   Make sure mk3.py is in the same directory")
    sys.exit(1)


class RAGInterface:
    """
    Conversational RAG Interface using mk3.py for retrieval and Ollama for generation
    """
    
    def __init__(self, 
                 chroma_path: str = "./hierarchical_chroma_db",  # FIXED: matches mk3.py
                 ollama_model: str = "gemma3:1b",
                 ollama_url: str = "http://localhost:11434/api/generate",
                 max_context_chunks: int = 5,
                 include_summaries: bool = True,
                 temperature: float = 0.3,
                 verbose: bool = True):
        """
        Initialize the RAG Interface
        
        Args:
            chroma_path: Path to ChromaDB storage (must match mk3.py)
            ollama_model: Ollama model name
            ollama_url: Ollama API endpoint
            max_context_chunks: Number of chunks to retrieve
            include_summaries: Whether to include cluster summaries
            temperature: LLM temperature (0-1)
            verbose: Print detailed information
        """
        self.verbose = verbose
        
        # Initialize the mk3 processor (this gives us all the retrieval capabilities)
        self.processor = HierarchicalClusterSummarizer(
            chroma_path=chroma_path,
            ollama_model=ollama_model,
            ollama_url=ollama_url
        )
        
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.max_context_chunks = max_context_chunks
        self.include_summaries = include_summaries
        self.temperature = temperature
        self.conversation_history = []
        self.current_session = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check connections
        self._check_ollama()
        self._show_database_info()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                if any(self.ollama_model in name for name in model_names):
                    if self.verbose:
                        print(f"✅ Ollama is running with model: {self.ollama_model}")
                    return True
                else:
                    print(f"⚠️ Model {self.ollama_model} not found. Run: ollama pull {self.ollama_model}")
                    return False
            return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("   Please start Ollama with: ollama serve")
            return False
    
    def _show_database_info(self):
        """Show information about the database"""
        try:
            total_chunks = self.processor.chunks_collection.count()
            total_summaries = self.processor.summaries_collection.count()
            
            if self.verbose:
                print(f"📚 Database: {total_chunks} chunks, {total_summaries} summaries")
                
                # Show source distribution
                if total_chunks > 0:
                    result = self.processor.chunks_collection.get(include=["metadatas"], limit=10000)
                    sources = {}
                    for meta in result['metadatas']:
                        source = meta.get('source_file', 'Unknown')
                        sources[source] = sources.get(source, 0) + 1
                    
                    print(f"📄 Sources: {len(sources)} document(s)")
                    for source, count in sources.items():
                        print(f"   • {source}: {count} chunks")
                else:
                    print("   ⚠️ Database is empty! Run mk3.py first to add documents.")
                    
        except Exception as e:
            print(f"⚠️ Could not get database info: {e}")
    
    def retrieve_context(self, query: str) -> Dict:
        """
        Retrieve relevant context from ChromaDB using the mk3 processor's collections
        
        Args:
            query: User question
            
        Returns:
            Dictionary with retrieved chunks, metadata, and scores
        """
        if self.verbose:
            print(f"\n🔍 Retrieving context for: {query[:80]}...")
        
        try:
            # Use mk3's chunks_collection directly for HNSW search
            results = self.processor.chunks_collection.query(
                query_texts=[query],
                n_results=self.max_context_chunks,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents'] or not results['documents'][0]:
                return {'success': False, 'message': 'No relevant documents found'}
            
            retrieved_chunks = results['documents'][0]
            retrieved_metadatas = results['metadatas'][0]
            retrieved_distances = results['distances'][0]
            
            # Calculate relevance scores (cosine distance to similarity)
            relevance_scores = [1 - d for d in retrieved_distances]
            
            # Get cluster summaries if requested (using mk3's summaries_collection)
            cluster_summaries = []
            if self.include_summaries:
                clusters = set()
                for meta in retrieved_metadatas:
                    if meta and 'cluster_id' in meta:
                        clusters.add(meta['cluster_id'])
                
                for cluster_id in clusters:
                    try:
                        # Use mk3's summaries_collection directly
                        summary_results = self.processor.summaries_collection.get(
                            where={"cluster_id": cluster_id},
                            limit=1,
                            include=["documents", "metadatas"]
                        )
                        if summary_results['documents']:
                            summary_text = summary_results['documents'][0]
                            summary_meta = summary_results['metadatas'][0]
                            cluster_summaries.append({
                                'cluster_id': cluster_id,
                                'level': summary_meta.get('level', 0),
                                'chunks_count': summary_meta.get('chunks_count', 0),
                                'summary': summary_text[:500]
                            })
                    except Exception as e:
                        if self.verbose:
                            print(f"  Could not fetch summary for cluster {cluster_id}: {e}")
            
            # Extract unique sources
            sources = []
            for meta in retrieved_metadatas:
                if meta and 'source_file' in meta:
                    sources.append(meta['source_file'])
            sources = list(set(sources))
            
            return {
                'success': True,
                'chunks': retrieved_chunks,
                'metadatas': retrieved_metadatas,
                'scores': relevance_scores,
                'cluster_summaries': cluster_summaries,
                'sources': sources,
                'query': query
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Error retrieving context: {str(e)}'}
    
    def build_prompt(self, context: Dict) -> str:
        """
        Build a prompt for the LLM using retrieved context
        
        Args:
            context: Retrieved context dictionary from retrieve_context()
            
        Returns:
            Formatted prompt
        """
        prompt_parts = []
        
        # System instruction
        prompt_parts.append("""You are a knowledgeable assistant. Answer the question based ONLY on the provided context.

INSTRUCTIONS:
- Answer based only on the information in the context
- If the context doesn't contain the answer, say "I don't have enough information to answer that"
- Be concise but thorough
- When quoting sources, mention the document name
- Use a helpful, friendly tone
""")
        
        # Add source information
        if context.get('sources'):
            prompt_parts.append(f"\n📚 SOURCES: {', '.join(context['sources'])}")
        
        # Add cluster summaries if available
        if context.get('cluster_summaries'):
            prompt_parts.append("\n📊 HIGH-LEVEL TOPIC SUMMARIES:")
            for summary_info in context['cluster_summaries']:
                prompt_parts.append(f"\nTopic {summary_info['cluster_id']} (Level {summary_info['level']}, {summary_info['chunks_count']} chunks):")
                prompt_parts.append(f"{summary_info['summary']}")
        
        # Add relevant chunks with metadata
        prompt_parts.append("\n📄 RELEVANT CONTENT:")
        for i, (chunk, score, meta) in enumerate(zip(
            context['chunks'], 
            context['scores'], 
            context['metadatas']
        )):
            source = meta.get('source_file', 'Unknown') if meta else 'Unknown'
            chunk_id = meta.get('chunk_id', 'Unknown') if meta else 'Unknown'
            prompt_parts.append(f"\n[{i+1}] (Source: {source}, Relevance: {score:.2%})")
            prompt_parts.append(f"{chunk}")
        
        # Add conversation history (last 3 exchanges)
        if self.conversation_history:
            prompt_parts.append("\n💬 CONVERSATION HISTORY:")
            for turn in self.conversation_history[-3:]:
                prompt_parts.append(f"Q: {turn['question']}")
                prompt_parts.append(f"A: {turn['answer'][:200]}...")
        
        # Add the current question
        prompt_parts.append(f"\n❓ QUESTION: {context['query']}")
        prompt_parts.append("\n✅ ANSWER:")
        
        return "\n".join(prompt_parts)
    
    def generate_response(self, prompt: str) -> Dict:
        """
        Generate response using Ollama
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Dictionary with response and metadata
        """
        if self.verbose:
            print("🤖 Generating response...")
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": -1,
                        "top_k": 40,
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                return {
                    'success': True,
                    'answer': answer,
                    'tokens': len(answer.split()),
                    'model': self.ollama_model
                }
            else:
                return {
                    'success': False,
                    'error': f"API error: {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            return {'success': False, 'error': "Request timeout"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def query(self, question: str) -> Dict:
        """
        Complete RAG query: Retrieve using mk3, Augment, Generate
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        print("\n" + "="*80)
        print(f"📝 Question: {question}")
        print("="*80)
        
        # Step 1: Retrieve context using mk3's HNSW search
        context = self.retrieve_context(question)
        
        if not context.get('success'):
            return {
                'question': question,
                'answer': f"Error: {context.get('message', 'Unknown error')}",
                'sources': [],
                'context_chunks': 0,
                'response_time': time.time() - start_time
            }
        
        # Step 2: Build prompt with context
        prompt = self.build_prompt(context)
        
        # Step 3: Generate response
        response = self.generate_response(prompt)
        
        elapsed_time = time.time() - start_time
        
        if response.get('success'):
            answer = response['answer']
            
            # Save to conversation history
            self.conversation_history.append({
                'question': question,
                'answer': answer,
                'timestamp': datetime.now().isoformat(),
                'sources': context.get('sources', [])
            })
            
            # Display results
            print(f"\n✅ Response ({elapsed_time:.2f}s, {response.get('tokens', 0)} tokens):")
            print("-"*80)
            print(answer)
            print("-"*80)
            
            if context.get('sources'):
                print(f"\n📚 Sources: {', '.join(context['sources'])}")
            
            if self.verbose:
                print(f"\n📊 Retrieved {len(context.get('chunks', []))} chunks with scores:")
                for i, (chunk, score, meta) in enumerate(zip(
                    context.get('chunks', [])[:3],
                    context.get('scores', [])[:3],
                    context.get('metadatas', [])[:3]
                )):
                    print(f"   [{i+1}] {score:.2%} - {meta.get('source_file', 'Unknown')[:40]}")
            
            return {
                'question': question,
                'answer': answer,
                'sources': context.get('sources', []),
                'context_chunks': len(context.get('chunks', [])),
                'relevance_scores': context.get('scores', []),
                'response_time': elapsed_time,
                'tokens_generated': response.get('tokens', 0),
                'conversation_turns': len(self.conversation_history)
            }
        else:
            return {
                'question': question,
                'answer': f"Error: {response.get('error', 'Unknown error')}",
                'sources': [],
                'context_chunks': 0,
                'response_time': elapsed_time
            }
    
    def interactive_mode(self):
        """Run interactive conversation mode"""
        print("\n" + "="*80)
        print("🎯 FACE - RAG Interface (Frontend for Hierarchical Knowledge Base)")
        print("="*80)
        print(f"📚 Database: {self.processor.chunks_collection.count()} chunks total")
        print(f"🤖 Model: {self.ollama_model}")
        print(f"📊 Context: {self.max_context_chunks} chunks, Summaries: {self.include_summaries}")
        print("\nCommands:")
        print("  /clear    - Clear conversation history")
        print("  /stats    - Show database statistics")
        print("  /sources  - Show all documents in database")
        print("  /clusters - Show cluster summaries")
        print("  /help     - Show this help")
        print("  /quit     - Exit")
        print("="*80)
        
        while True:
            try:
                user_input = input("\n❓ You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == '/quit':
                    print("\n👋 Goodbye!")
                    break
                    
                elif user_input.lower() == '/clear':
                    self.conversation_history = []
                    print("✅ Conversation history cleared")
                    continue
                    
                elif user_input.lower() == '/stats':
                    self.show_stats()
                    continue
                    
                elif user_input.lower() == '/sources':
                    self.show_sources()
                    continue
                    
                elif user_input.lower() == '/clusters':
                    self.show_clusters()
                    continue
                    
                elif user_input.lower() == '/help':
                    self.show_help()
                    continue
                
                # Process query
                result = self.query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def show_stats(self):
        """Show database statistics"""
        try:
            total_chunks = self.processor.chunks_collection.count()
            total_summaries = self.processor.summaries_collection.count()
            
            print("\n📊 DATABASE STATISTICS")
            print("="*60)
            print(f"Total chunks: {total_chunks}")
            print(f"Total summaries: {total_summaries}")
            print(f"Session queries: {len(self.conversation_history)}")
            
            # Show source distribution
            if total_chunks > 0:
                result = self.processor.chunks_collection.get(include=["metadatas"], limit=10000)
                sources = {}
                for meta in result['metadatas']:
                    source = meta.get('source_file', 'Unknown')
                    sources[source] = sources.get(source, 0) + 1
                
                print(f"\nDocument sources:")
                for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                    print(f"  • {source}: {count} chunks")
                
        except Exception as e:
            print(f"Error getting stats: {e}")
    
    def show_sources(self):
        """Show all documents in the database"""
        try:
            result = self.processor.chunks_collection.get(include=["metadatas"], limit=10000)
            sources = {}
            for meta in result['metadatas']:
                source = meta.get('source_file', 'Unknown')
                sources[source] = sources.get(source, 0) + 1
            
            print("\n📚 DOCUMENTS IN DATABASE")
            print("="*60)
            for source, count in sorted(sources.items()):
                print(f"  • {source}: {count} chunks")
                
        except Exception as e:
            print(f"Error getting sources: {e}")
    
    def show_clusters(self):
        """Show cluster summaries"""
        try:
            df = self.processor.get_summary_tree()
            if df.empty:
                print("\n📊 No cluster summaries found")
                return
            
            print("\n📊 CLUSTER SUMMARIES")
            print("="*60)
            
            for level in sorted(df['level'].unique()):
                level_df = df[df['level'] == level]
                print(f"\nLevel {level} ({len(level_df)} clusters):")
                print("-"*40)
                
                for _, row in level_df.iterrows():
                    print(f"\n  Cluster {row['cluster_id']}:")
                    print(f"    Chunks: {row['chunks_count']}")
                    print(f"    Silhouette: {row['silhouette_score']:.3f}")
                    print(f"    Preview: {row['summary_preview'][:100]}...")
                    
        except Exception as e:
            print(f"Error getting clusters: {e}")
    
    def show_help(self):
        """Show help information"""
        print("\n📖 HELP")
        print("="*60)
        print("Commands:")
        print("  /clear    - Clear conversation history")
        print("  /stats    - Show database statistics")
        print("  /sources  - Show all documents in database")
        print("  /clusters - Show cluster summaries")
        print("  /help     - Show this help")
        print("  /quit     - Exit")
        print("\nTips:")
        print("  • Ask questions about your documents")
        print("  • The system retrieves relevant chunks using HNSW search")
        print("  • Responses include source attribution")
        print("  • Conversation history provides context for follow-up questions")
    
    def batch_query(self, questions: List[str], output_file: str = None) -> List[Dict]:
        """
        Process multiple questions in batch
        
        Args:
            questions: List of questions
            output_file: Optional file to save results
            
        Returns:
            List of results
        """
        results = []
        for i, question in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] Processing...")
            result = self.query(question)
            results.append(result)
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.5)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Results saved to {output_file}")
        
        return results
    
    def export_conversation(self, filename: str = None):
        """Export conversation history to file"""
        if not self.conversation_history:
            print("No conversation history to export")
            return
        
        if filename is None:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'session': self.current_session,
            'model': self.ollama_model,
            'total_queries': len(self.conversation_history),
            'conversation': self.conversation_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Conversation exported to {filename}")


# ========== MAIN EXECUTION ==========

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FACE - RAG Interface for Hierarchical Knowledge Base')
    parser.add_argument('--db-path', 
                       default='./hierarchical_chroma_db',  # FIXED: matches mk3.py
                       help='Path to ChromaDB (must match mk3.py - default: ./hierarchical_chroma_db)')
    parser.add_argument('--model', default='gemma3:1b', help='Ollama model name')
    parser.add_argument('--chunks', type=int, default=5, help='Number of chunks to retrieve')
    parser.add_argument('--no-summaries', action='store_true', help='Disable cluster summaries')
    parser.add_argument('--temperature', type=float, default=0.3, help='LLM temperature')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--query', type=str, help='Single query (non-interactive mode)')
    parser.add_argument('--batch', type=str, help='File with list of questions (one per line)')
    parser.add_argument('--export', type=str, help='Export conversation to file')
    
    args = parser.parse_args()
    
    # Check if database path exists
    if not os.path.exists(args.db_path):
        print(f"\n⚠️ Database path does not exist: {args.db_path}")
        print("   Please run mk3.py first to create the database:")
        print("   python mk3.py")
        print(f"\n   Or specify a different path with --db-path")
        sys.exit(1)
    
    # Initialize interface
    interface = RAGInterface(
        chroma_path=args.db_path,
        ollama_model=args.model,
        max_context_chunks=args.chunks,
        include_summaries=not args.no_summaries,
        temperature=args.temperature,
        verbose=not args.quiet
    )
    
    # Check if database has content
    total_chunks = interface.processor.chunks_collection.count()
    if total_chunks == 0:
        print("\n⚠️ Database is empty!")
        print("Please add documents first using mk3.py")
        print("Example: python mk3.py")
        return
    
    # Handle different modes
    if args.query:
        # Single query mode
        result = interface.query(args.query)
        print(f"\n{result['answer']}")
        
    elif args.batch:
        # Batch query mode
        if not os.path.exists(args.batch):
            print(f"\n❌ Batch file not found: {args.batch}")
            return
            
        with open(args.batch, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(questions)} questions...")
        results = interface.batch_query(questions)
        
        # Export results
        output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        interface.batch_query(questions, output_file)
        
    else:
        # Interactive mode
        interface.interactive_mode()
        
        # Export conversation if requested
        if args.export:
            interface.export_conversation(args.export)


if __name__ == "__main__":
    main()
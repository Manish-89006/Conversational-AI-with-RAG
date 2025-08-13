#!/usr/bin/env python3
"""
Demo script for the Conversational AI with RAG system.
"""
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_pipeline import rag_pipeline
from src.document_processor import document_processor
from src.llm_manager import llm_manager
from src.vector_store import vector_store

def demo_document_processing():
    """Demonstrate document processing capabilities."""
    print("=== Document Processing Demo ===\n")
    
    # Process sample documents
    sample_docs = [
        "examples/sample_documents.txt",
        "examples/rag_system.md"
    ]
    
    documents = []
    for doc_path in sample_docs:
        if os.path.exists(doc_path):
            doc = document_processor.process_file(doc_path)
            if doc:
                documents.append(doc)
                print(f"✓ Processed: {doc_path}")
                print(f"  - Content length: {len(doc['text'])} characters")
                print(f"  - File type: {doc['file_type']}")
                print()
    
    if documents:
        print(f"Total documents processed: {len(documents)}")
        return documents
    else:
        print("No documents were processed successfully.")
        return []

def demo_rag_pipeline(documents):
    """Demonstrate RAG pipeline capabilities."""
    print("=== RAG Pipeline Demo ===\n")
    
    if not documents:
        print("No documents to process. Skipping RAG demo.")
        return
    
    # Process documents through RAG pipeline
    print("Processing documents through RAG pipeline...")
    success = rag_pipeline.process_documents(documents)
    
    if success:
        print("✓ Documents successfully processed and stored in vector database")
        print()
        
        # Get pipeline info
        info = rag_pipeline.get_pipeline_info()
        print("Pipeline Information:")
        print(f"  - Chunk size: {info['chunk_size']}")
        print(f"  - Chunk overlap: {info['chunk_overlap']}")
        print(f"  - Top-K retrieval: {info['top_k_retrieval']}")
        print(f"  - Embedding model: {info['embedding_model']}")
        print(f"  - Vector store documents: {info['vector_store_info'].get('document_count', 0)}")
        print()
    else:
        print("✗ Failed to process documents through RAG pipeline")
        return

def demo_question_answering():
    """Demonstrate question answering capabilities."""
    print("=== Question Answering Demo ===\n")
    
    # Sample questions
    questions = [
        "What is artificial intelligence?",
        "How does RAG work?",
        "What are the benefits of machine learning?",
        "What is supervised learning?",
        "What are the challenges of RAG systems?"
    ]
    
    print("Sample questions and answers:\n")
    
    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")
        
        try:
            # Generate RAG-enhanced response
            response = rag_pipeline.generate_response(question)
            print(f"A{i}: {response}")
        except Exception as e:
            print(f"A{i}: Error generating response: {e}")
        
        print("-" * 80)
        print()

def demo_llm_providers():
    """Demonstrate LLM provider capabilities."""
    print("=== LLM Providers Demo ===\n")
    
    # Get available providers
    providers = llm_manager.get_available_providers()
    active_provider = llm_manager.active_provider
    
    print(f"Available LLM providers: {providers}")
    print(f"Active provider: {active_provider}")
    print()
    
    if providers:
        for provider in providers:
            info = llm_manager.get_provider_info(provider)
            print(f"Provider: {provider}")
            print(f"  - Model: {info.get('model', 'Unknown')}")
            print(f"  - Type: {info.get('type', 'Unknown')}")
            print(f"  - Capabilities: {', '.join(info.get('capabilities', []))}")
            print()
    else:
        print("No LLM providers available. Please check your configuration.")

def main():
    """Run the complete demo."""
    print("🚀 Conversational AI with RAG - Demo\n")
    print("This demo showcases the system's capabilities:\n")
    
    try:
        # Check system status
        print("Checking system status...")
        if not llm_manager.get_available_providers():
            print("⚠️  Warning: No LLM providers configured. Some features may not work.")
            print("   Please set up your API keys in the environment variables.\n")
        
        # Run demos
        documents = demo_document_processing()
        
        if documents:
            demo_rag_pipeline(documents)
            demo_question_answering()
        
        demo_llm_providers()
        
        print("=== Demo Complete ===")
        print("\nTo start the web interface, run:")
        print("  python main.py")
        print("\nOr for the API server:")
        print("  python -m uvicorn src.api:app --reload")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("\nPlease check your configuration and try again.")

if __name__ == "__main__":
    main()


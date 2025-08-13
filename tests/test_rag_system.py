#!/usr/bin/env python3
"""
Test suite for the RAG system.
"""
import sys
import os
import unittest
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import config
from src.llm_manager import llm_manager
from src.vector_store import vector_store
from src.rag_pipeline import rag_pipeline
from src.document_processor import document_processor

class TestRAGSystem(unittest.TestCase):
    """Test cases for the RAG system."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test documents
        self.test_documents = [
            {
                "id": "test_doc_1",
                "text": "Artificial Intelligence is a field of computer science that focuses on creating intelligent machines.",
                "metadata": {"source": "test", "category": "AI"}
            },
            {
                "id": "test_doc_2", 
                "text": "Machine Learning is a subset of AI that enables computers to learn without being explicitly programmed.",
                "metadata": {"source": "test", "category": "ML"}
            }
        ]
    
    def test_config_loading(self):
        """Test configuration loading."""
        self.assertIsNotNone(config)
        self.assertIsInstance(config.CHUNK_SIZE, int)
        self.assertIsInstance(config.CHUNK_OVERLAP, int)
        self.assertIsInstance(config.TOP_K_RETRIEVAL, int)
    
    def test_document_processor(self):
        """Test document processing capabilities."""
        # Test text processing
        doc = document_processor.process_text("Test document content")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["text"], "Test document content")
        self.assertEqual(doc["file_type"], "text")
    
    def test_vector_store_operations(self):
        """Test vector store operations."""
        # Test adding documents
        success = vector_store.add_documents(self.test_documents)
        self.assertTrue(success)
        
        # Test search functionality
        results = vector_store.search("artificial intelligence")
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Test collection info
        info = vector_store.get_collection_info()
        self.assertIsInstance(info, dict)
        self.assertIn("document_count", info)
    
    def test_rag_pipeline(self):
        """Test RAG pipeline functionality."""
        # Test document processing
        success = rag_pipeline.process_documents(self.test_documents)
        self.assertTrue(success)
        
        # Test response generation
        response = rag_pipeline.generate_response("What is AI?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Test pipeline info
        info = rag_pipeline.get_pipeline_info()
        self.assertIsInstance(info, dict)
        self.assertIn("chunk_size", info)
    
    def test_llm_manager(self):
        """Test LLM manager functionality."""
        # Test available providers
        providers = llm_manager.get_available_providers()
        self.assertIsInstance(providers, list)
        
        # Test provider info
        if providers:
            info = llm_manager.get_provider_info(providers[0])
            self.assertIsInstance(info, dict)
            self.assertIn("provider", info)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create test document
        test_doc = {
            "id": "integration_test",
            "text": "This is a test document for integration testing of the RAG system.",
            "metadata": {"test": True}
        }
        
        # Process through RAG pipeline
        success = rag_pipeline.process_documents([test_doc])
        self.assertTrue(success)
        
        # Generate response
        response = rag_pipeline.generate_response("What is this document about?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Verify response contains relevant information
        self.assertIn("test document", response.lower())

def run_tests():
    """Run all tests."""
    print("🧪 Running RAG System Tests...\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRAGSystem)
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"  - Tests run: {result.testsRun}")
    print(f"  - Failures: {len(result.failures)}")
    print(f"  - Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)


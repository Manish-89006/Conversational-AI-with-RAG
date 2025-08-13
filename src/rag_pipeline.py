"""
RAG pipeline for document processing and response generation.
"""
import logging
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .llm_manager import llm_manager
from .vector_store import vector_store
from .config import config

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline orchestrator."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Process and store documents in the vector database."""
        try:
            processed_chunks = []
            
            for doc in documents:
                # Split document into chunks
                chunks = self.text_splitter.split_text(doc["text"])
                
                # Create chunk documents
                for i, chunk in enumerate(chunks):
                    chunk_doc = {
                        "id": f"{doc['id']}_chunk_{i}",
                        "text": chunk,
                        "metadata": {
                            **doc.get("metadata", {}),
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "source_id": doc["id"]
                        }
                    }
                    processed_chunks.append(chunk_doc)
            
            # Store in vector database
            success = vector_store.add_documents(processed_chunks)
            
            if success:
                logger.info(f"Processed {len(documents)} documents into {len(processed_chunks)} chunks")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return False
    
    def generate_response(self, query: str, context: List[Dict[str, Any]] = None, 
                         provider: str = None) -> str:
        """Generate response using RAG approach."""
        try:
            # Retrieve relevant documents if context not provided
            if context is None:
                context = vector_store.search(query)
            
            if not context:
                logger.warning("No relevant context found for query")
                return "I don't have enough information to answer that question. Please try rephrasing or ask about a different topic."
            
            # Prepare system message with context
            context_text = "\n\n".join([f"Context {i+1}: {doc['text']}" 
                                       for i, doc in enumerate(context)])
            
            system_message = f"""You are a helpful AI assistant. Use the following context to answer the user's question. 
            If the context doesn't contain enough information to answer the question, say so.
            
            Context:
            {context_text}
            
            Answer the user's question based on the context provided."""
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
            
            # Generate response
            response = llm_manager.generate_response(messages, provider=provider)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def chat(self, messages: List[Dict[str, str]], provider: str = None) -> str:
        """Handle chat conversation with RAG enhancement."""
        try:
            if not messages:
                return "No messages provided"
            
            # Get the last user message
            last_message = messages[-1]
            if last_message["role"] != "user":
                return "Last message must be from user"
            
            query = last_message["content"]
            
            # Generate RAG-enhanced response
            response = self.generate_response(query, provider=provider)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Error in chat: {str(e)}"
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the RAG pipeline."""
        return {
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k_retrieval": config.TOP_K_RETRIEVAL,
            "embedding_model": config.EMBEDDING_MODEL,
            "vector_store_info": vector_store.get_collection_info(),
            "available_llms": llm_manager.get_available_providers(),
            "active_llm": llm_manager.active_provider
        }

# Global RAG pipeline instance
rag_pipeline = RAGPipeline()


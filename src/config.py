"""
Configuration management for the RAG system.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration management."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Hugging Face Configuration
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    HUGGINGFACE_MODEL: str = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    
    # Chroma Configuration
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "documents")
    
    # Application Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY and not cls.HUGGINGFACE_API_KEY:
            print("Warning: No API keys configured. Some features may not work.")
            return False
        return True
    
    @classmethod
    def get_available_models(cls) -> dict:
        """Get available LLM models."""
        models = {
            "openai": {
                "gpt-3.5-turbo": "GPT-3.5 Turbo (OpenAI)",
                "gpt-4": "GPT-4 (OpenAI)",
                "gpt-4-turbo": "GPT-4 Turbo (OpenAI)"
            },
            "huggingface": {
                "meta-llama/Llama-2-7b-chat-hf": "Llama 2 7B Chat (Hugging Face)",
                "microsoft/DialoGPT-medium": "DialoGPT Medium (Hugging Face)",
                "google/flan-t5-base": "Flan-T5 Base (Hugging Face)"
            }
        }
        return models

# Global configuration instance
config = Config()


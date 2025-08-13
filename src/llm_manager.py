"""
Modular LLM manager supporting multiple providers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import HumanMessage, SystemMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from .config import config

logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from messages."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass

class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation."""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        self.model_name = model_name or config.OPENAI_MODEL
        self.api_key = api_key or config.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI."""
        try:
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                else:
                    langchain_messages.append(HumanMessage(content=msg["content"]))
            
            response = self.llm.invoke(langchain_messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {e}")
            return f"Error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "openai",
            "model": self.model_name,
            "type": "proprietary",
            "capabilities": ["chat", "completion", "function_calling"]
        }

class HuggingFaceLLM(BaseLLM):
    """Hugging Face LLM implementation."""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        self.model_name = model_name or config.HUGGINGFACE_MODEL
        self.api_key = api_key or config.HUGGINGFACE_API_KEY
        
        try:
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Create LangChain wrapper
            self.llm = HuggingFacePipeline(pipeline=self.pipeline)
            
        except Exception as e:
            logger.error(f"Error initializing Hugging Face model: {e}")
            raise
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Hugging Face model."""
        try:
            # Format messages for the model
            prompt = self._format_messages(messages)
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating response with Hugging Face: {e}")
            return f"Error: {str(e)}"
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Hugging Face models."""
        formatted = ""
        for msg in messages:
            if msg["role"] == "system":
                formatted += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                formatted += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted += f"Assistant: {msg['content']}\n"
        
        formatted += "Assistant: "
        return formatted
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Hugging Face model information."""
        return {
            "provider": "huggingface",
            "model": self.model_name,
            "type": "open_source",
            "capabilities": ["text_generation", "chat"]
        }

class LLMManager:
    """Manager for multiple LLM providers."""
    
    def __init__(self):
        self.llms: Dict[str, BaseLLM] = {}
        self.active_provider = None
        self._initialize_llms()
    
    def _initialize_llms(self):
        """Initialize available LLM providers."""
        try:
            # Initialize OpenAI if API key is available
            if config.OPENAI_API_KEY:
                self.llms["openai"] = OpenAILLM()
                self.active_provider = "openai"
                logger.info("OpenAI LLM initialized successfully")
            
            # Initialize Hugging Face if API key is available
            if config.HUGGINGFACE_API_KEY:
                self.llms["huggingface"] = HuggingFaceLLM()
                if not self.active_provider:
                    self.active_provider = "huggingface"
                logger.info("Hugging Face LLM initialized successfully")
            
            if not self.llms:
                logger.warning("No LLM providers initialized")
                
        except Exception as e:
            logger.error(f"Error initializing LLMs: {e}")
    
    def get_llm(self, provider: str = None) -> Optional[BaseLLM]:
        """Get LLM instance for specified provider."""
        provider = provider or self.active_provider
        return self.llms.get(provider)
    
    def set_active_provider(self, provider: str) -> bool:
        """Set active LLM provider."""
        if provider in self.llms:
            self.active_provider = provider
            logger.info(f"Active provider set to: {provider}")
            return True
        return False
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        return list(self.llms.keys())
    
    def generate_response(self, messages: List[Dict[str, str]], provider: str = None, **kwargs) -> str:
        """Generate response using specified or active provider."""
        llm = self.get_llm(provider)
        if not llm:
            return "Error: No LLM provider available"
        
        return llm.generate(messages, **kwargs)
    
    def get_provider_info(self, provider: str = None) -> Dict[str, Any]:
        """Get information about specified or active provider."""
        llm = self.get_llm(provider)
        if not llm:
            return {"error": "Provider not available"}
        
        return llm.get_model_info()

# Global LLM manager instance
llm_manager = LLMManager()


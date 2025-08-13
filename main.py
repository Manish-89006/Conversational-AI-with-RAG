#!/usr/bin/env python3
"""
Main entry point for the Conversational AI with RAG system.
"""
import logging
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api import app
from src.config import config

def main():
    """Main function to run the RAG system."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate configuration
        if not config.validate():
            logger.warning("Configuration validation failed. Some features may not work.")
        
        logger.info("Starting Conversational AI with RAG system...")
        logger.info(f"Configuration: {config.HOST}:{config.PORT}")
        logger.info(f"Debug mode: {config.DEBUG}")
        
        # Import and run the FastAPI app
        import uvicorn
        uvicorn.run(
            "src.api:app",
            host=config.HOST,
            port=config.PORT,
            reload=config.DEBUG,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


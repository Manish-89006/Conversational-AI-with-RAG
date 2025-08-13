"""
Document processing utilities for various file formats.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import markdown
from .config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process various document formats for RAG system."""
    
    def __init__(self):
        self.supported_extensions = {
            '.txt': self._process_text,
            '.md': self._process_markdown,
            '.html': self._process_html,
            '.htm': self._process_html
        }
    
    def process_file(self, file_path: str, metadata: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Process a single file and return document data."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            # Get file extension
            extension = file_path.suffix.lower()
            
            if extension not in self.supported_extensions:
                logger.warning(f"Unsupported file extension: {extension}")
                return None
            
            # Process file based on extension
            processor = self.supported_extensions[extension]
            text_content = processor(file_path)
            
            if not text_content:
                logger.warning(f"No content extracted from file: {file_path}")
                return None
            
            # Prepare document data
            document = {
                "id": str(file_path.stem),
                "text": text_content,
                "metadata": metadata or {},
                "source": str(file_path),
                "file_type": extension,
                "file_size": file_path.stat().st_size
            }
            
            logger.info(f"Successfully processed file: {file_path}")
            return document
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None
    
    def process_directory(self, directory_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Process all supported files in a directory."""
        try:
            directory_path = Path(directory_path)
            
            if not directory_path.exists() or not directory_path.is_dir():
                logger.error(f"Directory not found: {directory_path}")
                return []
            
            documents = []
            
            # Process all files in directory
            for file_path in directory_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                    document = self.process_file(str(file_path), metadata)
                    if document:
                        documents.append(document)
            
            logger.info(f"Processed {len(documents)} documents from directory: {directory_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
            return []
    
    def process_url(self, url: str, metadata: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Process content from a URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            text_content = soup.get_text(separator='\n', strip=True)
            
            if not text_content:
                logger.warning(f"No text content extracted from URL: {url}")
                return None
            
            # Prepare document data
            document = {
                "id": f"url_{hash(url)}",
                "text": text_content,
                "metadata": metadata or {},
                "source": url,
                "file_type": "url",
                "title": soup.title.string if soup.title else "Unknown"
            }
            
            logger.info(f"Successfully processed URL: {url}")
            return document
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return None
    
    def process_text(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process raw text content."""
        try:
            document = {
                "id": f"text_{hash(text)}",
                "text": text,
                "metadata": metadata or {},
                "source": "raw_text",
                "file_type": "text",
                "text_length": len(text)
            }
            
            logger.info("Successfully processed raw text")
            return document
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {"error": str(e)}
    
    def _process_text(self, file_path: Path) -> str:
        """Process plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return ""
    
    def _process_markdown(self, file_path: Path) -> str:
        """Process markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert markdown to HTML then extract text
            html_content = markdown.markdown(md_content)
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
            
        except Exception as e:
            logger.error(f"Error processing markdown file {file_path}: {e}")
            return ""
    
    def _process_html(self, file_path: Path) -> str:
        """Process HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML and extract text
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
            
        except Exception as e:
            logger.error(f"Error processing HTML file {file_path}: {e}")
            return ""
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.supported_extensions.keys())
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if a file can be processed."""
        try:
            file_path = Path(file_path)
            return (file_path.exists() and 
                   file_path.is_file() and 
                   file_path.suffix.lower() in self.supported_extensions)
        except Exception:
            return False

# Global document processor instance
document_processor = DocumentProcessor()


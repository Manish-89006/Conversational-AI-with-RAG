"""
FastAPI backend for the RAG system.
"""
import logging
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
import uvicorn

from .rag_pipeline import rag_pipeline
from .document_processor import document_processor
from .llm_manager import llm_manager
from .vector_store import vector_store
from .config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Conversational AI with RAG",
    description="A powerful RAG system with modular LLM integration",
    version="1.0.0"
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    provider: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    provider: str
    context_used: bool

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int

class SystemInfo(BaseModel):
    rag_pipeline: Dict[str, Any]
    available_llms: List[str]
    active_llm: Optional[str]
    vector_store: Dict[str, Any]

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main web interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Conversational AI with RAG</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .chat-container { display: flex; flex-direction: column; height: 500px; }
            .chat-messages { flex: 1; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 5px; background: #fafafa; }
            .message { margin-bottom: 10px; padding: 10px; border-radius: 5px; }
            .user-message { background: #007bff; color: white; margin-left: 20%; }
            .assistant-message { background: #e9ecef; color: #333; margin-right: 20%; }
            .input-container { display: flex; gap: 10px; }
            .chat-input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            .send-btn { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .send-btn:hover { background: #0056b3; }
            .upload-section { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 5px; }
            .file-input { margin: 10px 0; }
            .upload-btn { padding: 10px 20px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .upload-btn:hover { background: #1e7e34; }
            .status { margin-top: 10px; padding: 10px; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Conversational AI with RAG</h1>
            
            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant-message">Hello! I'm your AI assistant with RAG capabilities. Ask me anything or upload documents to enhance my knowledge.</div>
                </div>
                
                <div class="input-container">
                    <input type="text" class="chat-input" id="chatInput" placeholder="Type your message here...">
                    <button class="send-btn" onclick="sendMessage()">Send</button>
                </div>
            </div>
            
            <div class="upload-section">
                <h3>Upload Documents</h3>
                <input type="file" class="file-input" id="fileInput" multiple accept=".txt,.md,.html,.htm">
                <button class="upload-btn" onclick="uploadFiles()">Upload</button>
                <div id="uploadStatus"></div>
            </div>
        </div>
        
        <script>
            async function sendMessage() {
                const input = document.getElementById('chatInput');
                const message = input.value.trim();
                if (!message) return;
                
                // Add user message to chat
                addMessage('user', message);
                input.value = '';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            messages: [{ role: 'user', content: message }]
                        })
                    });
                    
                    const data = await response.json();
                    addMessage('assistant', data.response);
                } catch (error) {
                    addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
                }
            }
            
            function addMessage(role, content) {
                const messagesDiv = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${role}-message`;
                messageDiv.textContent = content;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            async function uploadFiles() {
                const fileInput = document.getElementById('fileInput');
                const statusDiv = document.getElementById('uploadStatus');
                const files = fileInput.files;
                
                if (files.length === 0) {
                    statusDiv.innerHTML = '<div class="status error">Please select files to upload.</div>';
                    return;
                }
                
                const formData = new FormData();
                for (let file of files) {
                    formData.append('files', file);
                }
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    if (data.success) {
                        statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
                        fileInput.value = '';
                    } else {
                        statusDiv.innerHTML = `<div class="status error">${data.message}</div>`;
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">Upload failed. Please try again.</div>';
                }
            }
            
            // Enter key to send message
            document.getElementById('chatInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests with RAG enhancement."""
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Convert to list of dicts for RAG pipeline
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Generate RAG-enhanced response
        response = rag_pipeline.chat(messages, provider=request.provider)
        
        # Check if context was used
        context_used = len(request.messages) > 0
        
        return ChatResponse(
            response=response,
            provider=llm_manager.active_provider or "unknown",
            context_used=context_used
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        processed_documents = []
        
        for file in files:
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{file.filename}"
                with open(temp_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Process document
                document = document_processor.process_file(temp_path)
                if document:
                    processed_documents.append(document)
                
                # Clean up temp file
                os.remove(temp_path)
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                continue
        
        if not processed_documents:
            raise HTTPException(status_code=400, detail="No valid documents could be processed")
        
        # Add to RAG pipeline
        success = rag_pipeline.process_documents(processed_documents)
        
        if success:
            return DocumentUploadResponse(
                success=True,
                message=f"Successfully processed {len(processed_documents)} documents",
                documents_processed=len(processed_documents)
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to process documents")
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents():
    """Get information about processed documents."""
    try:
        info = vector_store.get_collection_info()
        return info
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get available LLM models."""
    try:
        return {
            "available_providers": llm_manager.get_available_providers(),
            "active_provider": llm_manager.active_provider,
            "provider_info": {
                provider: llm_manager.get_provider_info(provider)
                for provider in llm_manager.get_available_providers()
            }
        }
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/switch_provider/{provider}")
async def switch_provider(provider: str):
    """Switch to a different LLM provider."""
    try:
        success = llm_manager.set_active_provider(provider)
        if success:
            return {"message": f"Switched to {provider} provider", "active_provider": provider}
        else:
            raise HTTPException(status_code=400, detail=f"Provider {provider} not available")
    except Exception as e:
        logger.error(f"Error switching provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system_info", response_model=SystemInfo)
async def get_system_info():
    """Get comprehensive system information."""
    try:
        return SystemInfo(
            rag_pipeline=rag_pipeline.get_pipeline_info(),
            available_llms=llm_manager.get_available_providers(),
            active_llm=llm_manager.active_provider,
            vector_store=vector_store.get_collection_info()
        )
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Conversational AI with RAG"}

if __name__ == "__main__":
    # Validate configuration
    config.validate()
    
    # Run the application
    uvicorn.run(
        "src.api:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG
    )


# Conversational AI with RAG

A powerful Retrieval-Augmented Generation (RAG) system that seamlessly integrates open-source and proprietary LLMs with Chroma vector database to enhance conversational AI capabilities.

## Features

- **Modular LLM Integration**: Support for both open-source (Hugging Face) and proprietary (OpenAI) language models
- **Chroma Vector Database**: Efficient vector storage and similarity search
- **Automated Data Pipeline**: Automated processes for loading, splitting, embedding, and retrieving data
- **Flexible Deployment**: Modular architecture supporting various LLM and vector database combinations
- **Modern Web Interface**: Clean, responsive web UI for easy interaction
- **Context-Aware Responses**: Precise, contextually relevant answers based on retrieved information

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   FastAPI Backend│    │   RAG Pipeline  │
│                 │◄──►│                 │◄──►│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   LLM Manager   │    │  Chroma Vector  │
                       │                 │    │    Database     │
                       └─────────────────┘    └─────────────────┘
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the Application**:
   ```bash
   python main.py
   ```

4. **Access the Web Interface**:
   Open http://localhost:8000 in your browser

## Configuration

The system supports multiple LLM providers:

- **OpenAI**: GPT-3.5/4 models via API
- **Hugging Face**: Open-source models (local or hosted)
- **Custom Models**: Extensible framework for other providers

## Data Sources

Supported data formats:
- Text files (.txt, .md)
- PDF documents
- Web pages
- Structured data (CSV, JSON)

## API Endpoints

- `POST /chat`: Send messages and get RAG-enhanced responses
- `POST /upload`: Upload documents for processing
- `GET /documents`: List processed documents
- `GET /models`: Available LLM models

## License

MIT License - see LICENSE file for details


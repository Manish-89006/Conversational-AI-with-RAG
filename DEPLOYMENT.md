# Deployment Guide

This guide covers deploying the Conversational AI with RAG system in various environments.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for version control)
- API keys for LLM providers (OpenAI, Hugging Face)

## Local Development Setup

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd conversational-ai-rag

# Run setup script
python setup.py

# Or manually install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp env_example.txt .env

# Edit .env with your API keys
nano .env
```

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `HUGGINGFACE_API_KEY`: Your Hugging Face API key (optional)
- `CHROMA_PERSIST_DIRECTORY`: Path for Chroma database
- `HOST` and `PORT`: Server configuration

### 3. Test the System

```bash
# Run tests
python tests/test_rag_system.py

# Run demo
python demo.py

# Start web interface
python main.py
```

## Production Deployment

### Docker Deployment

#### 1. Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p chroma_db logs uploads temp

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]
```

#### 2. Build and Run

```bash
# Build image
docker build -t rag-system .

# Run container
docker run -d \
    --name rag-system \
    -p 8000:8000 \
    -v $(pwd)/chroma_db:/app/chroma_db \
    -v $(pwd)/.env:/app/.env \
    rag-system
```

### Docker Compose

#### 1. Create docker-compose.yml

```yaml
version: '3.8'

services:
  rag-system:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./.env:/app/.env
      - ./uploads:/app/uploads
    environment:
      - HOST=0.0.0.0
      - PORT=8000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - rag-system
    restart: unless-stopped
```

#### 2. Deploy with Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance

```bash
# Launch EC2 instance (Ubuntu 20.04 LTS)
# Install dependencies
sudo apt update
sudo apt install python3 python3-pip git nginx

# Clone repository
git clone <repository-url>
cd conversational-ai-rag

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup systemd service
sudo nano /etc/systemd/system/rag-system.service
```

Systemd service file:
```ini
[Unit]
Description=RAG System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/conversational-ai-rag
Environment=PATH=/home/ubuntu/conversational-ai-rag/venv/bin
ExecStart=/home/ubuntu/conversational-ai-rag/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

#### 2. Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Google Cloud Platform

#### 1. App Engine

```yaml
# app.yaml
runtime: python39
entrypoint: python main.py

env_variables:
  HOST: "0.0.0.0"
  PORT: "8080"

automatic_scaling:
  target_cpu_utilization: 0.6
  min_instances: 1
  max_instances: 10

resources:
  cpu: 1
  memory_gb: 2
  disk_size_gb: 10
```

#### 2. Deploy to App Engine

```bash
# Deploy
gcloud app deploy

# View logs
gcloud app logs tail -s default
```

## Monitoring and Logging

### 1. Application Logging

```python
# Configure logging in your application
import logging
from logging.handlers import RotatingFileHandler

# Setup file logging
handler = RotatingFileHandler('logs/rag-system.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### 2. Health Monitoring

```bash
# Health check endpoint
curl http://localhost:8000/health

# System info endpoint
curl http://localhost:8000/system_info
```

### 3. Performance Monitoring

- Monitor API response times
- Track vector database performance
- Monitor LLM API usage and costs
- Set up alerts for system failures

## Security Considerations

### 1. API Key Management

- Use environment variables for sensitive data
- Never commit API keys to version control
- Use secret management services in production
- Rotate API keys regularly

### 2. Network Security

- Use HTTPS in production
- Implement rate limiting
- Set up firewall rules
- Use VPN for internal deployments

### 3. Data Privacy

- Encrypt data at rest
- Implement access controls
- Audit data access logs
- Comply with data protection regulations

## Scaling Considerations

### 1. Horizontal Scaling

- Use load balancers for multiple instances
- Implement session management
- Use shared storage for vector database
- Consider microservices architecture

### 2. Performance Optimization

- Cache frequently accessed data
- Optimize vector search algorithms
- Use CDN for static assets
- Implement database connection pooling

### 3. Cost Optimization

- Monitor API usage costs
- Use appropriate instance sizes
- Implement auto-scaling policies
- Consider reserved instances for predictable workloads

## Troubleshooting

### Common Issues

1. **API Key Errors**: Check environment variables and API key validity
2. **Memory Issues**: Monitor memory usage and adjust chunk sizes
3. **Performance Issues**: Check vector database performance and LLM response times
4. **Network Issues**: Verify firewall rules and network connectivity

### Debug Commands

```bash
# Check system status
curl http://localhost:8000/health

# View application logs
tail -f logs/rag-system.log

# Check system resources
htop
df -h
free -h

# Test vector database
python -c "from src.vector_store import vector_store; print(vector_store.get_collection_info())"
```

## Support and Maintenance

- Regular dependency updates
- Security patches
- Performance monitoring
- Backup and recovery procedures
- Documentation updates

For additional support, refer to the README.md file or create an issue in the repository.


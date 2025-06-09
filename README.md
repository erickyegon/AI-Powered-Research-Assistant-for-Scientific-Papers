# 🧠 AI-Powered Research Assistant for Scientific Papers

A production-grade AI assistant that reads, analyzes, and answers questions about scientific papers using advanced LangGraph workflows, RAG (Retrieval-Augmented Generation), and the Euri LLM API.

## ✨ Features

### 🔬 Advanced Research Capabilities
- **Intelligent Document Processing**: Support for PDF, TXT, DOCX, and Markdown files
- **Multi-Modal RAG Pipeline**: Sophisticated retrieval-augmented generation with LangGraph workflows
- **Query Enhancement**: Automatic query expansion and optimization for better results
- **Citation Analysis**: Comprehensive citation extraction and relationship mapping
- **Comparative Analysis**: Side-by-side comparison of research findings and methodologies

### 🏗️ Production-Grade Architecture
- **LangGraph Workflows**: Multi-step, stateful AI workflows with error handling and retry logic
- **LangServe Integration**: RESTful API endpoints for direct workflow access
- **Euri LLM Integration**: High-performance language model API with custom client
- **Vector Database**: Pinecone integration for semantic search and similarity matching
- **PostgreSQL Database**: Robust data persistence with SQLAlchemy ORM
- **Redis Caching**: High-performance caching and session management

### 🎨 Professional User Interface
- **Modern Streamlit Frontend**: Responsive, professional-grade UI with real-time updates
- **Interactive Query Interface**: Advanced query options with type classification
- **Document Management**: Upload, process, and manage research documents
- **Session Management**: Persistent query history and user sessions
- **Real-time Processing**: Live progress tracking and status updates

### 🔒 Enterprise Security & Monitoring
- **JWT Authentication**: Secure token-based authentication system
- **Role-Based Access Control**: Granular permissions and user management
- **Rate Limiting**: Configurable request throttling and abuse prevention
- **Comprehensive Logging**: Structured logging with Prometheus metrics
- **Health Monitoring**: Detailed health checks and system diagnostics

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- Redis 6+
- Pinecone account (for vector storage)
- Euri API key

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

**Required Environment Variables:**
- `EURI_API_KEY`: Your Euri LLM API key (get from https://euron.one)
- `SECRET_KEY`: Application secret key for JWT tokens
- `DATABASE_URL`: PostgreSQL connection string

**Optional Environment Variables:**
- `PINECONE_API_KEY`: For vector database (can use local storage for development)
- `PINECONE_ENVIRONMENT`: Pinecone environment name
- `REDIS_URL`: Redis connection string (defaults to localhost)

**Validate your configuration:**
```bash
cd backend
python validate_env.py
```

### 3. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start the backend server
# Option 1: Working Server (Recommended - bypasses dependency conflicts)
python working_server.py

# Option 2: Main Backend Server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Option 3: From backend directory
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Frontend Setup
```bash
cd frontend

# Install dependencies
pip install -r requirements.txt

# Start the Streamlit app
streamlit run app.py --server.port 8501
```

### 5. Docker Deployment (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📖 Usage Guide

### Document Upload and Processing
1. Navigate to the Streamlit interface at `http://localhost:8501`
2. Use the document upload section to add research papers
3. Supported formats: PDF, TXT, DOCX, Markdown
4. Documents are automatically processed and indexed

### Querying Research Papers
1. Enter your research question in the query interface
2. Select query type (General Q&A, Summarization, Comparison, etc.)
3. Configure options (citations, detailed response)
4. Submit and view results with sources and citations

### API Access
The backend provides RESTful APIs for programmatic access:

```python
import requests

# Query the RAG system
response = requests.post("http://localhost:8000/rag/query", json={
    "query": "What are the latest developments in machine learning?",
    "query_type": "general_qa",
    "include_citations": True
})

print(response.json())
```

## 🏗️ Architecture Overview

### Backend Components
```
backend/
├── main.py                 # FastAPI application with LangServe
├── config.py              # Configuration management
├── graphs/                # LangGraph workflows
│   ├── rag_workflow.py    # Main RAG pipeline
│   ├── summarization_graph.py
│   └── citation_graph.py
├── tools/                 # Processing tools
│   ├── embedding_tool.py  # Vector embeddings
│   ├── paper_loader.py    # Document processing
│   └── query_processor.py # Query enhancement
├── utils/                 # Utilities
│   ├── eur_client.py      # Euri LLM client
│   ├── db.py             # Database models
│   ├── auth.py           # Authentication
│   └── memory.py         # Redis management
└── chains/               # LangChain components
```

### Frontend Components
```
frontend/
├── app.py                # Main Streamlit application
├── components/           # UI components
│   ├── markdown_viewer.py
│   └── citation_display.py
└── assets/              # Static assets
    └── styles.css
```

## 🔧 Configuration

### Environment Variables
Key configuration options in `.env`:

```bash
# Euri LLM API
EURI_API_KEY=your-api-key
EURI_MODEL=gpt-4.1-nano

# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# Vector Store
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-environment

# Processing
CHUNK_SIZE=1000
TOP_K_RETRIEVAL=5
MAX_FILE_SIZE=52428800
```

### Advanced Configuration
- **RAG Parameters**: Adjust chunk size, overlap, and retrieval settings
- **LLM Settings**: Configure temperature, max tokens, and timeout
- **Security**: Set JWT secrets, rate limits, and CORS origins
- **Monitoring**: Enable metrics, logging levels, and health checks

## 🧪 Development

### Running Tests
```bash
# Backend tests
cd backend
pytest tests/ -v

# Frontend tests
cd frontend
streamlit run app.py --server.headless true
```

### Code Quality
```bash
# Format code
black backend/ frontend/
isort backend/ frontend/

# Type checking
mypy backend/

# Linting
flake8 backend/ frontend/
```

### Database Migrations
```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## 📊 Monitoring & Observability

### Health Checks
- **API Health**: `GET /health` - Basic health status
- **Detailed Health**: `GET /health/detailed` - Component-level diagnostics
- **Metrics**: `GET /metrics` - Prometheus metrics endpoint

### Logging
Structured logging with multiple levels:
- **INFO**: General application flow
- **WARNING**: Potential issues
- **ERROR**: Error conditions
- **DEBUG**: Detailed debugging information

### Metrics
Key metrics tracked:
- Request count and duration
- Document processing time
- Query response time
- Error rates
- System resource usage

## 🚢 Deployment

### Render Deployment
The application is configured for easy deployment on Render:

1. Connect your GitHub repository to Render
2. Configure environment variables in Render dashboard
3. Deploy using the provided `render.yaml` configuration

### Docker Production
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with production settings
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes
Kubernetes manifests are available in the `k8s/` directory for container orchestration.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for all public methods
- Write tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain & LangGraph**: For the powerful AI workflow framework
- **Euri**: For the high-performance LLM API
- **Streamlit**: For the excellent web app framework
- **FastAPI**: For the modern, fast web framework
- **Pinecone**: For vector database capabilities

## 📞 Support

- **Documentation**: [Wiki](https://github.com/your-username/ai-research-assistant/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-username/ai-research-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/ai-research-assistant/discussions)
- **Email**: support@your-domain.com

---

**Built with ❤️ for the research community**

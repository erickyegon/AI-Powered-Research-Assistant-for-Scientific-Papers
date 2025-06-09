"""
Fresh AI Research Assistant API - Production Ready
A clean, working implementation with proper dependency handling and fallbacks.
"""

import os
import sys
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    ENV_LOADED = True
except ImportError:
    ENV_LOADED = False
    print("Warning: python-dotenv not installed")

# Core dependencies
try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    print("Error: FastAPI not installed. Run: pip install fastapi uvicorn")
    exit(1)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("Warning: httpx not installed. Run: pip install httpx")

# Try to import our custom modules with better error handling
try:
    import sys
    import os
    backend_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, backend_path)

    from tools.query_processor import QueryProcessor
    QUERY_PROCESSOR_AVAILABLE = True
    print("✅ Query processor available")
except ImportError as e:
    QUERY_PROCESSOR_AVAILABLE = False
    print(f"Warning: Query processor not available - {e}")

try:
    from utils.eur_client import create_euri_llm
    EURI_CLIENT_AVAILABLE = True
    print("✅ Euri client available")
except ImportError as e:
    EURI_CLIENT_AVAILABLE = False
    print(f"Warning: Euri client not available - {e}")

# Try to import LangGraph workflows
try:
    from graphs.rag_workflow import create_rag_workflow
    from graphs.research_workflow import create_research_workflow
    LANGGRAPH_WORKFLOWS_AVAILABLE = True
    print("✅ LangGraph workflows available")
except ImportError as e:
    LANGGRAPH_WORKFLOWS_AVAILABLE = False
    print(f"Warning: LangGraph workflows not available - {e}")

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pydantic models
class QueryRequest(BaseModel):
    """Research query request model."""
    query: str = Field(..., description="Research question", min_length=1, max_length=2000)
    max_results: int = Field(5, description="Maximum results", ge=1, le=20)
    include_enhancement: bool = Field(True, description="Whether to enhance the query")
    query_type: str = Field("research", description="Type of query")

class QueryResponse(BaseModel):
    """Research query response model."""
    original_query: str = Field(..., description="Original query")
    enhanced_query: Optional[str] = Field(None, description="Enhanced query")
    response: str = Field(..., description="AI response")
    key_terms: List[str] = Field(default_factory=list, description="Extracted key terms")
    intent: Dict[str, Any] = Field(default_factory=dict, description="Query intent classification")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    processing_time: float = Field(..., description="Processing time in seconds")

class HealthStatus(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall status")
    timestamp: str = Field(..., description="Check timestamp")
    services: Dict[str, bool] = Field(..., description="Service availability")
    configuration: Dict[str, Any] = Field(..., description="Configuration details")

class DocumentUpload(BaseModel):
    """Document upload response model."""
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Upload status")
    size: int = Field(..., description="File size in bytes")
    message: str = Field(..., description="Status message")

# Create FastAPI application
app = FastAPI(
    title="AI Research Assistant API",
    description="Professional API for scientific paper analysis and research queries",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global application state
app_state = {
    "llm": None,
    "query_processor": None,
    "rag_workflow": None,
    "research_workflow": None,
    "startup_time": None,
    "request_count": 0
}

@app.on_event("startup")
async def startup_event():
    """Initialize application components."""
    logger.info("🚀 Starting AI Research Assistant...")

    # Configure LangSmith if available
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ai-research-assistant")
        logger.info("✅ LangSmith tracing configured")

    app_state["startup_time"] = datetime.now()

    # Initialize LLM if available
    if EURI_CLIENT_AVAILABLE:
        try:
            app_state["llm"] = create_euri_llm()
            logger.info("✅ Euri LLM client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Euri LLM: {e}")

    # Initialize query processor
    if QUERY_PROCESSOR_AVAILABLE:
        try:
            app_state["query_processor"] = QueryProcessor(app_state["llm"])
            logger.info("✅ Query processor initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize query processor: {e}")

    # Initialize LangGraph workflows
    if LANGGRAPH_WORKFLOWS_AVAILABLE:
        try:
            app_state["rag_workflow"] = create_rag_workflow()
            logger.info("✅ RAG workflow initialized")

            app_state["research_workflow"] = create_research_workflow()
            logger.info("✅ Research workflow initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LangGraph workflows: {e}")

    logger.info("🎉 Application startup complete!")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Research Assistant API",
        "version": "1.0.0",
        "status": "running",
        "uptime": str(datetime.now() - app_state["startup_time"]) if app_state["startup_time"] else "unknown",
        "requests_processed": app_state["request_count"],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "query": "/api/query",
            "upload": "/api/documents/upload",
            "config": "/api/config",
            "test": "/api/test"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Comprehensive health check."""
    euri_key = os.getenv("EURI_API_KEY")
    euri_configured = bool(euri_key and len(euri_key) > 10)
    
    services = {
        "api": True,
        "environment_loaded": ENV_LOADED,
        "httpx_available": HTTPX_AVAILABLE,
        "euri_client": EURI_CLIENT_AVAILABLE,
        "euri_configured": euri_configured,
        "query_processor": QUERY_PROCESSOR_AVAILABLE,
        "langgraph_workflows": LANGGRAPH_WORKFLOWS_AVAILABLE,
        "llm_initialized": app_state["llm"] is not None,
        "processor_initialized": app_state["query_processor"] is not None,
        "rag_workflow_initialized": app_state["rag_workflow"] is not None,
        "research_workflow_initialized": app_state["research_workflow"] is not None
    }
    
    configuration = {
        "euri_model": os.getenv("EURI_MODEL", "gpt-4.1-nano"),
        "euri_base_url": os.getenv("EURI_BASE_URL", "https://api.euron.one/api/v1/euri/alpha/chat/completions"),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "cors_origins": cors_origins,
        "uptime": str(datetime.now() - app_state["startup_time"]) if app_state["startup_time"] else "unknown"
    }
    
    # Determine overall status
    critical_services = ["euri_configured", "httpx_available"]
    overall_status = "healthy" if all(services[s] for s in critical_services) else "degraded"
    
    return HealthStatus(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        services=services,
        configuration=configuration
    )

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a research query with enhancement and analysis."""
    import time
    start_time = time.time()
    app_state["request_count"] += 1
    
    logger.info(f"Processing query: {request.query[:100]}...")
    
    # Validate Euri API configuration
    if not os.getenv("EURI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Euri API key not configured. Please set EURI_API_KEY environment variable."
        )
    
    try:
        # Initialize response data
        enhanced_query = request.query
        key_terms = []
        intent = {}
        
        # Try to use RAG workflow first
        if app_state["rag_workflow"]:
            try:
                logger.info("Using LangGraph RAG workflow")
                rag_result = await app_state["rag_workflow"].run(
                    query=request.query,
                    documents=None  # In production, would retrieve from vector store
                )

                response_text = rag_result.get("response", "No response generated")
                enhanced_query = rag_result.get("enhanced_query", request.query)

                # Extract additional metadata from RAG workflow
                rag_metadata = rag_result.get("metadata", {})
                if rag_metadata:
                    key_terms = rag_metadata.get("key_terms", [])
                    intent = rag_metadata.get("intent", {})

            except Exception as e:
                logger.warning(f"RAG workflow error: {e}")
                # Fallback to direct API call
                response_text = await call_euri_api(enhanced_query)
        else:
            # Use query processor if available
            if app_state["query_processor"]:
                try:
                    if request.include_enhancement:
                        enhanced_query = await app_state["query_processor"].enhance_query(request.query)
                        logger.info(f"Query enhanced: {enhanced_query[:100]}...")

                    key_terms = await app_state["query_processor"].extract_key_terms(request.query)
                    intent = await app_state["query_processor"].classify_intent(request.query)

                except Exception as e:
                    logger.warning(f"Query processor error: {e}")

            # Generate response using Euri API
            response_text = await call_euri_api(enhanced_query)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            original_query=request.query,
            enhanced_query=enhanced_query if enhanced_query != request.query else None,
            response=response_text,
            key_terms=key_terms,
            intent=intent,
            sources=[],  # Would be populated by RAG system
            metadata={
                "query_type": request.query_type,
                "max_results": request.max_results,
                "model": os.getenv("EURI_MODEL", "gpt-4.1-nano"),
                "enhancement_used": enhanced_query != request.query,
                "processor_available": app_state["query_processor"] is not None,
                "timestamp": datetime.now().isoformat()
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing query: {e}")
        
        # Return error response
        return QueryResponse(
            original_query=request.query,
            enhanced_query=None,
            response=f"I apologize, but I encountered an error while processing your query: {str(e)}",
            key_terms=[],
            intent={},
            sources=[],
            metadata={
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            processing_time=processing_time
        )

async def call_euri_api(query: str) -> str:
    """Make a direct call to the Euri API."""
    if not HTTPX_AVAILABLE:
        raise HTTPException(status_code=500, detail="HTTP client not available. Install httpx: pip install httpx")
    
    api_key = os.getenv("EURI_API_KEY")
    base_url = os.getenv("EURI_BASE_URL", "https://api.euron.one/api/v1/euri/alpha/chat/completions")
    model = os.getenv("EURI_MODEL", "gpt-4.1-nano")
    temperature = float(os.getenv("EURI_TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("EURI_MAX_TOKENS", "2000"))
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI research assistant. Provide detailed, accurate, and well-structured responses to research questions. Include relevant context and explanations."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(base_url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling Euri API: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=502, detail=f"Euri API error: {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"Request error calling Euri API: {e}")
        raise HTTPException(status_code=503, detail="Failed to connect to Euri API")
    except Exception as e:
        logger.error(f"Unexpected error calling Euri API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/documents/upload", response_model=DocumentUpload)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    import uuid
    
    logger.info(f"Uploading document: {file.filename}")
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # In a full implementation, this would:
        # 1. Parse the document based on type (PDF, DOCX, TXT)
        # 2. Extract text and metadata
        # 3. Generate embeddings
        # 4. Store in vector database
        # 5. Index for search
        
        return DocumentUpload(
            document_id=document_id,
            filename=file.filename,
            status="uploaded",
            size=len(content),
            message=f"Document '{file.filename}' uploaded successfully. Full processing pipeline not implemented in this version."
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/config")
async def get_configuration():
    """Get current API configuration."""
    return {
        "api": {
            "name": "AI Research Assistant",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development")
        },
        "euri": {
            "model": os.getenv("EURI_MODEL", "gpt-4.1-nano"),
            "base_url": os.getenv("EURI_BASE_URL", "https://api.euron.one/api/v1/euri/alpha/chat/completions"),
            "temperature": float(os.getenv("EURI_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("EURI_MAX_TOKENS", "2000")),
            "configured": bool(os.getenv("EURI_API_KEY"))
        },
        "features": {
            "query_enhancement": QUERY_PROCESSOR_AVAILABLE,
            "document_upload": True,
            "health_monitoring": True,
            "error_handling": True
        },
        "dependencies": {
            "fastapi": True,
            "httpx": HTTPX_AVAILABLE,
            "dotenv": ENV_LOADED,
            "euri_client": EURI_CLIENT_AVAILABLE,
            "query_processor": QUERY_PROCESSOR_AVAILABLE
        }
    }

@app.post("/api/test")
async def test_api_connection():
    """Test the Euri API connection."""
    try:
        test_query = "Hello! This is a test message. Please respond with a brief confirmation."
        response = await call_euri_api(test_query)
        
        return {
            "status": "success",
            "message": "Euri API connection successful",
            "test_query": test_query,
            "response_preview": response[:150] + "..." if len(response) > 150 else response,
            "response_length": len(response),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Euri API connection failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with detailed error information."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "status_code": exc.status_code,
                "detail": exc.detail,
                "timestamp": datetime.now().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "detail": "An unexpected error occurred",
                "timestamp": datetime.now().isoformat()
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print("\n" + "="*60)
    print("🔬 AI Research Assistant API - Fresh Start")
    print("="*60)
    print(f"🚀 Server: http://{host}:{port}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print(f"❤️ Health: http://{host}:{port}/health")
    print(f"🧪 Test: http://{host}:{port}/api/test")
    print(f"⚙️ Config: http://{host}:{port}/api/config")
    print("="*60)
    print("🔑 Features:")
    print("  • Research query processing")
    print("  • Query enhancement and analysis")
    print("  • Document upload")
    print("  • Health monitoring")
    print("  • Comprehensive error handling")
    print("="*60 + "\n")
    
    uvicorn.run(
        "main_fresh:app",
        host=host,
        port=port,
        reload=debug,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )

"""
Professional LangServe Application for AI Research Assistant.
Serves LangChain applications using LangServe with LangSmith integration.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# FastAPI and LangServe imports
try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Error: FastAPI not available")

# LangServe imports
try:
    from langserve import add_routes
    from langserve.pydantic_v1 import BaseModel as LangServeBaseModel
    LANGSERVE_AVAILABLE = True
except ImportError:
    LANGSERVE_AVAILABLE = False
    print("Warning: LangServe not available. Install with: pip install langserve")

# LangChain imports
try:
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available")

# LangSmith imports
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True

    # Configure LangSmith environment variables if API key is available
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ai-research-assistant")
        print("✅ LangSmith configured via environment variables")
    else:
        print("⚠️ LangSmith API key not found")

except ImportError:
    LANGSMITH_AVAILABLE = False
    print("Warning: LangSmith not available. Install with: pip install langsmith")

# Local imports
try:
    from .utils.eur_client import create_euri_llm
    from .graphs.rag_workflow import create_rag_workflow
    from .graphs.research_workflow import create_research_workflow
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    print("Warning: Local components not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for LangServe
class ResearchQuery(LangServeBaseModel if LANGSERVE_AVAILABLE else BaseModel):
    """Research query input model for LangServe."""
    query: str = Field(..., description="Research question or query")
    documents: Optional[List[Dict[str, Any]]] = Field(None, description="Optional documents")
    max_results: int = Field(5, description="Maximum number of results")
    include_citations: bool = Field(True, description="Whether to include citations")

class ResearchResponse(LangServeBaseModel if LANGSERVE_AVAILABLE else BaseModel):
    """Research response output model for LangServe."""
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    processing_time: float = Field(..., description="Processing time in seconds")

class SummarizationRequest(LangServeBaseModel if LANGSERVE_AVAILABLE else BaseModel):
    """Document summarization request."""
    text: str = Field(..., description="Text to summarize")
    summary_type: str = Field("detailed", description="Type of summary")
    max_length: int = Field(500, description="Maximum summary length")

# Global instances
app_state = {
    "llm": None,
    "rag_workflow": None,
    "research_workflow": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🚀 Starting LangServe AI Research Assistant...")
    
    # Initialize components
    try:
        if COMPONENTS_AVAILABLE:
            app_state["llm"] = create_euri_llm()
            app_state["rag_workflow"] = create_rag_workflow()
            app_state["research_workflow"] = create_research_workflow()
            logger.info("✅ Components initialized successfully")
        else:
            logger.warning("⚠️ Components not available")
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
    
    yield
    
    logger.info("🛑 Shutting down LangServe AI Research Assistant...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="AI Research Assistant LangServe API",
    description="Professional API using LangServe for serving LangChain applications",
    version="1.0.0",
    lifespan=lifespan if FASTAPI_AVAILABLE else None
)

# Add CORS middleware
if FASTAPI_AVAILABLE:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "AI Research Assistant LangServe",
        "version": "1.0.0",
        "components": {
            "fastapi": FASTAPI_AVAILABLE,
            "langserve": LANGSERVE_AVAILABLE,
            "langchain": LANGCHAIN_AVAILABLE,
            "langsmith": LANGSMITH_AVAILABLE,
            "euri_configured": bool(os.getenv("EURI_API_KEY")),
            "langsmith_configured": bool(os.getenv("LANGSMITH_API_KEY")),
            "llm": app_state["llm"] is not None,
            "rag_workflow": app_state["rag_workflow"] is not None,
            "research_workflow": app_state["research_workflow"] is not None
        }
    }

# Research query processing
@traceable(name="process_research_query") if LANGSMITH_AVAILABLE else lambda f: f
async def process_research_query(request: ResearchQuery) -> ResearchResponse:
    """Process a research query using the RAG workflow."""
    import time
    start_time = time.time()
    
    try:
        # Convert documents if provided
        documents = []
        if request.documents:
            for doc_data in request.documents:
                doc = Document(
                    page_content=doc_data.get("content", ""),
                    metadata=doc_data.get("metadata", {})
                )
                documents.append(doc)
        
        # Run RAG workflow
        if app_state["rag_workflow"]:
            result = await app_state["rag_workflow"].run(
                query=request.query,
                documents=documents if documents else None
            )
        else:
            # Fallback response
            result = {
                "response": f"I received your query: '{request.query}'. However, the RAG workflow is not available.",
                "retrieved_docs": [],
                "metadata": {"workflow": "fallback"}
            }
        
        processing_time = time.time() - start_time
        
        # Format response
        sources = []
        if request.include_citations and result.get("retrieved_docs"):
            for i, doc in enumerate(result["retrieved_docs"], 1):
                sources.append({
                    "id": i,
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                    "relevance_score": doc.metadata.get("similarity_score", 0.0)
                })
        
        return ResearchResponse(
            answer=result.get("response", "No response generated"),
            sources=sources,
            metadata=result.get("metadata", {}),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing research query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# Document summarization
@traceable(name="summarize_document") if LANGSMITH_AVAILABLE else lambda f: f
async def summarize_document(request: SummarizationRequest) -> Dict[str, Any]:
    """Summarize a document using the LLM."""
    try:
        if not app_state["llm"]:
            raise HTTPException(status_code=503, detail="LLM not available")
        
        # Create summarization prompt based on type
        if request.summary_type == "brief":
            prompt_template = """Provide a brief summary of the following text in 2-3 sentences:

{text}

Brief Summary:"""
        elif request.summary_type == "bullet_points":
            prompt_template = """Summarize the following text as bullet points:

{text}

Key Points:
•"""
        else:  # detailed
            prompt_template = """Provide a detailed summary of the following text, highlighting key findings, methodology, and conclusions:

{text}

Detailed Summary:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | app_state["llm"] | StrOutputParser()
        
        # Truncate text if too long
        text = request.text[:5000] if len(request.text) > 5000 else request.text
        
        summary = await chain.ainvoke({"text": text})
        
        # Truncate summary if needed
        if len(summary) > request.max_length:
            summary = summary[:request.max_length] + "..."
        
        return {
            "summary": summary,
            "original_length": len(request.text),
            "summary_length": len(summary),
            "summary_type": request.summary_type
        }
        
    except Exception as e:
        logger.error(f"Error in document summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

# Setup LangServe routes
def setup_langserve_routes():
    """Setup LangServe routes for the application."""
    if not LANGSERVE_AVAILABLE:
        logger.warning("LangServe not available, skipping route setup")
        return
    
    try:
        # Research query chain
        research_chain = RunnableLambda(process_research_query)
        add_routes(
            app,
            research_chain,
            path="/research",
            input_type=ResearchQuery,
            output_type=ResearchResponse,
        )
        
        # Summarization chain
        summarization_chain = RunnableLambda(summarize_document)
        add_routes(
            app,
            summarization_chain,
            path="/summarize",
            input_type=SummarizationRequest,
        )
        
        # Simple LLM chain for general queries
        if app_state["llm"]:
            simple_prompt = ChatPromptTemplate.from_template(
                "You are a helpful AI research assistant. Answer the following question: {question}"
            )
            simple_chain = simple_prompt | app_state["llm"] | StrOutputParser()
            add_routes(
                app,
                simple_chain,
                path="/chat",
                input_type=str,
                output_type=str,
            )
        
        logger.info("✅ LangServe routes configured")
        
    except Exception as e:
        logger.error(f"Error setting up LangServe routes: {e}")

# Custom API endpoints (non-LangServe)
@app.post("/api/research", response_model=ResearchResponse)
async def api_research_query(request: ResearchQuery):
    """Custom API endpoint for research queries."""
    return await process_research_query(request)

@app.post("/api/summarize")
async def api_summarize(request: SummarizationRequest):
    """Custom API endpoint for document summarization."""
    return await summarize_document(request)

@app.get("/api/models")
async def list_available_models():
    """List available models and their capabilities."""
    return {
        "models": {
            "euri": {
                "name": os.getenv("EURI_MODEL", "gpt-4.1-nano"),
                "provider": "Euri",
                "capabilities": ["text_generation", "summarization", "question_answering"],
                "max_tokens": int(os.getenv("EURI_MAX_TOKENS", "2000")),
                "temperature": float(os.getenv("EURI_TEMPERATURE", "0.7"))
            }
        },
        "workflows": {
            "rag": {
                "name": "RAG Workflow",
                "description": "Retrieval-Augmented Generation for research queries",
                "available": app_state["rag_workflow"] is not None
            },
            "research": {
                "name": "Research Workflow", 
                "description": "Comprehensive research analysis workflow",
                "available": app_state["research_workflow"] is not None
            }
        },
        "langsmith": {
            "enabled": LANGSMITH_AVAILABLE and bool(os.getenv("LANGSMITH_API_KEY")),
            "project": os.getenv("LANGSMITH_PROJECT", "ai-research-assistant")
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "http_error"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"}
    )

# Setup LangServe routes on startup
@app.on_event("startup")
async def startup_event():
    """Setup LangServe routes after startup."""
    setup_langserve_routes()

if __name__ == "__main__":
    import uvicorn
    
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))  # Different port from main app
    
    logger.info(f"🚀 Starting AI Research Assistant LangServe on {host}:{port}")
    logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
    logger.info(f"🔗 LangServe Research: http://{host}:{port}/research/playground")
    logger.info(f"🔗 LangServe Chat: http://{host}:{port}/chat/playground")
    logger.info(f"🔗 LangServe Summarize: http://{host}:{port}/summarize/playground")
    
    uvicorn.run(
        "langserve_app:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )

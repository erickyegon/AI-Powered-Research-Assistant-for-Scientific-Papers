#!/usr/bin/env python3
"""
Production server optimized for Render deployment.
"""

import os
import sys
import logging
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Core imports
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    import httpx
except ImportError as e:
    print(f"❌ Missing core dependencies: {e}")
    sys.exit(1)

# Try to import Euri client
try:
    from utils.eur_client import create_euri_llm
    EURI_AVAILABLE = True
except ImportError:
    EURI_AVAILABLE = False

# Setup logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Research Assistant - Production",
    description="Production AI Research Assistant API",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None
)

# Add CORS
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
app_state = {
    "llm": None,
    "startup_complete": False
}

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    include_enhancement: bool = False
    max_results: int = 5

class QueryResponse(BaseModel):
    response: str
    enhanced_query: str = None
    metadata: dict = {}

@app.on_event("startup")
async def startup_event():
    """Initialize production components."""
    logger.info("🚀 Starting production AI Research Assistant...")
    
    # Configure LangSmith if available
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ai-research-assistant-production")
        logger.info("✅ LangSmith tracing configured")
    
    # Try to initialize LLM
    if EURI_AVAILABLE:
        try:
            app_state["llm"] = create_euri_llm()
            logger.info("✅ Euri LLM initialized")
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize Euri LLM: {e}")
    
    app_state["startup_complete"] = True
    logger.info("🎉 Production startup complete!")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Research Assistant - Production API",
        "status": "running",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Render."""
    return {
        "status": "healthy",
        "version": "production-1.0.0",
        "startup_complete": app_state["startup_complete"],
        "components": {
            "euri_available": EURI_AVAILABLE,
            "euri_configured": bool(os.getenv("EURI_API_KEY")),
            "langsmith_configured": bool(os.getenv("LANGSMITH_API_KEY")),
            "llm_initialized": app_state["llm"] is not None
        },
        "environment": os.getenv("ENVIRONMENT", "development")
    }

async def call_euri_api_direct(query: str) -> str:
    """Direct API call to Euri with production error handling."""
    api_key = os.getenv("EURI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Euri API key not configured")
    
    try:
        base_url = os.getenv("EURI_BASE_URL", "https://api.euron.one/api/v1/euri/alpha/chat/completions")
        model = os.getenv("EURI_MODEL", "gpt-4.1-nano")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": query}
            ],
            "temperature": float(os.getenv("EURI_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("EURI_MAX_TOKENS", "2000"))
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(base_url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
    except httpx.HTTPStatusError as e:
        logger.error(f"Euri API HTTP error: {e}")
        raise HTTPException(status_code=502, detail=f"Euri API error: {e.response.status_code}")
    except Exception as e:
        logger.error(f"Euri API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using direct API calls."""
    try:
        # Enhance query if requested
        if request.include_enhancement:
            enhanced_query = f"Please provide a comprehensive research analysis of: {request.query}"
        else:
            enhanced_query = request.query
        
        # Call Euri API directly
        response_text = await call_euri_api_direct(enhanced_query)
        
        return QueryResponse(
            response=response_text,
            enhanced_query=enhanced_query,
            metadata={
                "original_query": request.query,
                "processing_method": "direct_api",
                "model": os.getenv("EURI_MODEL", "gpt-4.1-nano"),
                "enhanced": request.include_enhancement,
                "environment": os.getenv("ENVIRONMENT", "development")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")

@app.get("/api/config")
async def get_config():
    """Get current configuration (limited in production)."""
    is_production = os.getenv("ENVIRONMENT") == "production"
    
    config = {
        "euri": {
            "api_key_configured": bool(os.getenv("EURI_API_KEY")),
            "model": os.getenv("EURI_MODEL", "gpt-4.1-nano"),
            "temperature": os.getenv("EURI_TEMPERATURE", "0.7"),
            "max_tokens": os.getenv("EURI_MAX_TOKENS", "2000")
        },
        "app": {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug": os.getenv("DEBUG", "false"),
            "version": "1.0.0"
        }
    }
    
    # Add sensitive info only in development
    if not is_production:
        config["euri"]["base_url"] = os.getenv("EURI_BASE_URL", "https://api.euron.one/api/v1/euri/alpha/chat/completions")
        config["langsmith"] = {
            "api_key_configured": bool(os.getenv("LANGSMITH_API_KEY")),
            "project": os.getenv("LANGSMITH_PROJECT", "ai-research-assistant"),
            "tracing_enabled": os.getenv("LANGCHAIN_TRACING_V2", "false")
        }
    
    return config

@app.post("/api/test")
async def test_euri_connection():
    """Test Euri API connection."""
    try:
        test_response = await call_euri_api_direct("Hello! Please respond with 'Connection successful' to confirm the API is working.")
        
        return {
            "status": "success",
            "message": "Euri API connection successful",
            "model": os.getenv("EURI_MODEL", "gpt-4.1-nano"),
            "environment": os.getenv("ENVIRONMENT", "development")
        }
    except HTTPException as e:
        return {
            "status": "error",
            "message": f"Euri API connection failed: {e.detail}",
            "model": os.getenv("EURI_MODEL", "gpt-4.1-nano")
        }

@app.post("/api/summarize")
async def summarize_text(text: str, summary_type: str = "brief"):
    """Summarize text using Euri API."""
    try:
        if summary_type == "brief":
            prompt = f"Provide a brief summary of the following text in 2-3 sentences:\n\n{text}"
        elif summary_type == "detailed":
            prompt = f"Provide a detailed summary of the following text:\n\n{text}"
        else:
            prompt = f"Summarize the following text:\n\n{text}"
        
        summary = await call_euri_api_direct(prompt)
        
        return {
            "summary": summary,
            "summary_type": summary_type,
            "original_length": len(text),
            "summary_length": len(summary)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail="Summarization failed")

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

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting Production AI Research Assistant on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

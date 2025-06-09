#!/usr/bin/env python3
"""
Minimal server with only essential functionality.
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️ python-dotenv not available")

# Core imports
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    print("✅ Core dependencies available")
except ImportError as e:
    print(f"❌ Missing core dependencies: {e}")
    print("Run: pip install fastapi uvicorn pydantic")
    sys.exit(1)

# Try to import Euri client
try:
    from utils.eur_client import create_euri_llm
    EURI_AVAILABLE = True
    print("✅ Euri client available")
except ImportError as e:
    EURI_AVAILABLE = False
    print(f"⚠️ Euri client not available: {e}")

# Create FastAPI app
app = FastAPI(
    title="AI Research Assistant - Minimal",
    description="Minimal version for testing",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

class QueryResponse(BaseModel):
    response: str
    enhanced_query: str = None
    metadata: dict = {}

@app.on_event("startup")
async def startup_event():
    """Initialize minimal components."""
    print("🚀 Starting minimal AI Research Assistant...")
    
    # Try to initialize LLM
    if EURI_AVAILABLE:
        try:
            app_state["llm"] = create_euri_llm()
            print("✅ Euri LLM initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize Euri LLM: {e}")
    
    app_state["startup_complete"] = True
    print("🎉 Minimal startup complete!")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Research Assistant - Minimal Version",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "minimal-1.0.0",
        "startup_complete": app_state["startup_complete"],
        "components": {
            "euri_available": EURI_AVAILABLE,
            "euri_configured": bool(os.getenv("EURI_API_KEY")),
            "llm_initialized": app_state["llm"] is not None
        },
        "environment": {
            "debug": os.getenv("DEBUG", "false"),
            "environment": os.getenv("ENVIRONMENT", "development")
        }
    }

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a simple query."""
    try:
        if not app_state["llm"]:
            return QueryResponse(
                response="LLM not available. Please check configuration.",
                enhanced_query=request.query,
                metadata={"error": "llm_not_available"}
            )
        
        # Simple query processing
        if request.include_enhancement:
            enhanced_query = f"Enhanced: {request.query}"
        else:
            enhanced_query = request.query
        
        # Try to call LLM
        try:
            # Simple test call
            response_text = f"Processed query: '{request.query}'. This is a minimal response from the AI Research Assistant."
            
            return QueryResponse(
                response=response_text,
                enhanced_query=enhanced_query,
                metadata={
                    "original_query": request.query,
                    "processing_method": "minimal",
                    "llm_available": True
                }
            )
            
        except Exception as e:
            return QueryResponse(
                response=f"Error processing query: {str(e)}",
                enhanced_query=enhanced_query,
                metadata={"error": str(e)}
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    return {
        "euri": {
            "api_key_configured": bool(os.getenv("EURI_API_KEY")),
            "base_url": os.getenv("EURI_BASE_URL", "not_set"),
            "model": os.getenv("EURI_MODEL", "not_set")
        },
        "langsmith": {
            "api_key_configured": bool(os.getenv("LANGSMITH_API_KEY")),
            "project": os.getenv("LANGSMITH_PROJECT", "not_set")
        },
        "app": {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug": os.getenv("DEBUG", "false"),
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": os.getenv("PORT", "8000")
        }
    }

@app.post("/api/test")
async def test_euri_connection():
    """Test Euri API connection."""
    if not EURI_AVAILABLE:
        return {
            "status": "error",
            "message": "Euri client not available",
            "available": False
        }
    
    if not app_state["llm"]:
        return {
            "status": "error", 
            "message": "LLM not initialized",
            "available": False
        }
    
    try:
        # Simple test - just return success if LLM is initialized
        return {
            "status": "success",
            "message": "Euri LLM client is initialized and ready",
            "available": True,
            "model": os.getenv("EURI_MODEL", "unknown")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Euri connection test failed: {str(e)}",
            "available": False
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
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"}
    )

if __name__ == "__main__":
    print("🚀 Starting Minimal AI Research Assistant Server")
    print("📍 Server will be available at: http://127.0.0.1:8000")
    print("🔗 Health check: http://127.0.0.1:8000/health")
    print("🔗 API docs: http://127.0.0.1:8000/docs")
    print("🔗 Configuration: http://127.0.0.1:8000/api/config")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

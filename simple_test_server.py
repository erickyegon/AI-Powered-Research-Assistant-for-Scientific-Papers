#!/usr/bin/env python3
"""
Simple test server to verify basic functionality.
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
    print("⚠️ python-dotenv not available, but continuing...")

# Test basic imports
try:
    from fastapi import FastAPI
    print("✅ FastAPI available")
except ImportError:
    print("❌ FastAPI not available - run: pip install fastapi")
    sys.exit(1)

try:
    import uvicorn
    print("✅ Uvicorn available")
except ImportError:
    print("❌ Uvicorn not available - run: pip install uvicorn")
    sys.exit(1)

# Check environment variables
euri_key = os.getenv("EURI_API_KEY")
if euri_key:
    print(f"✅ EURI_API_KEY configured (length: {len(euri_key)})")
else:
    print("⚠️ EURI_API_KEY not found in environment")

# Create a simple FastAPI app
app = FastAPI(title="Simple Test Server")

@app.get("/")
async def root():
    return {"message": "Simple test server is running!"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "message": "Simple test server health check",
        "euri_configured": bool(os.getenv("EURI_API_KEY"))
    }

@app.get("/test")
async def test():
    return {
        "message": "Test endpoint working",
        "environment": os.getenv("ENVIRONMENT", "not_set"),
        "debug": os.getenv("DEBUG", "not_set")
    }

if __name__ == "__main__":
    print("🚀 Starting simple test server...")
    print("📍 Server will be available at: http://127.0.0.1:8000")
    print("🔗 Health check: http://127.0.0.1:8000/health")
    print("🔗 Test endpoint: http://127.0.0.1:8000/test")
    print("📚 API docs: http://127.0.0.1:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

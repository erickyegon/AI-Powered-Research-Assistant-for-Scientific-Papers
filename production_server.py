#!/usr/bin/env python3
"""
Production server for Render deployment.
This file is in the root directory for easy Render deployment.
"""

import os
import sys
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Import and run the actual production server
if __name__ == "__main__":
    try:
        from backend.production_server import app
        import uvicorn
        
        # Get port from environment (Render sets this)
        port = int(os.getenv("PORT", "8000"))
        host = os.getenv("HOST", "0.0.0.0")
        
        print(f"🚀 Starting AI Research Assistant on {host}:{port}")
        print(f"📁 Backend path: {backend_path}")
        print(f"🌍 Environment: {os.getenv('ENVIRONMENT', 'development')}")
        
        # Run the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Falling back to working server...")
        
        # Fallback to working server
        try:
            from working_server import app
            import uvicorn
            
            port = int(os.getenv("PORT", "8000"))
            host = os.getenv("HOST", "0.0.0.0")
            
            print(f"🚀 Starting Working Server on {host}:{port}")
            
            uvicorn.run(
                app,
                host=host,
                port=port,
                reload=False,
                log_level="info"
            )
            
        except Exception as fallback_error:
            print(f"❌ Fallback failed: {fallback_error}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        sys.exit(1)

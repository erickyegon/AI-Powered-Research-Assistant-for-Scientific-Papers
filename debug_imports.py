#!/usr/bin/env python3
"""
Debug script to check what imports are failing.
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

print("🔍 Debugging Import Issues")
print("=" * 50)

# Test environment loading
print("\n📁 Environment Variables:")
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ python-dotenv: Available")
    
    euri_key = os.getenv("EURI_API_KEY")
    if euri_key:
        print(f"✅ EURI_API_KEY: Configured (length: {len(euri_key)})")
    else:
        print("❌ EURI_API_KEY: Not found")
        
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_key:
        print(f"✅ LANGSMITH_API_KEY: Configured (length: {len(langsmith_key)})")
    else:
        print("❌ LANGSMITH_API_KEY: Not found")
        
except ImportError as e:
    print(f"❌ python-dotenv: Not available - {e}")

# Test core dependencies
print("\n🔧 Core Dependencies:")
dependencies = [
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("pydantic", "Pydantic"),
    ("httpx", "HTTPX"),
]

for module, name in dependencies:
    try:
        __import__(module)
        print(f"✅ {name}: Available")
    except ImportError as e:
        print(f"❌ {name}: Not available - {e}")

# Test LangChain dependencies
print("\n🤖 LangChain Dependencies:")
langchain_deps = [
    ("langchain", "LangChain"),
    ("langchain_core", "LangChain Core"),
    ("langgraph", "LangGraph"),
    ("langserve", "LangServe"),
    ("langsmith", "LangSmith"),
]

for module, name in langchain_deps:
    try:
        __import__(module)
        print(f"✅ {name}: Available")
    except ImportError as e:
        print(f"❌ {name}: Not available - {e}")

# Test custom modules
print("\n🛠️ Custom Modules:")
custom_modules = [
    ("utils.eur_client", "Euri Client"),
    ("tools.query_processor", "Query Processor"),
    ("tools.embedding_tool", "Embedding Tool"),
    ("tools.summarization_tool", "Summarization Tool"),
    ("tools.citation_extractor", "Citation Extractor"),
]

for module, name in custom_modules:
    try:
        __import__(module)
        print(f"✅ {name}: Available")
    except ImportError as e:
        print(f"❌ {name}: Not available - {e}")

# Test workflow modules
print("\n🔄 Workflow Modules:")
workflow_modules = [
    ("graphs.rag_workflow", "RAG Workflow"),
    ("graphs.research_workflow", "Research Workflow"),
]

for module, name in workflow_modules:
    try:
        __import__(module)
        print(f"✅ {name}: Available")
    except ImportError as e:
        print(f"❌ {name}: Not available - {e}")

# Test main application import
print("\n🚀 Main Application:")
try:
    import main
    print("✅ Main application: Can be imported")
except ImportError as e:
    print(f"❌ Main application: Import failed - {e}")
except Exception as e:
    print(f"⚠️ Main application: Import succeeded but error during initialization - {e}")

print("\n" + "=" * 50)
print("🎯 Summary:")
print("If you see ❌ for core dependencies, install them with:")
print("   pip install fastapi uvicorn pydantic httpx python-dotenv")
print("\nIf you see ❌ for LangChain dependencies, install them with:")
print("   pip install langchain langchain-core langgraph langserve langsmith")
print("\nIf custom modules fail, check the file paths and syntax.")

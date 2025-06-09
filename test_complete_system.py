#!/usr/bin/env python3
"""
Comprehensive Test Script for AI Research Assistant
Tests all components including LangChain, LangServe, LangGraph, and LangSmith integration.
"""

import os
import sys
import asyncio
import logging
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_euri_client():
    """Test Euri LLM client."""
    print("\n🧪 Testing Euri Client...")
    
    try:
        from utils.eur_client import create_euri_llm
        
        llm = create_euri_llm()
        if llm:
            print("✅ Euri LLM client created successfully")
            
            # Test a simple query
            try:
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.output_parsers import StrOutputParser
                
                prompt = ChatPromptTemplate.from_template("Say hello in one sentence: {input}")
                chain = prompt | llm | StrOutputParser()
                
                result = await chain.ainvoke({"input": "test"})
                print(f"✅ Euri API test successful: {result[:50]}...")
                return True
            except Exception as e:
                print(f"❌ Euri API test failed: {e}")
                return False
        else:
            print("❌ Failed to create Euri LLM client")
            return False
            
    except Exception as e:
        print(f"❌ Euri client import failed: {e}")
        return False

async def test_query_processor():
    """Test query processor."""
    print("\n🧪 Testing Query Processor...")
    
    try:
        from tools.query_processor import QueryProcessor
        from utils.eur_client import create_euri_llm
        
        llm = create_euri_llm()
        processor = QueryProcessor(llm)
        
        test_query = "What is machine learning?"
        
        # Test query enhancement
        enhanced = await processor.enhance_query(test_query)
        print(f"✅ Query enhancement: {enhanced[:100]}...")
        
        # Test key term extraction
        key_terms = await processor.extract_key_terms(test_query)
        print(f"✅ Key terms extracted: {key_terms}")
        
        # Test intent classification
        intent = await processor.classify_intent(test_query)
        print(f"✅ Intent classified: {intent}")
        
        return True
        
    except Exception as e:
        print(f"❌ Query processor test failed: {e}")
        return False

async def test_summarization_tool():
    """Test summarization tool."""
    print("\n🧪 Testing Summarization Tool...")
    
    try:
        from tools.summarization_tool import SummarizationTool
        from utils.eur_client import create_euri_llm
        
        llm = create_euri_llm()
        summarizer = SummarizationTool(llm)
        
        test_text = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms 
        that can learn from and make predictions or decisions based on data. It involves 
        training models on large datasets to identify patterns and relationships. Common 
        applications include image recognition, natural language processing, and recommendation 
        systems. The field has grown rapidly with advances in computing power and data availability.
        """
        
        # Test different summary types
        for summary_type in ["brief", "detailed", "bullet_points"]:
            result = await summarizer.summarize(test_text, summary_type=summary_type)
            print(f"✅ {summary_type} summary: {result['summary'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Summarization tool test failed: {e}")
        return False

async def test_citation_extractor():
    """Test citation extractor."""
    print("\n🧪 Testing Citation Extractor...")
    
    try:
        from tools.citation_extractor import CitationExtractor
        from utils.eur_client import create_euri_llm
        
        llm = create_euri_llm()
        extractor = CitationExtractor(llm)
        
        test_text = """
        References:
        Smith, J. (2023). Machine Learning Fundamentals. Journal of AI Research, 15(3), 45-67.
        Johnson, A., & Brown, B. (2022). "Deep Learning Applications in Healthcare." 
        Proceedings of the International Conference on AI, 123-145.
        """
        
        result = await extractor.extract_citations(test_text, format_style="apa")
        print(f"✅ Citations extracted: {len(result['citations'])} citations found")
        
        if result['citations']:
            print(f"   First citation: {result['citations'][0][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Citation extractor test failed: {e}")
        return False

async def test_rag_workflow():
    """Test RAG workflow."""
    print("\n🧪 Testing RAG Workflow...")
    
    try:
        from graphs.rag_workflow import create_rag_workflow
        from langchain_core.documents import Document
        
        workflow = create_rag_workflow()
        
        # Create test documents
        test_docs = [
            Document(
                page_content="Machine learning is a powerful tool for data analysis and prediction.",
                metadata={"source": "test_doc_1"}
            ),
            Document(
                page_content="Neural networks are inspired by biological neural networks in the brain.",
                metadata={"source": "test_doc_2"}
            )
        ]
        
        # Test workflow
        result = await workflow.run(
            query="What is machine learning?",
            documents=test_docs
        )
        
        print(f"✅ RAG workflow completed")
        print(f"   Response: {result.get('response', 'No response')[:100]}...")
        print(f"   Enhanced query: {result.get('enhanced_query', 'None')}")
        print(f"   Retrieved docs: {len(result.get('retrieved_docs', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG workflow test failed: {e}")
        return False

async def test_research_workflow():
    """Test research workflow."""
    print("\n🧪 Testing Research Workflow...")
    
    try:
        from graphs.research_workflow import create_research_workflow
        from langchain_core.documents import Document
        
        workflow = create_research_workflow()
        
        # Create test documents
        test_docs = [
            Document(
                page_content="This study investigates machine learning applications in healthcare.",
                metadata={"source": "research_paper_1"}
            )
        ]
        
        # Test workflow
        result = await workflow.run(
            query="Analyze machine learning in healthcare",
            research_type="literature_review",
            documents=test_docs
        )
        
        print(f"✅ Research workflow completed")
        print(f"   Final report length: {len(result.get('final_report', ''))}")
        print(f"   Citations found: {len(result.get('citations', []))}")
        print(f"   Summary available: {bool(result.get('summary'))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Research workflow test failed: {e}")
        return False

async def test_health_checks():
    """Test health checks for all components."""
    print("\n🧪 Testing Health Checks...")
    
    components_to_test = [
        ("Query Processor", "tools.query_processor", "QueryProcessor"),
        ("Summarization Tool", "tools.summarization_tool", "SummarizationTool"),
        ("Citation Extractor", "tools.citation_extractor", "CitationExtractor"),
        ("RAG Workflow", "graphs.rag_workflow", "RAGWorkflow"),
        ("Research Workflow", "graphs.research_workflow", "ResearchWorkflow"),
    ]
    
    health_results = {}
    
    for name, module_path, class_name in components_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            component_class = getattr(module, class_name)
            
            if hasattr(component_class, 'health_check'):
                if name in ["Query Processor", "Summarization Tool", "Citation Extractor"]:
                    from utils.eur_client import create_euri_llm
                    llm = create_euri_llm()
                    component = component_class(llm)
                else:
                    component = component_class()
                
                health = component.health_check()
                health_results[name] = health
                status = health.get('status', 'unknown')
                print(f"✅ {name}: {status}")
            else:
                print(f"⚠️ {name}: No health check available")
                
        except Exception as e:
            print(f"❌ {name}: Health check failed - {e}")
            health_results[name] = {"status": "error", "error": str(e)}
    
    return health_results

def check_environment():
    """Check environment configuration."""
    print("\n🧪 Checking Environment Configuration...")
    
    required_vars = ["EURI_API_KEY"]
    optional_vars = ["LANGSMITH_API_KEY", "PINECONE_API_KEY", "DATABASE_URL"]
    
    print("Required Environment Variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: Configured (length: {len(value)})")
        else:
            print(f"❌ {var}: Not configured")
    
    print("\nOptional Environment Variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: Configured")
        else:
            print(f"⚠️ {var}: Not configured")

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n🧪 Checking Dependencies...")
    
    dependencies = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("langchain", "LangChain"),
        ("langgraph", "LangGraph"),
        ("langserve", "LangServe"),
        ("langsmith", "LangSmith"),
        ("streamlit", "Streamlit"),
        ("httpx", "HTTPX"),
        ("pydantic", "Pydantic"),
    ]
    
    for package, name in dependencies:
        try:
            __import__(package)
            print(f"✅ {name}: Installed")
        except ImportError:
            print(f"❌ {name}: Not installed")

async def main():
    """Run comprehensive system test."""
    print("🚀 AI Research Assistant - Comprehensive System Test")
    print("=" * 60)
    
    # Check environment and dependencies
    check_environment()
    check_dependencies()
    
    # Test individual components
    test_results = {}
    
    test_results["euri_client"] = await test_euri_client()
    test_results["query_processor"] = await test_query_processor()
    test_results["summarization_tool"] = await test_summarization_tool()
    test_results["citation_extractor"] = await test_citation_extractor()
    test_results["rag_workflow"] = await test_rag_workflow()
    test_results["research_workflow"] = await test_research_workflow()
    
    # Test health checks
    health_results = await test_health_checks()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    print(f"Component Tests: {passed}/{total} passed")
    
    for component, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {component}: {status}")
    
    print(f"\nHealth Checks: {len(health_results)} components checked")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready for use.")
        print("\n🚀 Next steps:")
        print("1. Start the main API: uvicorn backend.main:app --reload")
        print("2. Start LangServe API: uvicorn backend.langserve_app:app --port 8001 --reload")
        print("3. Start Streamlit UI: streamlit run frontend/app.py")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed and environment variables are configured.")

if __name__ == "__main__":
    asyncio.run(main())

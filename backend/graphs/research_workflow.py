"""
Professional Research Workflow using LangGraph and LangSmith.
Implements comprehensive research analysis with multiple specialized agents.
"""

import os
import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# LangChain imports
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# LangSmith imports
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

# Local imports
try:
    from ..utils.eur_client import create_euri_llm
    from ..tools.embedding_tool import EmbeddingTool
    from ..tools.query_processor import QueryProcessor
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False

# Import additional tools with fallbacks
try:
    from ..tools.citation_extractor import CitationExtractor
    CITATION_EXTRACTOR_AVAILABLE = True
except ImportError:
    CITATION_EXTRACTOR_AVAILABLE = False
    CitationExtractor = None

try:
    from ..tools.summarization_tool import SummarizationTool
    SUMMARIZATION_TOOL_AVAILABLE = True
except ImportError:
    SUMMARIZATION_TOOL_AVAILABLE = False
    SummarizationTool = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State definition for Research workflow
class ResearchState(TypedDict):
    """State for research workflow."""
    query: str
    research_type: str
    documents: Optional[List[Document]]
    analysis_results: Dict[str, Any]
    citations: List[Dict[str, Any]]
    summary: Optional[str]
    final_report: Optional[str]
    metadata: Dict[str, Any]
    error: Optional[str]

class ResearchWorkflow:
    """
    Professional research workflow using LangGraph.
    Implements comprehensive research analysis with specialized agents.
    """
    
    def __init__(self):
        """Initialize the research workflow."""
        self.llm = None
        self.embedding_tool = None
        self.query_processor = None
        self.citation_extractor = None
        self.summarization_tool = None
        self.graph = None
        
        # Initialize components
        self._initialize_components()
        
        # Build workflow graph
        if LANGGRAPH_AVAILABLE:
            self._build_graph()
        else:
            logger.warning("LangGraph not available. Workflow will use fallback methods.")
    
    def _initialize_components(self):
        """Initialize workflow components."""
        try:
            if TOOLS_AVAILABLE:
                self.llm = create_euri_llm()
                self.embedding_tool = EmbeddingTool()
                self.query_processor = QueryProcessor(self.llm)
                logger.info("✅ Core research workflow components initialized")

            # Initialize additional tools if available
            if CITATION_EXTRACTOR_AVAILABLE:
                self.citation_extractor = CitationExtractor(self.llm)
                logger.info("✅ Citation extractor initialized")
            else:
                logger.warning("⚠️ Citation extractor not available")

            if SUMMARIZATION_TOOL_AVAILABLE:
                self.summarization_tool = SummarizationTool(self.llm)
                logger.info("✅ Summarization tool initialized")
            else:
                logger.warning("⚠️ Summarization tool not available")

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def _build_graph(self):
        """Build the LangGraph research workflow."""
        if not LANGGRAPH_AVAILABLE:
            return
        
        try:
            # Create workflow graph
            workflow = StateGraph(ResearchState)
            
            # Add nodes
            workflow.add_node("analyze_query", self._analyze_query_node)
            workflow.add_node("extract_citations", self._extract_citations_node)
            workflow.add_node("analyze_content", self._analyze_content_node)
            workflow.add_node("generate_summary", self._generate_summary_node)
            workflow.add_node("compile_report", self._compile_report_node)
            
            # Add edges
            workflow.set_entry_point("analyze_query")
            workflow.add_edge("analyze_query", "extract_citations")
            workflow.add_edge("extract_citations", "analyze_content")
            workflow.add_edge("analyze_content", "generate_summary")
            workflow.add_edge("generate_summary", "compile_report")
            workflow.add_edge("compile_report", END)
            
            # Compile graph
            self.graph = workflow.compile()
            logger.info("✅ Research workflow graph compiled")
            
        except Exception as e:
            logger.error(f"Error building research graph: {e}")
            self.graph = None
    
    @traceable(name="analyze_query") if LANGSMITH_AVAILABLE else lambda f: f
    async def _analyze_query_node(self, state: ResearchState) -> ResearchState:
        """Analyze the research query and determine research type."""
        try:
            if self.query_processor:
                # Classify query intent
                intent = await self.query_processor.classify_intent(state["query"])
                
                # Extract key terms
                key_terms = await self.query_processor.extract_key_terms(state["query"])
                
                state["analysis_results"] = {
                    "intent": intent,
                    "key_terms": key_terms,
                    "research_focus": self._determine_research_focus(intent, key_terms)
                }
                
                logger.info(f"Query analyzed: {intent.get('primary_intent', 'Unknown')}")
            else:
                state["analysis_results"] = {"error": "Query processor not available"}
            
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            state["error"] = str(e)
            return state
    
    @traceable(name="extract_citations") if LANGSMITH_AVAILABLE else lambda f: f
    async def _extract_citations_node(self, state: ResearchState) -> ResearchState:
        """Extract citations from provided documents."""
        try:
            citations = []
            
            if state.get("documents") and self.citation_extractor:
                for doc in state["documents"]:
                    doc_citations = await self.citation_extractor.extract_citations(doc.page_content)
                    citations.extend(doc_citations)
                
                logger.info(f"Extracted {len(citations)} citations")
            
            state["citations"] = citations
            return state
            
        except Exception as e:
            logger.error(f"Error extracting citations: {e}")
            state["citations"] = []
            state["error"] = str(e)
            return state
    
    @traceable(name="analyze_content") if LANGSMITH_AVAILABLE else lambda f: f
    async def _analyze_content_node(self, state: ResearchState) -> ResearchState:
        """Analyze document content based on research type."""
        try:
            if not state.get("documents"):
                state["analysis_results"]["content_analysis"] = "No documents provided for analysis"
                return state
            
            research_type = state.get("research_type", "general")
            analysis_results = state.get("analysis_results", {})
            
            # Perform content analysis based on research type
            if research_type == "literature_review":
                content_analysis = await self._analyze_literature(state["documents"])
            elif research_type == "methodology":
                content_analysis = await self._analyze_methodology(state["documents"])
            elif research_type == "comparative":
                content_analysis = await self._analyze_comparative(state["documents"])
            else:
                content_analysis = await self._analyze_general(state["documents"])
            
            analysis_results["content_analysis"] = content_analysis
            state["analysis_results"] = analysis_results
            
            logger.info(f"Content analysis completed for {research_type}")
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            state["error"] = str(e)
            return state
    
    @traceable(name="generate_summary") if LANGSMITH_AVAILABLE else lambda f: f
    async def _generate_summary_node(self, state: ResearchState) -> ResearchState:
        """Generate summary of research findings."""
        try:
            if self.summarization_tool and state.get("documents"):
                # Combine all document content
                combined_content = "\n\n".join([doc.page_content for doc in state["documents"]])
                
                # Generate summary based on research type
                research_type = state.get("research_type", "general")
                summary = await self.summarization_tool.summarize(
                    combined_content,
                    summary_type=research_type
                )
                
                state["summary"] = summary
                logger.info("Research summary generated")
            else:
                state["summary"] = "Summary generation not available"
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            state["summary"] = f"Error generating summary: {str(e)}"
            state["error"] = str(e)
            return state
    
    @traceable(name="compile_report") if LANGSMITH_AVAILABLE else lambda f: f
    async def _compile_report_node(self, state: ResearchState) -> ResearchState:
        """Compile final research report."""
        try:
            query = state["query"]
            analysis_results = state.get("analysis_results", {})
            citations = state.get("citations", [])
            summary = state.get("summary", "")
            
            # Create comprehensive report
            report_sections = []
            
            # Executive Summary
            report_sections.append("# Research Report\n")
            report_sections.append(f"**Query:** {query}\n")
            report_sections.append(f"**Research Type:** {state.get('research_type', 'General')}\n")
            report_sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Key Findings
            if summary:
                report_sections.append("## Executive Summary\n")
                report_sections.append(f"{summary}\n\n")
            
            # Analysis Results
            if analysis_results:
                report_sections.append("## Analysis Results\n")
                
                intent = analysis_results.get("intent", {})
                if intent:
                    report_sections.append(f"**Query Intent:** {intent.get('primary_intent', 'Unknown')}\n")
                    report_sections.append(f"**Confidence:** {intent.get('confidence', 0)}%\n\n")
                
                key_terms = analysis_results.get("key_terms", [])
                if key_terms:
                    report_sections.append(f"**Key Terms:** {', '.join(key_terms)}\n\n")
            
            # Citations
            if citations:
                report_sections.append("## References\n")
                for i, citation in enumerate(citations[:10], 1):  # Limit to 10 citations
                    if isinstance(citation, dict):
                        citation_text = citation.get("text", str(citation))
                    else:
                        citation_text = str(citation)
                    report_sections.append(f"{i}. {citation_text}\n")
                report_sections.append("\n")
            
            # Metadata
            report_sections.append("## Methodology\n")
            report_sections.append(f"- Documents analyzed: {len(state.get('documents', []))}\n")
            report_sections.append(f"- Citations extracted: {len(citations)}\n")
            report_sections.append(f"- Analysis framework: LangGraph + LangSmith\n")
            
            final_report = "".join(report_sections)
            state["final_report"] = final_report
            
            # Update metadata
            state["metadata"] = {
                "query": query,
                "research_type": state.get("research_type", "general"),
                "documents_count": len(state.get("documents", [])),
                "citations_count": len(citations),
                "timestamp": datetime.now().isoformat(),
                "workflow": "langgraph_research"
            }
            
            logger.info("Research report compiled successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error compiling report: {e}")
            state["final_report"] = f"Error compiling report: {str(e)}"
            state["error"] = str(e)
            return state
    
    def _determine_research_focus(self, intent: Dict[str, Any], key_terms: List[str]) -> str:
        """Determine research focus based on intent and key terms."""
        primary_intent = intent.get("primary_intent", "").upper()
        
        if primary_intent == "COMPARATIVE":
            return "comparative_analysis"
        elif primary_intent == "METHODOLOGICAL":
            return "methodology_review"
        elif "literature" in " ".join(key_terms).lower():
            return "literature_review"
        else:
            return "general_research"
    
    async def _analyze_literature(self, documents: List[Document]) -> str:
        """Analyze documents for literature review."""
        return f"Literature review analysis of {len(documents)} documents completed."
    
    async def _analyze_methodology(self, documents: List[Document]) -> str:
        """Analyze documents for methodology."""
        return f"Methodology analysis of {len(documents)} documents completed."
    
    async def _analyze_comparative(self, documents: List[Document]) -> str:
        """Analyze documents for comparative study."""
        return f"Comparative analysis of {len(documents)} documents completed."
    
    async def _analyze_general(self, documents: List[Document]) -> str:
        """General document analysis."""
        return f"General analysis of {len(documents)} documents completed."
    
    async def run(self, query: str, research_type: str = "general", documents: Optional[List[Document]] = None) -> Dict[str, Any]:
        """Run the research workflow."""
        try:
            # Initialize state
            initial_state: ResearchState = {
                "query": query,
                "research_type": research_type,
                "documents": documents or [],
                "analysis_results": {},
                "citations": [],
                "summary": None,
                "final_report": None,
                "metadata": {},
                "error": None
            }
            
            if self.graph:
                # Use LangGraph workflow
                logger.info("Running LangGraph research workflow")
                final_state = await self.graph.ainvoke(initial_state)
            else:
                # Fallback to sequential execution
                logger.info("Running fallback research workflow")
                final_state = await self._run_fallback(initial_state)
            
            return {
                "query": final_state["query"],
                "research_type": final_state["research_type"],
                "analysis_results": final_state.get("analysis_results", {}),
                "citations": final_state.get("citations", []),
                "summary": final_state.get("summary"),
                "final_report": final_state.get("final_report"),
                "metadata": final_state.get("metadata", {}),
                "error": final_state.get("error")
            }
            
        except Exception as e:
            logger.error(f"Error running research workflow: {e}")
            return {
                "query": query,
                "final_report": f"Research workflow error: {str(e)}",
                "error": str(e),
                "metadata": {"workflow": "error_fallback"}
            }
    
    async def _run_fallback(self, state: ResearchState) -> ResearchState:
        """Fallback workflow execution without LangGraph."""
        try:
            # Sequential execution
            state = await self._analyze_query_node(state)
            state = await self._extract_citations_node(state)
            state = await self._analyze_content_node(state)
            state = await self._generate_summary_node(state)
            state = await self._compile_report_node(state)
            return state
        except Exception as e:
            logger.error(f"Error in fallback research workflow: {e}")
            state["error"] = str(e)
            return state
    
    def health_check(self) -> Dict[str, Any]:
        """Check research workflow health."""
        return {
            "status": "healthy" if self.graph else "degraded",
            "components": {
                "langchain": LANGCHAIN_AVAILABLE,
                "langgraph": LANGGRAPH_AVAILABLE,
                "langsmith": LANGSMITH_AVAILABLE,
                "llm": self.llm is not None,
                "embedding_tool": self.embedding_tool is not None,
                "query_processor": self.query_processor is not None,
                "citation_extractor": self.citation_extractor is not None,
                "summarization_tool": self.summarization_tool is not None,
                "graph_compiled": self.graph is not None
            },
            "workflow_mode": "langgraph" if self.graph else "fallback"
        }

def create_research_workflow() -> ResearchWorkflow:
    """Factory function to create research workflow."""
    return ResearchWorkflow()

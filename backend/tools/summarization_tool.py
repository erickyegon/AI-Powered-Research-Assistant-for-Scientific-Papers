"""
Professional Summarization Tool for AI Research Assistant.
Provides comprehensive document summarization with multiple strategies and LangSmith tracing.
"""

import os
import logging
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# LangChain imports
try:
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available")

# LangSmith imports
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    print("Warning: LangSmith not available")

# Try to import httpx for direct API calls
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("Warning: httpx not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationTool:
    """
    Professional summarization tool with multiple summarization strategies.
    Supports various summary types and integrates with LangSmith for tracing.
    """

    def __init__(self, llm=None):
        """Initialize the summarization tool."""
        self.llm = llm
        self.max_chunk_size = int(os.getenv("SUMMARIZATION_MAX_CHUNK_SIZE", "4000"))
        self.overlap_size = int(os.getenv("SUMMARIZATION_OVERLAP_SIZE", "200"))

        # Summary type configurations
        self.summary_configs = {
            "brief": {
                "max_length": 200,
                "style": "concise",
                "focus": "key_points"
            },
            "detailed": {
                "max_length": 800,
                "style": "comprehensive",
                "focus": "methodology_and_findings"
            },
            "bullet_points": {
                "max_length": 400,
                "style": "structured",
                "focus": "key_insights"
            },
            "executive": {
                "max_length": 300,
                "style": "business",
                "focus": "implications_and_conclusions"
            },
            "technical": {
                "max_length": 600,
                "style": "academic",
                "focus": "methodology_and_results"
            }
        }

        logger.info("✅ Summarization tool initialized")

    @traceable(name="summarize_document") if LANGSMITH_AVAILABLE else lambda f: f
    async def summarize(
        self,
        content: Union[str, Document, List[Document]],
        summary_type: str = "detailed",
        custom_prompt: Optional[str] = None,
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Summarize content using the specified strategy.

        Args:
            content: Text content, Document, or list of Documents to summarize
            summary_type: Type of summary (brief, detailed, bullet_points, executive, technical)
            custom_prompt: Optional custom prompt template
            max_length: Optional custom maximum length

        Returns:
            Dictionary containing summary and metadata
        """
        try:
            # Prepare content
            text_content = self._prepare_content(content)

            # Get summary configuration
            config = self.summary_configs.get(summary_type, self.summary_configs["detailed"])
            if max_length:
                config["max_length"] = max_length

            # Check if content needs chunking
            if len(text_content) > self.max_chunk_size:
                logger.info(f"Content too long ({len(text_content)} chars), using chunked summarization")
                summary_result = await self._chunked_summarization(text_content, summary_type, config, custom_prompt)
            else:
                logger.info(f"Using direct summarization for {len(text_content)} chars")
                summary_result = await self._direct_summarization(text_content, summary_type, config, custom_prompt)

            # Add metadata
            summary_result.update({
                "summary_type": summary_type,
                "original_length": len(text_content),
                "compression_ratio": len(text_content) / len(summary_result["summary"]) if summary_result["summary"] else 0,
                "timestamp": datetime.now().isoformat(),
                "tool": "summarization_tool"
            })

            return summary_result

        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "error": str(e),
                "summary_type": summary_type,
                "timestamp": datetime.now().isoformat()
            }

    def _prepare_content(self, content: Union[str, Document, List[Document]]) -> str:
        """Prepare content for summarization."""
        if isinstance(content, str):
            return content
        elif isinstance(content, Document):
            return content.page_content
        elif isinstance(content, list):
            # Combine multiple documents
            combined_content = []
            for i, doc in enumerate(content):
                if isinstance(doc, Document):
                    combined_content.append(f"Document {i+1}:\n{doc.page_content}")
                else:
                    combined_content.append(f"Document {i+1}:\n{str(doc)}")
            return "\n\n".join(combined_content)
        else:
            return str(content)

    @traceable(name="direct_summarization") if LANGSMITH_AVAILABLE else lambda f: f
    async def _direct_summarization(
        self,
        text: str,
        summary_type: str,
        config: Dict[str, Any],
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform direct summarization for shorter content."""
        try:
            # Create prompt based on summary type
            if custom_prompt:
                prompt_template = custom_prompt
            else:
                prompt_template = self._get_prompt_template(summary_type, config)

            if LANGCHAIN_AVAILABLE and self.llm:
                # Use LangChain for summarization
                prompt = ChatPromptTemplate.from_template(prompt_template)
                chain = prompt | self.llm | StrOutputParser()

                summary = await chain.ainvoke({
                    "text": text,
                    "max_length": config["max_length"],
                    "style": config["style"],
                    "focus": config["focus"]
                })
            else:
                # Fallback to direct API call
                summary = await self._direct_api_summarization(text, prompt_template, config)

            # Post-process summary
            processed_summary = self._post_process_summary(summary, config)

            return {
                "summary": processed_summary,
                "method": "direct",
                "chunks_processed": 1,
                "total_tokens_estimated": len(text.split())
            }

        except Exception as e:
            logger.error(f"Error in direct summarization: {e}")
            raise

    @traceable(name="chunked_summarization") if LANGSMITH_AVAILABLE else lambda f: f
    async def _chunked_summarization(
        self,
        text: str,
        summary_type: str,
        config: Dict[str, Any],
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform chunked summarization for longer content."""
        try:
            # Split text into chunks
            chunks = self._split_text_into_chunks(text)
            logger.info(f"Split content into {len(chunks)} chunks")

            # Summarize each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")

                chunk_result = await self._direct_summarization(
                    chunk,
                    "brief",  # Use brief summaries for chunks
                    self.summary_configs["brief"],
                    custom_prompt
                )
                chunk_summaries.append(chunk_result["summary"])

            # Combine chunk summaries
            combined_summary = "\n\n".join(chunk_summaries)

            # Final summarization of combined summaries
            final_result = await self._direct_summarization(
                combined_summary,
                summary_type,
                config,
                custom_prompt
            )

            final_result.update({
                "method": "chunked",
                "chunks_processed": len(chunks),
                "total_tokens_estimated": len(text.split())
            })

            return final_result

        except Exception as e:
            logger.error(f"Error in chunked summarization: {e}")
            raise

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.max_chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                sentence_end = text.rfind('.', start + self.max_chunk_size - 200, end)
                if sentence_end > start:
                    end = sentence_end + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.overlap_size
            if start >= len(text):
                break

        return chunks

    def _get_prompt_template(self, summary_type: str, config: Dict[str, Any]) -> str:
        """Get prompt template based on summary type."""
        base_instruction = f"""You are a professional research assistant. Summarize the following text in a {config['style']} manner, focusing on {config['focus'].replace('_', ' ')}.

Keep the summary under {config['max_length']} words."""

        if summary_type == "brief":
            return base_instruction + """

Text to summarize:
{text}

Provide a brief, concise summary in 2-3 sentences:"""

        elif summary_type == "bullet_points":
            return base_instruction + """

Text to summarize:
{text}

Provide a structured summary as bullet points:
•"""

        elif summary_type == "executive":
            return base_instruction + """

Text to summarize:
{text}

Provide an executive summary focusing on key insights, implications, and actionable conclusions:"""

        elif summary_type == "technical":
            return base_instruction + """

Text to summarize:
{text}

Provide a technical summary focusing on methodology, data, results, and technical details:"""

        else:  # detailed
            return base_instruction + """

Text to summarize:
{text}

Provide a comprehensive summary covering all major points, methodology, findings, and conclusions:"""

    async def _direct_api_summarization(self, text: str, prompt_template: str, config: Dict[str, Any]) -> str:
        """Fallback summarization using direct API call."""
        if not HTTPX_AVAILABLE:
            return f"Summarization not available - missing dependencies. Text length: {len(text)} characters."

        api_key = os.getenv("EURI_API_KEY")
        if not api_key:
            return f"Summarization not available - API key not configured. Text length: {len(text)} characters."

        try:
            base_url = os.getenv("EURI_BASE_URL", "https://api.euron.one/api/v1/euri/alpha/chat/completions")
            model = os.getenv("EURI_MODEL", "gpt-4.1-nano")

            # Format prompt
            formatted_prompt = prompt_template.format(
                text=text[:3000],  # Truncate for API limits
                max_length=config["max_length"],
                style=config["style"],
                focus=config["focus"]
            )

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": formatted_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": min(config["max_length"] * 2, 1000)
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(base_url, headers=headers, json=payload)
                response.raise_for_status()

                data = response.json()
                return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Error in direct API summarization: {e}")
            return f"Error generating summary via API: {str(e)}"

    def _post_process_summary(self, summary: str, config: Dict[str, Any]) -> str:
        """Post-process the generated summary."""
        # Remove common prefixes from LLM responses
        prefixes_to_remove = [
            "Here is a summary:",
            "Summary:",
            "Here's a summary:",
            "The summary is:",
            "Brief summary:",
            "Detailed summary:",
        ]

        cleaned_summary = summary.strip()
        for prefix in prefixes_to_remove:
            if cleaned_summary.lower().startswith(prefix.lower()):
                cleaned_summary = cleaned_summary[len(prefix):].strip()

        # Ensure summary doesn't exceed max length (word count)
        words = cleaned_summary.split()
        if len(words) > config["max_length"]:
            cleaned_summary = " ".join(words[:config["max_length"]]) + "..."

        return cleaned_summary

    @traceable(name="extract_key_insights") if LANGSMITH_AVAILABLE else lambda f: f
    async def extract_key_insights(self, content: Union[str, Document, List[Document]]) -> Dict[str, Any]:
        """Extract key insights from content."""
        try:
            text_content = self._prepare_content(content)

            insight_prompt = """Extract the key insights, findings, and important conclusions from the following text. Focus on:
1. Main discoveries or findings
2. Methodological innovations
3. Practical implications
4. Future research directions

Text:
{text}

Key Insights:"""

            if LANGCHAIN_AVAILABLE and self.llm:
                prompt = ChatPromptTemplate.from_template(insight_prompt)
                chain = prompt | self.llm | StrOutputParser()
                insights = await chain.ainvoke({"text": text_content[:4000]})
            else:
                insights = await self._direct_api_summarization(
                    text_content[:4000],
                    insight_prompt,
                    {"max_length": 300, "style": "analytical", "focus": "insights"}
                )

            return {
                "insights": insights,
                "extraction_method": "llm_analysis",
                "content_length": len(text_content),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return {
                "insights": f"Error extracting insights: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the summarization tool."""
        return {
            "status": "healthy",
            "langchain_available": LANGCHAIN_AVAILABLE,
            "langsmith_available": LANGSMITH_AVAILABLE,
            "httpx_available": HTTPX_AVAILABLE,
            "llm_configured": self.llm is not None,
            "euri_api_configured": bool(os.getenv("EURI_API_KEY")),
            "supported_summary_types": list(self.summary_configs.keys()),
            "max_chunk_size": self.max_chunk_size,
            "overlap_size": self.overlap_size
        }
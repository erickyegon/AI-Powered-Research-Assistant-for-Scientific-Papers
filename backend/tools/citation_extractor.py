"""
Professional Citation Extraction Tool for AI Research Assistant.
Extracts, formats, and validates academic citations with multiple format support.
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
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


class CitationExtractor:
    """
    Professional citation extraction and formatting tool.
    Supports multiple citation formats and provides validation.
    """

    def __init__(self, llm=None):
        """Initialize the citation extractor."""
        self.llm = llm

        # Citation format patterns
        self.citation_patterns = {
            "apa": {
                "pattern": r'([A-Z][a-zA-Z\s,&]+)\s*\((\d{4}[a-z]?)\)\.\s*([^.]+)\.\s*([^.]+)\.',
                "description": "APA format (Author, Year, Title, Journal)"
            },
            "mla": {
                "pattern": r'([A-Z][a-zA-Z\s,]+)\.\s*"([^"]+)"\s*([^,]+),\s*(\d{4})',
                "description": "MLA format (Author, Title, Journal, Year)"
            },
            "chicago": {
                "pattern": r'([A-Z][a-zA-Z\s,]+)\.\s*"([^"]+)"\s*([^.]+)\s*(\d{4})',
                "description": "Chicago format (Author, Title, Journal, Year)"
            },
            "ieee": {
                "pattern": r'\[(\d+)\]\s*([A-Z][a-zA-Z\s,&]+),\s*"([^"]+),"\s*([^,]+),\s*(\d{4})',
                "description": "IEEE format [Number] Author, Title, Journal, Year"
            },
            "harvard": {
                "pattern": r'([A-Z][a-zA-Z\s,&]+)\s*(\d{4})\s*([^.]+)\.\s*([^.]+)\.',
                "description": "Harvard format (Author Year Title Journal)"
            }
        }

        # Common citation indicators
        self.citation_indicators = [
            "References", "Bibliography", "Works Cited", "Literature Cited",
            "Citations", "Sources", "Further Reading"
        ]

        # DOI pattern
        self.doi_pattern = r'10\.\d{4,}\/[^\s]+'

        # URL pattern
        self.url_pattern = r'https?:\/\/[^\s]+'

        logger.info("✅ Citation extractor initialized")

    @traceable(name="extract_citations") if LANGSMITH_AVAILABLE else lambda f: f
    async def extract_citations(
        self,
        content: Union[str, Document, List[Document]],
        format_style: str = "apa",
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Extract citations from content.

        Args:
            content: Text content, Document, or list of Documents
            format_style: Desired citation format (apa, mla, chicago, ieee, harvard)
            include_metadata: Whether to include extraction metadata

        Returns:
            Dictionary containing extracted citations and metadata
        """
        try:
            # Prepare content
            text_content = self._prepare_content(content)

            # Extract citations using multiple methods
            regex_citations = self._extract_with_regex(text_content)
            section_citations = self._extract_from_sections(text_content)

            # Use LLM for intelligent extraction if available
            if LANGCHAIN_AVAILABLE and self.llm:
                llm_citations = await self._extract_with_llm(text_content, format_style)
            else:
                llm_citations = await self._extract_with_api(text_content, format_style)

            # Combine and deduplicate citations
            all_citations = self._combine_citations(regex_citations, section_citations, llm_citations)

            # Format citations
            formatted_citations = self._format_citations(all_citations, format_style)

            # Extract additional metadata
            metadata = {}
            if include_metadata:
                metadata = {
                    "extraction_methods": ["regex", "section_based", "llm"],
                    "total_citations_found": len(all_citations),
                    "formatted_citations_count": len(formatted_citations),
                    "format_style": format_style,
                    "content_length": len(text_content),
                    "timestamp": datetime.now().isoformat(),
                    "dois_found": len(re.findall(self.doi_pattern, text_content)),
                    "urls_found": len(re.findall(self.url_pattern, text_content))
                }

            return {
                "citations": formatted_citations,
                "raw_citations": all_citations,
                "metadata": metadata,
                "format_style": format_style
            }

        except Exception as e:
            logger.error(f"Error extracting citations: {e}")
            return {
                "citations": [],
                "raw_citations": [],
                "error": str(e),
                "format_style": format_style,
                "metadata": {"timestamp": datetime.now().isoformat()}
            }

    def _prepare_content(self, content: Union[str, Document, List[Document]]) -> str:
        """Prepare content for citation extraction."""
        if isinstance(content, str):
            return content
        elif isinstance(content, Document):
            return content.page_content
        elif isinstance(content, list):
            # Combine multiple documents
            combined_content = []
            for doc in content:
                if isinstance(doc, Document):
                    combined_content.append(doc.page_content)
                else:
                    combined_content.append(str(doc))
            return "\n\n".join(combined_content)
        else:
            return str(content)

    def _extract_with_regex(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations using regex patterns."""
        citations = []

        for format_name, pattern_info in self.citation_patterns.items():
            pattern = pattern_info["pattern"]
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)

            for match in matches:
                citation = {
                    "text": match.group(0),
                    "format": format_name,
                    "extraction_method": "regex",
                    "confidence": 0.7,
                    "position": match.start(),
                    "groups": match.groups()
                }
                citations.append(citation)

        # Extract DOIs
        doi_matches = re.finditer(self.doi_pattern, text)
        for match in doi_matches:
            citation = {
                "text": match.group(0),
                "type": "doi",
                "extraction_method": "regex",
                "confidence": 0.9,
                "position": match.start()
            }
            citations.append(citation)

        # Extract URLs
        url_matches = re.finditer(self.url_pattern, text)
        for match in url_matches:
            citation = {
                "text": match.group(0),
                "type": "url",
                "extraction_method": "regex",
                "confidence": 0.8,
                "position": match.start()
            }
            citations.append(citation)

        logger.info(f"Extracted {len(citations)} citations using regex")
        return citations

    def _extract_from_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations from reference sections."""
        citations = []

        # Find reference sections
        for indicator in self.citation_indicators:
            # Look for section headers
            pattern = rf'\b{re.escape(indicator)}\b.*?(?=\n\n|\n[A-Z]|\Z)'
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)

            for match in matches:
                section_text = match.group(0)

                # Split into individual citations (assuming each line is a citation)
                lines = section_text.split('\n')[1:]  # Skip header

                for i, line in enumerate(lines):
                    line = line.strip()
                    if len(line) > 20:  # Filter out short lines
                        citation = {
                            "text": line,
                            "section": indicator,
                            "extraction_method": "section_based",
                            "confidence": 0.8,
                            "line_number": i + 1
                        }
                        citations.append(citation)

        logger.info(f"Extracted {len(citations)} citations from sections")
        return citations

    @traceable(name="extract_citations_llm") if LANGSMITH_AVAILABLE else lambda f: f
    async def _extract_with_llm(self, text: str, format_style: str) -> List[Dict[str, Any]]:
        """Extract citations using LLM."""
        try:
            prompt_template = f"""Extract all academic citations from the following text. Format them in {format_style.upper()} style.

Text:
{{text}}

Please extract citations and format them as a numbered list. Include:
1. Author names
2. Publication year
3. Title
4. Journal/Conference name
5. DOI or URL if available

Citations:"""

            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | StrOutputParser()

            # Truncate text for LLM processing
            truncated_text = text[:4000] if len(text) > 4000 else text

            result = await chain.ainvoke({"text": truncated_text})

            # Parse LLM response into structured citations
            citations = self._parse_llm_citations(result, format_style)

            logger.info(f"Extracted {len(citations)} citations using LLM")
            return citations

        except Exception as e:
            logger.error(f"Error in LLM citation extraction: {e}")
            return []

    async def _extract_with_api(self, text: str, format_style: str) -> List[Dict[str, Any]]:
        """Fallback citation extraction using direct API call."""
        if not HTTPX_AVAILABLE or not os.getenv("EURI_API_KEY"):
            return []

        try:
            api_key = os.getenv("EURI_API_KEY")
            base_url = os.getenv("EURI_BASE_URL", "https://api.euron.one/api/v1/euri/alpha/chat/completions")
            model = os.getenv("EURI_MODEL", "gpt-4.1-nano")

            prompt = f"""Extract all academic citations from the following text and format them in {format_style.upper()} style.

Text:
{text[:3000]}

Please extract citations as a numbered list:"""

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(base_url, headers=headers, json=payload)
                response.raise_for_status()

                data = response.json()
                result = data["choices"][0]["message"]["content"]

                # Parse API response into structured citations
                citations = self._parse_llm_citations(result, format_style)

                logger.info(f"Extracted {len(citations)} citations using API")
                return citations

        except Exception as e:
            logger.error(f"Error in API citation extraction: {e}")
            return []

    def _parse_llm_citations(self, llm_response: str, format_style: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured citations."""
        citations = []

        # Split by numbered list items
        lines = llm_response.split('\n')
        current_citation = ""

        for line in lines:
            line = line.strip()

            # Check if line starts with a number (citation item)
            if re.match(r'^\d+\.', line):
                # Save previous citation if exists
                if current_citation:
                    citation = {
                        "text": current_citation.strip(),
                        "format": format_style,
                        "extraction_method": "llm",
                        "confidence": 0.9
                    }
                    citations.append(citation)

                # Start new citation
                current_citation = re.sub(r'^\d+\.\s*', '', line)
            elif line and current_citation:
                # Continue current citation
                current_citation += " " + line

        # Add last citation
        if current_citation:
            citation = {
                "text": current_citation.strip(),
                "format": format_style,
                "extraction_method": "llm",
                "confidence": 0.9
            }
            citations.append(citation)

        return citations

    def _combine_citations(self, *citation_lists) -> List[Dict[str, Any]]:
        """Combine and deduplicate citations from multiple sources."""
        all_citations = []
        seen_texts = set()

        for citation_list in citation_lists:
            for citation in citation_list:
                # Simple deduplication based on text similarity
                citation_text = citation.get("text", "").lower().strip()

                # Skip if too similar to existing citation
                is_duplicate = False
                for seen_text in seen_texts:
                    if self._calculate_similarity(citation_text, seen_text) > 0.8:
                        is_duplicate = True
                        break

                if not is_duplicate and len(citation_text) > 10:
                    all_citations.append(citation)
                    seen_texts.add(citation_text)

        # Sort by confidence and position
        all_citations.sort(key=lambda x: (x.get("confidence", 0), -x.get("position", 0)), reverse=True)

        return all_citations

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _format_citations(self, citations: List[Dict[str, Any]], format_style: str) -> List[str]:
        """Format citations according to the specified style."""
        formatted = []

        for i, citation in enumerate(citations, 1):
            citation_text = citation.get("text", "")

            # Basic formatting based on style
            if format_style.lower() == "apa":
                # Ensure APA format
                formatted_citation = self._ensure_apa_format(citation_text)
            elif format_style.lower() == "mla":
                formatted_citation = self._ensure_mla_format(citation_text)
            elif format_style.lower() == "chicago":
                formatted_citation = self._ensure_chicago_format(citation_text)
            elif format_style.lower() == "ieee":
                formatted_citation = f"[{i}] {citation_text}"
            else:
                formatted_citation = citation_text

            formatted.append(formatted_citation)

        return formatted

    def _ensure_apa_format(self, citation: str) -> str:
        """Ensure citation follows APA format."""
        # Basic APA formatting (simplified)
        if not citation.endswith('.'):
            citation += '.'
        return citation

    def _ensure_mla_format(self, citation: str) -> str:
        """Ensure citation follows MLA format."""
        # Basic MLA formatting (simplified)
        return citation

    def _ensure_chicago_format(self, citation: str) -> str:
        """Ensure citation follows Chicago format."""
        # Basic Chicago formatting (simplified)
        return citation

    @traceable(name="validate_citations") if LANGSMITH_AVAILABLE else lambda f: f
    async def validate_citations(self, citations: List[str]) -> Dict[str, Any]:
        """Validate extracted citations for completeness and format."""
        validation_results = []

        for i, citation in enumerate(citations):
            result = {
                "citation_index": i + 1,
                "citation": citation,
                "is_valid": True,
                "issues": [],
                "completeness_score": 0.0
            }

            # Check for required elements
            has_author = bool(re.search(r'[A-Z][a-z]+,?\s+[A-Z]', citation))
            has_year = bool(re.search(r'\b(19|20)\d{2}\b', citation))
            has_title = len(citation) > 20  # Simple title check

            completeness_elements = [has_author, has_year, has_title]
            result["completeness_score"] = sum(completeness_elements) / len(completeness_elements)

            if not has_author:
                result["issues"].append("Missing or unclear author")
            if not has_year:
                result["issues"].append("Missing publication year")
            if not has_title:
                result["issues"].append("Citation appears incomplete")

            if result["issues"]:
                result["is_valid"] = False

            validation_results.append(result)

        # Overall statistics
        valid_count = sum(1 for r in validation_results if r["is_valid"])
        avg_completeness = sum(r["completeness_score"] for r in validation_results) / len(validation_results) if validation_results else 0

        return {
            "validation_results": validation_results,
            "summary": {
                "total_citations": len(citations),
                "valid_citations": valid_count,
                "invalid_citations": len(citations) - valid_count,
                "average_completeness_score": avg_completeness,
                "validation_timestamp": datetime.now().isoformat()
            }
        }

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the citation extractor."""
        return {
            "status": "healthy",
            "langchain_available": LANGCHAIN_AVAILABLE,
            "langsmith_available": LANGSMITH_AVAILABLE,
            "httpx_available": HTTPX_AVAILABLE,
            "llm_configured": self.llm is not None,
            "euri_api_configured": bool(os.getenv("EURI_API_KEY")),
            "supported_formats": list(self.citation_patterns.keys()),
            "citation_indicators": self.citation_indicators,
            "patterns_loaded": len(self.citation_patterns)
        }

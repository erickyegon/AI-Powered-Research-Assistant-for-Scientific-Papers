"""Professional query processing and enhancement utilities."""

import os
import re
import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from ..utils.eur_client import create_euri_llm

logger = logging.getLogger(__name__)


# Simple in-memory cache for development
class SimpleCache:
    """Simple in-memory cache for query processing."""

    def __init__(self):
        self._cache = {}

    async def get_cached_result(self, key: str, namespace: str = "default"):
        """Get cached result."""
        cache_key = f"{namespace}:{key}"
        return self._cache.get(cache_key)

    async def cache_result(self, key: str, value: Any, ttl: int = 3600, namespace: str = "default"):
        """Cache a result."""
        cache_key = f"{namespace}:{key}"
        self._cache[cache_key] = value
        # Note: TTL not implemented in this simple version

# Global cache instance
cache_manager = SimpleCache()


class QueryProcessor:
    """
    Advanced query processing and enhancement tool.

    Features:
    - Query expansion and enhancement
    - Key term extraction
    - Intent classification
    - Query reformulation
    - Semantic analysis
    """

    def __init__(self):
        self.llm = create_euri_llm()
        self.scientific_terms = self._load_scientific_terms()

    def _load_scientific_terms(self) -> Set[str]:
        """Load common scientific terms for enhancement."""
        # In production, this could be loaded from a file or database
        return {
            "hypothesis", "methodology", "analysis", "results", "conclusion",
            "experiment", "data", "statistical", "significant", "correlation",
            "causation", "variable", "control", "sample", "population",
            "research", "study", "investigation", "findings", "evidence",
            "theory", "model", "framework", "approach", "technique",
            "algorithm", "machine learning", "artificial intelligence",
            "neural network", "deep learning", "classification", "regression",
            "clustering", "optimization", "validation", "evaluation"
        }

    async def enhance_query(self, query: str) -> str:
        """
        Enhance a user query for better retrieval.

        Args:
            query: Original user query

        Returns:
            Enhanced query
        """
        try:
            # Check cache first
            cache_key = f"enhanced_query:{hash(query)}"
            cached_result = await cache_manager.get_cached_result(
                cache_key,
                namespace="query_processing"
            )

            if cached_result:
                logger.debug("Query enhancement cache hit", query=query[:50])
                return cached_result

            # Enhancement prompt
            enhancement_prompt = PromptTemplate.from_template("""
You are an expert at enhancing research queries for better document retrieval.
Your task is to expand and improve the given query while maintaining its original intent.

Original Query: {query}

Please enhance this query by:
1. Adding relevant scientific terminology
2. Including synonyms and related concepts
3. Expanding abbreviations if present
4. Adding context that would help find relevant research papers
5. Maintaining the core question/intent

Enhanced Query: """)

            chain = enhancement_prompt | self.llm | StrOutputParser()

            enhanced_query = await chain.ainvoke({"query": query})

            # Clean up the response
            enhanced_query = self._clean_enhanced_query(enhanced_query)

            # Cache the result
            await cache_manager.cache_result(
                cache_key,
                enhanced_query,
                ttl=3600,  # 1 hour
                namespace="query_processing"
            )

            logger.info(
                "Query enhanced",
                original=query[:50],
                enhanced=enhanced_query[:50]
            )

            return enhanced_query

        except Exception as e:
            logger.error("Error enhancing query", error=str(e), query=query)
            return query  # Return original query if enhancement fails

    async def extract_key_terms(self, query: str) -> List[str]:
        """
        Extract key terms and concepts from a query.

        Args:
            query: Query text

        Returns:
            List of key terms
        """
        try:
            # Check cache first
            cache_key = f"key_terms:{hash(query)}"
            cached_result = await cache_manager.get_cached_result(
                cache_key,
                namespace="query_processing"
            )

            if cached_result:
                logger.debug("Key terms cache hit", query=query[:50])
                return cached_result

            # Key term extraction prompt
            extraction_prompt = PromptTemplate.from_template("""
Extract the most important key terms and concepts from this research query.
Focus on scientific terms, methodologies, domains, and specific concepts.

Query: {query}

Return the key terms as a comma-separated list (maximum 10 terms):
""")

            chain = extraction_prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({"query": query})

            # Parse the result
            key_terms = self._parse_key_terms(result)

            # Add terms found in our scientific vocabulary
            additional_terms = self._find_scientific_terms(query)
            key_terms.extend(additional_terms)

            # Remove duplicates and limit
            key_terms = list(set(key_terms))[:10]

            # Cache the result
            await cache_manager.cache_result(
                cache_key,
                key_terms,
                ttl=3600,
                namespace="query_processing"
            )

            logger.info("Key terms extracted", query=query[:50], terms=key_terms)
            return key_terms

        except Exception as e:
            logger.error("Error extracting key terms", error=str(e))
            return self._fallback_key_terms(query)

    async def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify the intent of a research query.

        Args:
            query: Query text

        Returns:
            Intent classification with confidence scores
        """
        try:
            # Check cache first
            cache_key = f"intent:{hash(query)}"
            cached_result = await cache_manager.get_cached_result(
                cache_key,
                namespace="query_processing"
            )

            if cached_result:
                return cached_result

            # Intent classification prompt
            intent_prompt = PromptTemplate.from_template("""
Classify the intent of this research query into one or more categories.

Query: {query}

Categories:
1. FACTUAL - Seeking specific facts or information
2. METHODOLOGICAL - Asking about research methods or techniques
3. COMPARATIVE - Comparing different approaches, studies, or results
4. ANALYTICAL - Requesting analysis or interpretation
5. DEFINITIONAL - Seeking definitions or explanations
6. CAUSAL - Asking about cause-and-effect relationships
7. PREDICTIVE - Seeking predictions or future trends
8. EVALUATIVE - Requesting evaluation or assessment

Provide your classification as:
PRIMARY_INTENT: [category]
CONFIDENCE: [0-100]
SECONDARY_INTENT: [category] (if applicable)
REASONING: [brief explanation]
""")

            chain = intent_prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({"query": query})

            # Parse the classification
            intent_data = self._parse_intent_classification(result)

            # Cache the result
            await cache_manager.cache_result(
                cache_key,
                intent_data,
                ttl=3600,
                namespace="query_processing"
            )

            logger.info("Intent classified", query=query[:50], intent=intent_data)
            return intent_data

        except Exception as e:
            logger.error("Error classifying intent", error=str(e))
            return {
                "primary_intent": "FACTUAL",
                "confidence": 50,
                "secondary_intent": None,
                "reasoning": "Default classification due to error"
            }

    async def reformulate_query(self, query: str, context: str = "") -> List[str]:
        """
        Generate alternative formulations of a query.

        Args:
            query: Original query
            context: Additional context for reformulation

        Returns:
            List of reformulated queries
        """
        try:
            # Reformulation prompt
            reformulation_prompt = PromptTemplate.from_template("""
Generate 3-5 alternative formulations of this research query that would help find relevant scientific papers.
Each reformulation should approach the same question from a different angle or use different terminology.

Original Query: {query}
Context: {context}

Alternative Formulations:
1. """)

            chain = reformulation_prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({
                "query": query,
                "context": context or "No additional context provided"
            })

            # Parse reformulations
            reformulations = self._parse_reformulations(result)

            logger.info("Query reformulated", original=query[:50], count=len(reformulations))
            return reformulations

        except Exception as e:
            logger.error("Error reformulating query", error=str(e))
            return [query]  # Return original query if reformulation fails

    def _clean_enhanced_query(self, enhanced_query: str) -> str:
        """Clean up enhanced query response."""
        # Remove common prefixes from LLM responses
        prefixes_to_remove = [
            "Enhanced Query:",
            "Enhanced query:",
            "Here is the enhanced query:",
            "The enhanced query is:",
        ]

        cleaned = enhanced_query.strip()
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        # Remove quotes if present
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]

        return cleaned

    def _parse_key_terms(self, result: str) -> List[str]:
        """Parse key terms from LLM response."""
        # Split by commas and clean up
        terms = [term.strip() for term in result.split(',')]

        # Filter out empty terms and common words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_terms = []

        for term in terms:
            # Remove quotes and extra whitespace
            term = term.strip().strip('"\'')

            # Skip if empty or stop word
            if term and term.lower() not in stop_words and len(term) > 2:
                filtered_terms.append(term)

        return filtered_terms[:10]  # Limit to 10 terms

    def _find_scientific_terms(self, query: str) -> List[str]:
        """Find scientific terms in the query using our vocabulary."""
        query_lower = query.lower()
        found_terms = []

        for term in self.scientific_terms:
            if term in query_lower:
                found_terms.append(term)

        return found_terms

    def _fallback_key_terms(self, query: str) -> List[str]:
        """Extract key terms using simple heuristics as fallback."""
        import re

        # Remove common words and extract meaningful terms
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())

        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'that', 'this', 'these',
            'those', 'can', 'could', 'would', 'should', 'will', 'may', 'might', 'must'
        }

        key_terms = [word for word in words if word not in stop_words]

        # Add any scientific terms found
        key_terms.extend(self._find_scientific_terms(query))

        return list(set(key_terms))[:10]

    def _parse_intent_classification(self, result: str) -> Dict[str, Any]:
        """Parse intent classification from LLM response."""
        intent_data = {
            "primary_intent": "FACTUAL",
            "confidence": 50,
            "secondary_intent": None,
            "reasoning": "Default classification"
        }

        lines = result.split('\n')
        for line in lines:
            line = line.strip()

            if line.startswith('PRIMARY_INTENT:'):
                intent_data["primary_intent"] = line.split(':', 1)[1].strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = int(re.search(r'\d+', line.split(':', 1)[1]).group())
                    intent_data["confidence"] = confidence
                except:
                    pass
            elif line.startswith('SECONDARY_INTENT:'):
                secondary = line.split(':', 1)[1].strip()
                if secondary and secondary != "None":
                    intent_data["secondary_intent"] = secondary
            elif line.startswith('REASONING:'):
                intent_data["reasoning"] = line.split(':', 1)[1].strip()

        return intent_data

    def _parse_reformulations(self, result: str) -> List[str]:
        """Parse reformulated queries from LLM response."""
        reformulations = []

        # Split by numbered lines
        lines = result.split('\n')
        for line in lines:
            line = line.strip()

            # Look for numbered items
            if re.match(r'^\d+\.', line):
                reformulation = re.sub(r'^\d+\.\s*', '', line).strip()
                if reformulation:
                    reformulations.append(reformulation)

        # If no numbered items found, try splitting by sentences
        if not reformulations:
            sentences = result.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:
                    reformulations.append(sentence)

        return reformulations[:5]  # Limit to 5 reformulations

    async def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze the complexity of a query.

        Args:
            query: Query to analyze

        Returns:
            Complexity analysis
        """
        analysis = {
            "word_count": len(query.split()),
            "character_count": len(query),
            "question_words": 0,
            "scientific_terms": 0,
            "complexity_score": 0.0,
            "complexity_level": "simple"
        }

        # Count question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        query_lower = query.lower()

        for word in question_words:
            if word in query_lower:
                analysis["question_words"] += 1

        # Count scientific terms
        for term in self.scientific_terms:
            if term in query_lower:
                analysis["scientific_terms"] += 1

        # Calculate complexity score
        score = 0
        score += min(analysis["word_count"] / 10, 3)  # Word count factor
        score += analysis["question_words"] * 0.5  # Question complexity
        score += analysis["scientific_terms"] * 0.3  # Scientific terminology

        analysis["complexity_score"] = round(score, 2)

        # Determine complexity level
        if score < 2:
            analysis["complexity_level"] = "simple"
        elif score < 4:
            analysis["complexity_level"] = "moderate"
        else:
            analysis["complexity_level"] = "complex"

        return analysis

    async def suggest_search_strategies(self, query: str) -> List[Dict[str, str]]:
        """
        Suggest search strategies based on query analysis.

        Args:
            query: Query to analyze

        Returns:
            List of search strategy suggestions
        """
        try:
            # Analyze query first
            complexity = await self.analyze_query_complexity(query)
            intent = await self.classify_intent(query)

            strategies = []

            # Strategy based on complexity
            if complexity["complexity_level"] == "simple":
                strategies.append({
                    "strategy": "Direct Search",
                    "description": "Use the query as-is for direct document matching",
                    "reason": "Query is simple and specific"
                })

            elif complexity["complexity_level"] == "complex":
                strategies.append({
                    "strategy": "Query Decomposition",
                    "description": "Break down the complex query into simpler sub-queries",
                    "reason": "Complex query may benefit from decomposition"
                })

            # Strategy based on intent
            primary_intent = intent.get("primary_intent", "FACTUAL")

            if primary_intent == "COMPARATIVE":
                strategies.append({
                    "strategy": "Comparative Analysis",
                    "description": "Search for documents that compare different approaches or methods",
                    "reason": "Query seeks comparison between different concepts"
                })

            elif primary_intent == "METHODOLOGICAL":
                strategies.append({
                    "strategy": "Methodology Focus",
                    "description": "Prioritize documents that describe research methods and techniques",
                    "reason": "Query is focused on research methodology"
                })

            elif primary_intent == "CAUSAL":
                strategies.append({
                    "strategy": "Causal Relationship Search",
                    "description": "Look for documents that establish cause-and-effect relationships",
                    "reason": "Query seeks causal relationships"
                })

            # Default strategies
            if not strategies:
                strategies.append({
                    "strategy": "Semantic Search",
                    "description": "Use semantic similarity to find relevant documents",
                    "reason": "Standard approach for factual queries"
                })

            # Always suggest query expansion
            strategies.append({
                "strategy": "Query Expansion",
                "description": "Expand the query with synonyms and related terms",
                "reason": "May find additional relevant documents"
            })

            return strategies

        except Exception as e:
            logger.error("Error suggesting search strategies", error=str(e))
            return [{
                "strategy": "Basic Search",
                "description": "Perform basic semantic search",
                "reason": "Default strategy due to analysis error"
            }]

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the query processor."""
        status = {
            "llm_available": self.llm is not None,
            "scientific_terms_loaded": len(self.scientific_terms) > 0,
            "cache_available": cache_manager is not None
        }

        # Test query enhancement
        try:
            test_query = "What is machine learning?"
            enhanced = await self.enhance_query(test_query)
            status["enhancement_test"] = len(enhanced) > len(test_query)
        except Exception as e:
            status["enhancement_test"] = False
            status["enhancement_error"] = str(e)

        return status


# Factory function
def create_query_processor() -> QueryProcessor:
    """Factory function to create a query processor instance."""
    return QueryProcessor()


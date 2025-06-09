"""Professional embedding tool for document and query vectorization."""

import os
import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Fallback Document class
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    pinecone = None

logger = logging.getLogger(__name__)


# Simple in-memory cache
class SimpleCache:
    """Simple in-memory cache for embeddings."""

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

# Global cache instance
cache_manager = SimpleCache()


class EmbeddingTool:
    """
    Production-grade embedding tool with caching and vector store integration.

    Features:
    - Sentence transformer embeddings
    - Pinecone vector store integration
    - Intelligent caching
    - Batch processing
    - Error handling and retry logic
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.pinecone_index = None
        self._initialize_model()
        self._initialize_pinecone()

    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model initialized", model=self.model_name)
        except Exception as e:
            logger.error("Failed to initialize embedding model", error=str(e))
            raise

    def _initialize_pinecone(self):
        """Initialize Pinecone vector store."""
        if not PINECONE_AVAILABLE:
            logger.warning("Pinecone not available, skipping initialization")
            return

        try:
            api_key = os.getenv("PINECONE_API_KEY")
            environment = os.getenv("PINECONE_ENVIRONMENT")
            index_name = os.getenv("PINECONE_INDEX_NAME", "research-papers")
            embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))

            if not api_key or not environment:
                logger.warning("Pinecone credentials not configured, skipping initialization")
                return

            pinecone.init(api_key=api_key, environment=environment)

            # Create index if it doesn't exist
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=embedding_dimension,
                    metric="cosine"
                )
                logger.info("Pinecone index created", index=index_name)

            self.pinecone_index = pinecone.Index(index_name)
            logger.info("Pinecone initialized", index=index_name)

        except Exception as e:
            logger.error("Failed to initialize Pinecone", error=str(e))
            # Continue without Pinecone for development
            self.pinecone_index = None

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            # Check cache first
            cache_key = f"embedding:{hash(text)}"
            cached_embedding = await cache_manager.get_cached_result(
                cache_key,
                namespace="embeddings"
            )

            if cached_embedding:
                logger.debug("Embedding cache hit", text_length=len(text))
                return cached_embedding

            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=False).tolist()

            # Cache the result
            await cache_manager.cache_result(
                cache_key,
                embedding,
                ttl=86400,  # 24 hours
                namespace="embeddings"
            )

            logger.debug("Embedding generated", text_length=len(text), dim=len(embedding))
            return embedding

        except Exception as e:
            logger.error("Error generating embedding", error=str(e), text_length=len(text))
            raise

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            # Check cache for each text
            embeddings = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                cache_key = f"embedding:{hash(text)}"
                cached_embedding = await cache_manager.get_cached_result(
                    cache_key,
                    namespace="embeddings"
                )

                if cached_embedding:
                    embeddings.append(cached_embedding)
                else:
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Generate embeddings for uncached texts
            if uncached_texts:
                batch_embeddings = self.model.encode(
                    uncached_texts,
                    convert_to_tensor=False,
                    batch_size=32
                ).tolist()

                # Cache and insert results
                for idx, embedding in zip(uncached_indices, batch_embeddings):
                    embeddings[idx] = embedding

                    # Cache the result
                    cache_key = f"embedding:{hash(texts[idx])}"
                    await cache_manager.cache_result(
                        cache_key,
                        embedding,
                        ttl=86400,
                        namespace="embeddings"
                    )

            logger.info(
                "Batch embeddings generated",
                total=len(texts),
                cached=len(texts) - len(uncached_texts),
                generated=len(uncached_texts)
            )

            return embeddings

        except Exception as e:
            logger.error("Error generating batch embeddings", error=str(e))
            raise

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query (alias for embed_text).

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        return await self.embed_text(query)

    async def embed_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for documents and prepare for vector store.

        Args:
            documents: List of documents to embed

        Returns:
            List of document embeddings with metadata
        """
        try:
            # Extract text content
            texts = [doc.page_content for doc in documents]

            # Generate embeddings
            embeddings = await self.embed_texts(texts)

            # Prepare document embeddings with metadata
            doc_embeddings = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                doc_embedding = {
                    "id": f"doc_{i}_{hash(doc.page_content)}",
                    "embedding": embedding,
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "timestamp": datetime.utcnow().isoformat()
                }
                doc_embeddings.append(doc_embedding)

            logger.info("Document embeddings generated", count=len(doc_embeddings))
            return doc_embeddings

        except Exception as e:
            logger.error("Error generating document embeddings", error=str(e))
            raise

    async def store_embeddings(self, doc_embeddings: List[Dict[str, Any]]) -> bool:
        """
        Store embeddings in Pinecone vector store.

        Args:
            doc_embeddings: List of document embeddings

        Returns:
            True if successful, False otherwise
        """
        if not self.pinecone_index:
            logger.warning("Pinecone not available, skipping storage")
            return False

        try:
            # Prepare vectors for Pinecone
            vectors = []
            for doc_emb in doc_embeddings:
                vector = (
                    doc_emb["id"],
                    doc_emb["embedding"],
                    {
                        "text": doc_emb["text"][:1000],  # Truncate for metadata
                        "timestamp": doc_emb["timestamp"],
                        **doc_emb["metadata"]
                    }
                )
                vectors.append(vector)

            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.pinecone_index.upsert(vectors=batch)

            logger.info("Embeddings stored in Pinecone", count=len(vectors))
            return True

        except Exception as e:
            logger.error("Error storing embeddings in Pinecone", error=str(e))
            return False

    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        threshold: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search in vector store.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Similarity threshold
            filter_metadata: Optional metadata filters

        Returns:
            List of similar documents
        """
        if not self.pinecone_index:
            logger.warning("Pinecone not available, returning empty results")
            return []

        try:
            # Query Pinecone
            query_response = self.pinecone_index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                filter=filter_metadata
            )

            # Convert results to Documents
            documents = []
            for match in query_response.matches:
                if match.score >= threshold:
                    metadata = match.metadata.copy()
                    text = metadata.pop("text", "")

                    doc = Document(
                        page_content=text,
                        metadata={
                            **metadata,
                            "similarity_score": match.score,
                            "vector_id": match.id
                        }
                    )
                    documents.append(doc)

            logger.info(
                "Similarity search completed",
                query_results=len(query_response.matches),
                filtered_results=len(documents),
                threshold=threshold
            )

            return documents

        except Exception as e:
            logger.error("Error in similarity search", error=str(e))
            return []

    async def delete_embeddings(self, document_ids: List[str]) -> bool:
        """
        Delete embeddings from vector store.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.pinecone_index:
            logger.warning("Pinecone not available, skipping deletion")
            return False

        try:
            self.pinecone_index.delete(ids=document_ids)
            logger.info("Embeddings deleted", count=len(document_ids))
            return True

        except Exception as e:
            logger.error("Error deleting embeddings", error=str(e))
            return False

    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.

        Returns:
            Index statistics
        """
        if not self.pinecone_index:
            return {"error": "Pinecone not available"}

        try:
            stats = self.pinecone_index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }

        except Exception as e:
            logger.error("Error getting index stats", error=str(e))
            return {"error": str(e)}

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error("Error computing similarity", error=str(e))
            return 0.0

    async def find_similar_documents(
        self,
        document: Document,
        k: int = 5,
        threshold: float = 0.8
    ) -> List[Tuple[Document, float]]:
        """
        Find documents similar to a given document.

        Args:
            document: Reference document
            k: Number of similar documents to find
            threshold: Similarity threshold

        Returns:
            List of (document, similarity_score) tuples
        """
        try:
            # Generate embedding for the document
            doc_embedding = await self.embed_text(document.page_content)

            # Search for similar documents
            similar_docs = await self.similarity_search(
                doc_embedding,
                k=k + 1,  # +1 to account for the document itself
                threshold=threshold
            )

            # Filter out the original document and return with scores
            results = []
            for doc in similar_docs:
                if doc.page_content != document.page_content:
                    similarity_score = doc.metadata.get("similarity_score", 0.0)
                    results.append((doc, similarity_score))

            return results[:k]

        except Exception as e:
            logger.error("Error finding similar documents", error=str(e))
            return []

    async def cluster_documents(
        self,
        documents: List[Document],
        n_clusters: int = 5
    ) -> Dict[int, List[Document]]:
        """
        Cluster documents based on their embeddings.

        Args:
            documents: List of documents to cluster
            n_clusters: Number of clusters

        Returns:
            Dictionary mapping cluster IDs to document lists
        """
        try:
            from sklearn.cluster import KMeans

            # Generate embeddings for all documents
            embeddings = await self.embed_texts([doc.page_content for doc in documents])

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Group documents by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(documents[i])

            logger.info(
                "Documents clustered",
                total_docs=len(documents),
                n_clusters=len(clusters)
            )

            return clusters

        except Exception as e:
            logger.error("Error clustering documents", error=str(e))
            return {0: documents}  # Return all documents in one cluster

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the embedding tool.

        Returns:
            Health status information
        """
        status = {
            "model_loaded": self.model is not None,
            "pinecone_connected": self.pinecone_index is not None,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Test embedding generation
        try:
            test_embedding = await self.embed_text("test")
            expected_dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))
            status["embedding_test"] = len(test_embedding) == expected_dimension
        except Exception as e:
            status["embedding_test"] = False
            status["embedding_error"] = str(e)

        # Test Pinecone connection
        if self.pinecone_index:
            try:
                stats = await self.get_index_stats()
                status["pinecone_stats"] = stats
                status["pinecone_test"] = "error" not in stats
            except Exception as e:
                status["pinecone_test"] = False
                status["pinecone_error"] = str(e)

        return status


# Factory function
def create_embedding_tool(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingTool:
    """
    Factory function to create an embedding tool instance.

    Args:
        model_name: Name of the sentence transformer model

    Returns:
        Configured EmbeddingTool instance
    """
    return EmbeddingTool(model_name=model_name)


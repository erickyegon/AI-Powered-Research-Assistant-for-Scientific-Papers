"""Configuration for AI-Powered Research Assistant for Scientific Papers."""

import os
from typing import Optional, List

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables from .env file won't be loaded.")

# Try to import pydantic, but work without it if not available
try:
    from pydantic import BaseSettings, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Create a simple fallback class
    class BaseSettings:
        pass
    def Field(**kwargs):
        return None

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings with environment variable support."""

    def __init__(self):
        # Application
        self.app_name = "AI-Powered Research Assistant"
        self.app_version = "1.0.0"
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.environment = os.getenv("ENVIRONMENT", "development")

        # Server
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.workers = int(os.getenv("WORKERS", "1"))

        # Security
        self.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        self.algorithm = os.getenv("ALGORITHM", "HS256")

        # Euri LLM API
        self.euri_api_key = os.getenv("EURI_API_KEY", "")
        self.euri_base_url = os.getenv(
            "EURI_BASE_URL",
            "https://api.euron.one/api/v1/euri/alpha/chat/completions"
        )
        self.euri_model = os.getenv("EURI_MODEL", "gpt-4.1-nano")
        self.euri_temperature = float(os.getenv("EURI_TEMPERATURE", "0.7"))
        self.euri_max_tokens = int(os.getenv("EURI_MAX_TOKENS", "2000"))
        self.euri_timeout = int(os.getenv("EURI_TIMEOUT", "60"))

        # Database
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./research_assistant.db")
        self.db_pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.db_max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))

        # Vector Store (Pinecone)
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "research-papers")
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))

        # Redis (for caching and session management)
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_ttl = int(os.getenv("REDIS_TTL", "3600"))  # 1 hour

        # File Processing
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", str(50 * 1024 * 1024)))  # 50MB
        self.allowed_file_types = self._parse_list(os.getenv("ALLOWED_FILE_TYPES", "pdf,txt,docx,md"))
        self.upload_dir = os.getenv("UPLOAD_DIR", "./uploads")

        # RAG Configuration
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.top_k_retrieval = int(os.getenv("TOP_K_RETRIEVAL", "5"))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

        # Rate Limiting
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour

        # Monitoring
        self.sentry_dsn = os.getenv("SENTRY_DSN")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"

        # CORS
        self.cors_origins = self._parse_list(os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501"))

    def _parse_list(self, value: str) -> List[str]:
        """Parse comma-separated string into list."""
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]


# Global settings instance
settings = Settings()


# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "formatter": "detailed",
            "class": "logging.FileHandler",
            "filename": "app.log",
        },
    },
    "root": {
        "level": settings.log_level,
        "handlers": ["default", "file"],
    },
}
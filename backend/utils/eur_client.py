"""Enhanced Euri LLM Client with LangChain integration and os.getenv support."""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from concurrent.futures import ThreadPoolExecutor

import httpx
import structlog
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk, LLMResult, Generation

try:
    from langchain_core.pydantic_v1 import Field, validator
    from pydantic import BaseModel
except ImportError:
    from pydantic import BaseModel, Field
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup structured logging
logger = structlog.get_logger(__name__)


class EuriAPIError(Exception):
    """Custom exception for Euri API errors."""
    pass


class EuriRateLimitError(EuriAPIError):
    """Exception for rate limit errors."""
    pass


class EuriAuthenticationError(EuriAPIError):
    """Exception for authentication errors."""
    pass


class EuriLLMConfig(BaseModel):
    """Configuration for Euri LLM client using os.getenv."""
    api_key: str = Field(default_factory=lambda: os.getenv("EURI_API_KEY", ""), description="Euri API key")
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            "EURI_BASE_URL",
            "https://api.euron.one/api/v1/euri/alpha/chat/completions"
        ),
        description="Base URL for Euri API"
    )
    model: str = Field(default_factory=lambda: os.getenv("EURI_MODEL", "gpt-4.1-nano"), description="Model name")
    temperature: float = Field(default_factory=lambda: float(os.getenv("EURI_TEMPERATURE", "0.7")), description="Temperature")
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("EURI_MAX_TOKENS", "2000")), description="Max tokens")
    timeout: int = Field(default_factory=lambda: int(os.getenv("EURI_TIMEOUT", "60")), description="Timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries")

    @validator("temperature")
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class EuriLLM(LLM):
    """
    Production-grade Euri LLM implementation with LangChain integration.

    Features:
    - Async/sync support
    - Retry logic with exponential backoff
    - Rate limiting
    - Comprehensive error handling
    - Structured logging
    - Token usage tracking
    - Streaming support
    """

    config: EuriLLMConfig = Field(default_factory=lambda: EuriLLMConfig())
    client: Optional[httpx.AsyncClient] = Field(default=None, exclude=True)
    _executor: Optional[ThreadPoolExecutor] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_client()
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _setup_client(self):
        """Setup HTTP client with proper configuration."""
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"research-assistant/{os.getenv('APP_VERSION', '1.0.0')}"
            }
        )

    @property
    def _llm_type(self) -> str:
        """Return identifier of the LLM."""
        return "euri"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    def _format_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Format prompt as messages for the API."""
        return [{"role": "user", "content": prompt}]

    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Make API request with retry logic."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
            **kwargs
        }

        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                logger.info(
                    "Making Euri API request",
                    attempt=attempt + 1,
                    model=self.config.model,
                    message_count=len(messages)
                )

                response = await self.client.post(
                    self.config.base_url,
                    json=payload
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(
                        "Euri API request successful",
                        status_code=response.status_code,
                        usage=result.get("usage", {})
                    )
                    return result

                elif response.status_code == 401:
                    raise EuriAuthenticationError("Invalid API key")

                elif response.status_code == 429:
                    raise EuriRateLimitError("Rate limit exceeded")

                else:
                    response.raise_for_status()

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        "Euri API request failed, retrying",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Euri API request failed after all retries",
                        attempts=self.config.max_retries + 1,
                        error=str(e)
                    )

        raise EuriAPIError(f"Request failed after {self.config.max_retries + 1} attempts: {last_exception}")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous call to Euri LLM."""
        return asyncio.run(self._acall(prompt, stop, run_manager, **kwargs))

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronous call to Euri LLM."""
        if not self.client:
            self._setup_client()

        messages = self._format_messages(prompt)

        try:
            response = await self._make_request(messages, **kwargs)

            if "choices" not in response or not response["choices"]:
                raise EuriAPIError("Invalid response format: no choices")

            content = response["choices"][0]["message"]["content"]

            # Handle stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in content:
                        content = content.split(stop_seq)[0]

            return content

        except Exception as e:
            logger.error("Error in Euri LLM call", error=str(e), prompt_length=len(prompt))
            raise EuriAPIError(f"LLM call failed: {str(e)}")

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """Stream response from Euri LLM."""
        if not self.client:
            self._setup_client()

        messages = self._format_messages(prompt)

        try:
            async with self.client.stream(
                "POST",
                self.config.base_url,
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stream": True,
                    **kwargs
                }
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk_data = httpx._content.json.loads(data)
                            if "choices" in chunk_data and chunk_data["choices"]:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]

                                    # Handle stop sequences
                                    if stop:
                                        for stop_seq in stop:
                                            if stop_seq in content:
                                                content = content.split(stop_seq)[0]
                                                yield GenerationChunk(text=content)
                                                return

                                    yield GenerationChunk(text=content)
                        except Exception as e:
                            logger.warning("Error parsing stream chunk", error=str(e))
                            continue

        except Exception as e:
            logger.error("Error in Euri LLM stream", error=str(e))
            raise EuriAPIError(f"Stream failed: {str(e)}")

    def __del__(self):
        """Cleanup resources."""
        if self.client:
            asyncio.create_task(self.client.aclose())
        if self._executor:
            self._executor.shutdown(wait=False)


# Factory function for easy instantiation
def create_euri_llm(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> EuriLLM:
    """
    Factory function to create EuriLLM instance with custom configuration.

    Args:
        api_key: Euri API key (defaults to environment variable)
        model: Model name (defaults to environment variable)
        temperature: Temperature for generation (defaults to environment variable)
        max_tokens: Maximum tokens (defaults to environment variable)
        **kwargs: Additional configuration parameters

    Returns:
        Configured EuriLLM instance
    """
    # Use provided values or fall back to settings (which read from environment)
    config_params = {}

    if api_key is not None:
        config_params["api_key"] = api_key
    if model is not None:
        config_params["model"] = model
    if temperature is not None:
        config_params["temperature"] = temperature
    if max_tokens is not None:
        config_params["max_tokens"] = max_tokens

    # Add any additional parameters
    config_params.update(kwargs)

    # Create config - will use defaults from settings if not provided
    config = EuriLLMConfig(**config_params)
    return EuriLLM(config=config)


# Legacy function for backward compatibility
def euri_chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> str:
    """
    Legacy function for backward compatibility.

    Args:
        messages: List of message dictionaries
        model: Model name (defaults to environment variable)
        temperature: Temperature for generation (defaults to environment variable)
        max_tokens: Maximum tokens (defaults to environment variable)

    Returns:
        Generated text content
    """
    # Use environment defaults if not provided
    llm = create_euri_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Convert messages to prompt (simple concatenation for compatibility)
    prompt = "\n".join([msg.get("content", "") for msg in messages])

    return llm._call(prompt)

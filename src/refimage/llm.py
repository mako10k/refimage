"""
LLM Integration Module for RefImage.
Provides unified interface for multiple LLM providers.
"""

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    CLAUDE = "claude"
    LOCAL = "local"


class LLMError(Exception):
    """Base exception for LLM operations."""


class LLMMessage(BaseModel):
    """LLM message format."""

    role: str = Field(
        ..., description="Message role (system, user, assistant)"
    )
    content: str = Field(..., description="Message content")


class LLMResponse(BaseModel):
    """LLM response format."""

    content: str = Field(..., description="Generated content")
    provider: str = Field(..., description="LLM provider used")
    model: str = Field(..., description="Model name")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")
    processing_time_ms: int = Field(
        ..., description="Processing time in milliseconds"
    )


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with configuration."""
        self.config = config

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate response from messages."""

    @abstractmethod
    def get_model_name(self) -> str:
        """Get current model name."""


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "gpt-4")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")

        if not self.api_key:
            raise LLMError("OpenAI API key not provided")

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        payload = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                )
                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                tokens_used = data.get("usage", {}).get("total_tokens")

                processing_time = int((time.time() - start_time) * 1000)

                return LLMResponse(
                    content=content,
                    provider="openai",
                    model=self.model,
                    tokens_used=tokens_used,
                    processing_time_ms=processing_time,
                )

        except httpx.RequestError as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise LLMError(f"OpenAI API request failed: {e}")
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMError(f"OpenAI API error: {e}")

    def get_model_name(self) -> str:
        """Get current model name."""
        return self.model


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "claude-3-sonnet-20240229")
        self.base_url = config.get("base_url", "https://api.anthropic.com/v1")

        if not self.api_key:
            raise LLMError("Claude API key not provided")

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate response using Claude API."""
        start_time = time.time()

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Convert messages to Claude format
        system_message = None
        claude_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                claude_messages.append(
                    {"role": msg.role, "content": msg.content}
                )

        payload = {
            "model": self.model,
            "messages": claude_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 1000,
        }

        if system_message:
            payload["system"] = system_message

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                )
                response.raise_for_status()

                data = response.json()
                content = data["content"][0]["text"]
                tokens_used = data.get("usage", {}).get("total_tokens")

                processing_time = int((time.time() - start_time) * 1000)

                return LLMResponse(
                    content=content,
                    provider="claude",
                    model=self.model,
                    tokens_used=tokens_used,
                    processing_time_ms=processing_time,
                )

        except httpx.RequestError as e:
            logger.error(f"Claude API request failed: {e}")
            raise LLMError(f"Claude API request failed: {e}")
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise LLMError(f"Claude API error: {e}")

    def get_model_name(self) -> str:
        """Get current model name."""
        return self.model


class LocalProvider(BaseLLMProvider):
    """Local LLM provider (e.g., Ollama, vLLM)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "llama2")
        self.base_url = config.get("base_url", "http://localhost:11434")

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate response using local LLM."""
        start_time = time.time()

        # Convert messages to single prompt for local models
        prompt = self._messages_to_prompt(messages)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
        }

        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()

                data = response.json()
                content = data["response"]

                processing_time = int((time.time() - start_time) * 1000)

                return LLMResponse(
                    content=content,
                    provider="local",
                    model=self.model,
                    tokens_used=None,  # Not available for local models
                    processing_time_ms=processing_time,
                )

        except httpx.RequestError as e:
            logger.error(f"Local LLM request failed: {e}")
            raise LLMError(f"Local LLM request failed: {e}")
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            raise LLMError(f"Local LLM error: {e}")

    def _messages_to_prompt(self, messages: List[LLMMessage]) -> str:
        """Convert messages to single prompt for local models."""
        prompt_parts = []

        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"Human: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def get_model_name(self) -> str:
        """Get current model name."""
        return self.model


class LLMManager:
    """Manager for LLM providers with automatic switching."""

    def __init__(self, settings):
        """Initialize LLM manager with settings."""
        self.settings = settings
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.current_provider = settings.llm_provider
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available LLM providers."""
        # OpenAI provider
        if self.settings.openai_api_key:
            try:
                self.providers[LLMProvider.OPENAI] = OpenAIProvider(
                    {
                        "api_key": self.settings.openai_api_key,
                        "model": self.settings.openai_model,
                        "base_url": self.settings.openai_base_url,
                    }
                )
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider: {e}")

        # Claude provider
        if self.settings.claude_api_key:
            try:
                self.providers[LLMProvider.CLAUDE] = ClaudeProvider(
                    {
                        "api_key": self.settings.claude_api_key,
                        "model": self.settings.claude_model,
                        "base_url": self.settings.claude_base_url,
                    }
                )
                logger.info("Claude provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude provider: {e}")

        # Local provider
        if self.settings.local_llm_enabled:
            try:
                self.providers[LLMProvider.LOCAL] = LocalProvider(
                    {
                        "model": self.settings.local_llm_model,
                        "base_url": self.settings.local_llm_base_url,
                    }
                )
                logger.info("Local LLM provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Local LLM provider: {e}")

    async def generate(
        self,
        messages: List[LLMMessage],
        provider: Optional[LLMProvider] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate response using specified or current provider."""
        provider = provider or self.current_provider

        if provider not in self.providers:
            raise LLMError(f"Provider {provider} not available")

        return await self.providers[provider].generate(
            messages, temperature, max_tokens, **kwargs
        )

    def switch_provider(self, provider: LLMProvider):
        """Switch current provider."""
        if provider not in self.providers:
            raise LLMError(f"Provider {provider} not available")

        self.current_provider = provider
        logger.info(f"Switched to LLM provider: {provider}")

    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers."""
        return list(self.providers.keys())

    def get_current_provider(self) -> LLMProvider:
        """Get current provider."""
        return self.current_provider


# Text-to-DSL conversion prompts
TEXT_TO_DSL_SYSTEM_PROMPT = """You are an expert at converting natural language
image search queries into a specialized DSL (Domain Specific Language).

The DSL supports these operations:
- TEXT("query"): Basic text search
- AND(query1, query2): Both conditions must match
- OR(query1, query2): Either condition can match
- EXCLUDE(base_query, exclude_query): Find base_query but exclude exclude_query
- WEIGHT(query, weight): Apply weight (0.0-2.0) to query importance

Examples:
- "cats" → TEXT("cats")
- "red cars or blue cars" → OR(TEXT("red cars"), TEXT("blue cars"))
- "beaches without people" → EXCLUDE(TEXT("beaches"), TEXT("people"))
- "important: dogs, less important: puppies" → AND(WEIGHT(TEXT("dogs"), 1.5),
  WEIGHT(TEXT("puppies"), 0.8))

Rules:
1. Use simple TEXT() for basic queries
2. Use OR() for alternative options
3. Use AND() when multiple conditions are explicitly required
4. Use EXCLUDE() for negative conditions ("without", "not", "except")
5. Use WEIGHT() when importance is specified
6. Keep queries simple and focused
7. Only use operations that are clearly implied by the user's language

Convert the following query to DSL:"""

TEXT_TO_DSL_EXAMPLES = [
    {
        "input": "cats sitting on couches",
        "output": 'TEXT("cats sitting on couches")',
        "explanation": "Simple text query for specific scene",
    },
    {
        "input": "red sports cars or motorcycles",
        "output": 'OR(TEXT("red sports cars"), TEXT("motorcycles"))',
        "explanation": "OR operation for alternative vehicle types",
    },
    {
        "input": "beaches at sunset without people",
        "output": 'EXCLUDE(TEXT("beaches at sunset"), TEXT("people"))',
        "explanation": "EXCLUDE operation to remove unwanted elements",
    },
    {
        "input": "dogs and cats playing together",
        "output": 'AND(TEXT("dogs"), TEXT("cats playing together"))',
        "explanation": "AND operation for explicit requirement of both elements",
    },
    {
        "input": "very important: mountains, somewhat important: snow",
        "output": 'AND(WEIGHT(TEXT("mountains"), 1.8), WEIGHT(TEXT("snow"), 1.2))',
        "explanation": "WEIGHT operations based on importance indicators",
    },
]

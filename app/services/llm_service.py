"""Chat completion providers.

``LLM_PROVIDER=ollama`` (default) talks to a local Ollama server and needs no
API key. ``LLM_PROVIDER=openai`` enables any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import httpx

from app.core.config import settings
from app.core.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the language model cannot produce an answer."""


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


class BaseLLMService(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def generate(self, messages: list[ChatMessage], *, temperature: float = 0.1) -> str:
        """Return the assistant's reply for the given message list."""

    def health(self) -> tuple[bool, str]:
        return True, "ready"


class OllamaLLMService(BaseLLMService):
    provider_name = "ollama"

    def __init__(self, base_url: str, model_name: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def generate(self, messages: list[ChatMessage], *, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 4096},
        }
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            if exc.response is not None and exc.response.status_code == 404:
                raise ServiceUnavailableError(
                    f"Ollama model '{self.model_name}' is not installed. Run "
                    f"`ollama pull {self.model_name}`."
                ) from exc
            raise LLMError(f"Ollama returned an error: {detail}") from exc
        except httpx.HTTPError as exc:
            raise ServiceUnavailableError(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?"
            ) from exc

        content = (data.get("message") or {}).get("content", "")
        content = content.strip()
        if not content:
            raise LLMError("Ollama returned an empty response")
        return content

    def health(self) -> tuple[bool, str]:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            response.raise_for_status()
            models = {model.get("name", "") for model in response.json().get("models", [])}
        except httpx.HTTPError as exc:
            return False, f"Ollama unreachable at {self.base_url}: {exc.__class__.__name__}"
        if self.model_name in models or any(m.startswith(f"{self.model_name}:") for m in models):
            return True, f"model '{self.model_name}' available"
        return False, f"model '{self.model_name}' not pulled (run `ollama pull {self.model_name}`)"


class OpenAILLMService(BaseLLMService):
    provider_name = "openai"

    def __init__(self, api_key: str, model_name: str, base_url: str, timeout_seconds: float) -> None:
        if not api_key:
            raise ServiceUnavailableError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(self, messages: list[ChatMessage], *, temperature: float = 0.1) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model_name,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "temperature": temperature,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPError as exc:
            raise ServiceUnavailableError(f"LLM request failed: {exc}") from exc
        except (KeyError, IndexError, AttributeError) as exc:
            raise LLMError("The language model returned a malformed response") from exc
        if not content:
            raise LLMError("The language model returned an empty response")
        return content

    def health(self) -> tuple[bool, str]:
        return True, f"OpenAI-compatible model '{self.model_name}' configured"


def build_llm_service() -> BaseLLMService:
    if settings.llm_provider == "ollama":
        return OllamaLLMService(
            base_url=settings.ollama_base_url,
            model_name=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    if settings.llm_provider == "openai":
        return OpenAILLMService(
            api_key=settings.openai_api_key,
            model_name=settings.openai_llm_model,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    raise ServiceUnavailableError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'")


@lru_cache(maxsize=1)
def get_llm_service() -> BaseLLMService:
    return build_llm_service()

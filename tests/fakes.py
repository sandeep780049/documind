"""Test doubles for the external AI providers.

These are used *only* in unit/integration tests so the suite is fast and
deterministic. Application code never falls back to them.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from app.core.config import settings
from app.services.embedding_service import BaseEmbeddingService
from app.services.llm_service import BaseLLMService, ChatMessage

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FakeEmbeddingService(BaseEmbeddingService):
    """Hashing bag-of-words embedding: relevant text really scores higher."""

    provider_name = "fake"
    model_name = "fake-bow-embedding"

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension or settings.embedding_dimension
        self.embed_calls: list[list[str]] = []

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def health(self) -> tuple[bool, str]:
        return True, "fake embedding provider"


class FakeLLMService(BaseLLMService):
    provider_name = "fake"
    model_name = "fake-chat-model"

    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []

    def generate(self, messages: list[ChatMessage], *, temperature: float = 0.1) -> str:
        self.calls.append(messages)
        return "According to the uploaded document, the answer is stated in source [1]."

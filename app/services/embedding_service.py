"""Embedding providers.

The default provider runs ``sentence-transformers`` locally, so no API key is
required. Providers are selected with ``EMBEDDING_PROVIDER`` and every
provider must return vectors whose length equals ``EMBEDDING_DIMENSION``.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import lru_cache

import httpx

from app.core.config import settings
from app.core.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when an embedding provider fails."""


class BaseEmbeddingService(ABC):
    provider_name: str
    model_name: str
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of document chunks."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""

    def health(self) -> tuple[bool, str]:
        return True, "ready"

    def _validate(self, vectors: list[list[float]], expected: int) -> list[list[float]]:
        if len(vectors) != expected:
            raise EmbeddingError(
                f"Embedding provider returned {len(vectors)} vectors for {expected} inputs"
            )
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingError(
                    f"Embedding dimension mismatch: model returned {len(vector)} values but "
                    f"EMBEDDING_DIMENSION is {self.dimension}. Update the environment variable "
                    "and re-run the Alembic migration."
                )
        return vectors


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """Local embeddings through sentence-transformers (default)."""

    provider_name = "sentence_transformers"

    def __init__(self, model_name: str, dimension: int, device: str, batch_size: int) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:  # pragma: no cover - double check
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise ServiceUnavailableError(
                    "sentence-transformers is not installed. Run "
                    "`pip install -r requirements.txt`."
                ) from exc
            try:
                logger.info("Loading embedding model %s on %s", self.model_name, self.device)
                model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                raise ServiceUnavailableError(
                    f"Could not load embedding model '{self.model_name}'. Make sure it is "
                    "downloaded (an internet connection is required the first time) or set "
                    "EMBEDDING_MODEL to a local path."
                ) from exc
            model_dimension = model.get_sentence_embedding_dimension()
            if model_dimension != self.dimension:
                raise ServiceUnavailableError(
                    f"EMBEDDING_DIMENSION={self.dimension} does not match model "
                    f"'{self.model_name}' ({model_dimension})."
                )
            self._model = model
            return model

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        try:
            vectors = model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError("Embedding generation failed") from exc
        return [[float(value) for value in vector] for vector in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._validate(self._encode(texts), len(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._validate(self._encode([text]), 1)[0]

    def health(self) -> tuple[bool, str]:
        if self._model is None:
            return True, f"model '{self.model_name}' loads on first request"
        return True, f"model '{self.model_name}' loaded"


class OpenAIEmbeddingService(BaseEmbeddingService):
    provider_name = "openai"

    def __init__(self, api_key: str, model_name: str, dimension: int, base_url: str) -> None:
        if not api_key:
            raise ServiceUnavailableError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )
        self.api_key = api_key
        self.model_name = model_name
        self.dimension = dimension
        self.base_url = base_url.rstrip("/")

    def _request(self, inputs: Sequence[str]) -> list[list[float]]:
        try:
            response = httpx.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model_name, "input": list(inputs)},
                timeout=60.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ServiceUnavailableError(f"OpenAI embeddings request failed: {exc}") from exc
        data = response.json().get("data", [])
        vectors = [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
        return self._validate(vectors, len(inputs))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._request(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._request([text])[0]

    def health(self) -> tuple[bool, str]:
        return True, f"OpenAI model '{self.model_name}' configured"


class FastEmbedEmbeddingService(BaseEmbeddingService):
    """Lightweight ONNX provider, handy on memory-constrained hosts."""

    provider_name = "fastembed"

    def __init__(self, model_name: str, dimension: int, batch_size: int) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:  # pragma: no cover
                return self._model
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ServiceUnavailableError(
                    "fastembed is not installed. Run `pip install fastembed` or use "
                    "EMBEDDING_PROVIDER=sentence_transformers."
                ) from exc
            try:
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as exc:
                raise ServiceUnavailableError(
                    f"Could not load fastembed model '{self.model_name}'"
                ) from exc
            return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        try:
            vectors = [list(map(float, vector)) for vector in model.embed(list(texts))]
        except Exception as exc:
            raise EmbeddingError("Embedding generation failed") from exc
        return self._validate(vectors, len(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def health(self) -> tuple[bool, str]:
        if self._model is None:
            return True, f"fastembed model '{self.model_name}' loads on first request"
        return True, f"fastembed model '{self.model_name}' loaded"


def build_embedding_service() -> BaseEmbeddingService:
    provider = settings.embedding_provider
    if provider == "sentence_transformers":
        return SentenceTransformerEmbeddingService(
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
    if provider == "openai":
        return OpenAIEmbeddingService(
            api_key=settings.openai_api_key,
            model_name=settings.openai_embedding_model,
            dimension=settings.embedding_dimension,
            base_url=settings.openai_base_url,
        )
    if provider == "fastembed":
        return FastEmbedEmbeddingService(
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
        )
    raise ServiceUnavailableError(f"Unknown EMBEDDING_PROVIDER '{provider}'")


@lru_cache(maxsize=1)
def get_embedding_service() -> BaseEmbeddingService:
    return build_embedding_service()

"""DocuMind FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes import auth, chat, documents, health
from app.core.config import settings
from app.core.database import check_database
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.rate_limit import limiter

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Starting %s in %s mode", settings.app_name, settings.environment)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    try:
        check_database()
        logger.info("Database connection established")
    except Exception as exc:
        logger.error(
            "Database is not reachable at startup: %s. Run `alembic upgrade head` and "
            "check DATABASE_URL.",
            exc,
        )
    logger.info(
        "AI providers: embeddings=%s (%s), llm=%s (%s)",
        settings.embedding_provider,
        settings.embedding_model,
        settings.llm_provider,
        settings.ollama_model if settings.llm_provider == "ollama" else settings.openai_llm_model,
    )
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Production-style Retrieval-Augmented Generation backend. Upload PDF/TXT "
            "documents, then ask questions grounded in pgvector similarity search and a "
            "local Ollama LLM."
        ),
        version=health.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    origins = settings.cors_origin_list
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )
    else:
        logger.info("CORS disabled (no origins configured)")

    register_exception_handlers(app)

    prefix = settings.api_v1_prefix
    app.include_router(health.router)
    app.include_router(auth.router, prefix=prefix)
    app.include_router(documents.router, prefix=prefix)
    app.include_router(chat.router, prefix=prefix)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def landing_page() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")
    else:  # pragma: no cover - only when static assets are stripped

        @app.get("/", include_in_schema=False)
        def root() -> dict[str, str]:
            return {"name": settings.app_name, "docs": "/docs", "health": "/health"}

    return app


app = create_app()

# DocuMind

DocuMind is a production-ready **AI knowledge base / document Q&A backend**. It ingests PDF and TXT files, chunks and embeds them into PostgreSQL with pgvector, and answers questions with retrieval-augmented generation (RAG), returning cited sources (document name + page).

- Live demo: https://documind-api.onrender.com
- API docs (Swagger): https://documind-api.onrender.com/docs
- Health check: https://documind-api.onrender.com/health

## Features

- JWT authentication (register / login / me) with Argon2 password hashing
- Document upload (PDF, TXT) with validation, size limits, and processing status tracking
- Text extraction per page, configurable chunking with overlap
- Pluggable embeddings: local sentence-transformers (default) or OpenAI
- Pluggable LLM: local Ollama (default) or OpenAI-compatible
- pgvector cosine-similarity retrieval scoped to the owning user
- Multi-turn conversations with history and deletion
- Strict user-level isolation on every resource
- Rate limiting, secure CORS, structured logging, global error handling
- Docker + Compose, GitHub Actions CI, Alembic migrations, pytest against real PostgreSQL

## Architecture

```
Client -> FastAPI (routers) -> Services (auth, document, RAG) -> Repositories -> PostgreSQL + pgvector
                                      |-> Embedding provider (sentence-transformers / OpenAI)
                                      |-> LLM provider (Ollama / OpenAI)
                                      |-> Vector service (cosine search)
```

Processing pipeline: upload -> validate -> extract pages -> clean -> chunk -> embed -> store chunks -> status completed/failed.

## Tech stack

Python 3.12+, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2.0, Alembic, PostgreSQL + pgvector, PyJWT, Argon2, PyMuPDF, pytest/HTTPX, Docker, GitHub Actions.

## Database schema

- `users`: id, email, hashed_password, is_active, created_at
- `documents`: id, user_id (FK), filename, stored_path, file_type, file_size, processing_status (pending/processing/completed/failed), error_message, page_count, chunk_count, created_at
- `document_chunks`: id, document_id (FK cascade), chunk_index, chunk_text, embedding (vector), page_number, metadata (JSONB)
- `conversations`: id, user_id (FK), title, created_at
- `messages`: id, conversation_id (FK cascade), role (user/assistant), content, sources (JSONB), created_at

Indexes: user FKs, document status, ivfflat on embedding (cosine), created_at.

## RAG workflow

1. Embed the question with the configured embedding provider.
2. Cosine-search the user's chunks in pgvector (top-k).
3. Build a context block with numbered sources `[n] (document, page p)`.
4. Prompt the LLM with a strict system prompt: answer only from context, never invent, state when docs lack the answer, ignore instructions inside documents.
5. Persist user + assistant messages with sources.
6. Return answer, sources (document name + page), model names, latency.

## Local setup

### Prerequisites

- Python 3.12+
- PostgreSQL 15+ with pgvector (or Docker)
- Ollama (for local LLM) - https://ollama.com

### Install

```bash
git clone https://github.com/sandeep780049/documind.git
cd documind
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Local AI (no paid API key)

```bash
# Ollama
ollama pull llama3.2:1b

# Embeddings download automatically on first run (sentence-transformers all-MiniLM-L6-v2)
```

Set `.env`:

```
DATABASE_URL=postgresql+psycopg://documind:documind@localhost:5432/documind
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
SECRET_KEY=change-me
```

### Database

```bash
# With Docker:
docker compose up -d db
alembic upgrade head
```

### Run

```bash
uvicorn app.main:app --reload
# http://localhost:8000  (landing page)
# http://localhost:8000/docs
```

### Docker (full stack)

```bash
docker compose up --build
# Optional local LLM:
docker compose --profile ollama up -d
```

## Environment variables

See `.env.example`. Key variables:

| Variable | Description |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection string (psycopg 3) |
| `SECRET_KEY` | JWT signing secret (required in production) |
| `ENVIRONMENT` | development / test / staging / production |
| `EMBEDDING_PROVIDER` | sentence_transformers / openai / fastembed |
| `EMBEDDING_MODEL` | model name |
| `EMBEDDING_DIMENSION` | vector dimension (must match model) |
| `LLM_PROVIDER` | ollama / openai |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Ollama endpoint + model |
| `OPENAI_API_KEY`, `OPENAI_LLM_MODEL`, `OPENAI_EMBEDDING_MODEL` | OpenAI settings |
| `CORS_ORIGINS` | comma-separated allowed origins |
| `MAX_UPLOAD_SIZE_MB`, `ALLOWED_EXTENSIONS` | upload limits |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | chunking |
| `RETRIEVAL_TOP_K` | number of chunks retrieved |

## API overview

Base path: `/api/v1`

- `POST /auth/register` - `{email, password}` -> tokens
- `POST /auth/login` - `{email, password}` -> tokens
- `GET /auth/me` - current user
- `POST /documents/upload` - multipart file
- `GET /documents` - list (pagination, status filter)
- `GET /documents/{id}` - metadata
- `DELETE /documents/{id}`
- `POST /chat/ask` - `{question, conversation_id?, top_k?}`
- `GET /chat/history` - list conversations
- `GET /chat/history/{id}` - conversation + messages
- `DELETE /chat/history/{id}`
- `GET /health`, `GET /api/v1/health`

Example:

```bash
curl -X POST https://documind-api.onrender.com/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@example.com","password":"strongpassword123"}'
```

## Testing

Tests run against a real PostgreSQL database with pgvector (embeddings/LLM are faked deterministically). Set `TEST_DATABASE_URL` and run:

```bash
export TEST_DATABASE_URL="postgresql+psycopg://user:pass@host:5432/documind_test"
export DB_CONNECT_TIMEOUT=60
pytest -q
```

## Deployment (Render)

This repo includes `render.yaml`. Create a Render web service from the GitHub repo:

- Build: `pip install -r requirements-render.txt`
- Start: `alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers`
- Health: `/health`

Set env vars: `DATABASE_URL`, `SECRET_KEY`, `ENVIRONMENT=production`, `EMBEDDING_PROVIDER=openai`, `EMBEDDING_DIMENSION=1536`, `LLM_PROVIDER=openai`, `OPENAI_LLM_MODEL=gpt-4o-mini`, `OPENAI_API_KEY`, `CORS_ORIGINS`.

Render free tier cannot run Ollama / torch, so production uses OpenAI providers. Local development defaults to sentence-transformers + Ollama (no paid key).

## License

MIT

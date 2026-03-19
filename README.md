# AgentHub

AgentHub is a self-hosted application for tenant-aware document ingestion, retrieval-augmented querying, prompt-driven workflow execution, and tool-using chat. The repository is implemented as a modular monolith: one FastAPI service owns HTTP delivery, request validation, workflow orchestration, LLM transport, retrieval, and agent execution, while a React/Vite frontend provides an operator-facing control surface over the exposed API.

---

## About

At runtime, the platform is divided into a small number of concrete subsystems:

- **HTTP control plane** - `api/app/http/` assembles the FastAPI application, installs middleware, exposes versioned routers under `/api/v1`, and formats streaming responses as Server-Sent Events.
- **Persistence layer** - SQLAlchemy models in `api/app/models.py` store source documents, retrieval chunks, and AI audit records in SQLite or PostgreSQL. Redis is optional and is used only for per-tenant rate limiting and short-lived document caching.
- **LLM transport layer** - `api/app/llm/` provides Ollama and OpenAI-compatible adapters with timeout handling, retry policy, Prometheus instrumentation, and circuit-breaker protection. The agent runtime uses separate LangChain chat-model bindings for tool calling.
- **Retrieval layer** - `api/app/rag/` chunks document text, computes embeddings, stores vectors as JSON, and performs in-process cosine-similarity ranking for top-k retrieval.
- **Workflow layer** - `api/app/flows/` implements prompt-specialized execution paths for classification, grounded question answering, and notarial summarization, with audit persistence in `ai_call_audit`.
- **Agent runtime** - `api/app/agents/` runs a LangGraph ReAct loop with calculator, web-search, and document-lookup tools. Responses can be returned as JSON or streamed token-by-token over SSE.
- **Operator UI** - `frontend/` is a React 19 single-page control surface for exercising health checks, document ingestion, retrieval, workflow endpoints, and agent chat against a selected tenant and optional API key.

The deployment model is intentionally simple: document indexing, retrieval, workflow execution, and agent orchestration all run in-process during the request lifecycle. There is no external task queue, vector database, or distributed agent scheduler yet, which keeps the stack easy to operate but means throughput and latency are bounded by the API process and the configured model and search backends.

**Core stack:** FastAPI, SQLAlchemy asyncio, Pydantic, LangGraph, LangChain, React 19, Vite 7, SQLite/PostgreSQL, optional Redis, and Ollama or OpenAI-compatible model endpoints.

---

## Table of Contents

- [About](#about)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Development](#development)
- [Project Structure](#project-structure)

---

## Quick Start

### Local development (backend + frontend)

```bash
# Backend
cd api
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload

# Frontend (separate terminal)
cd frontend
corepack enable
pnpm install
pnpm run dev
```

- API: http://localhost:8000
- Frontend: http://localhost:5173 (proxies `/api` and `/metrics` to the API)

### Docker Compose (full stack with Postgres, Redis, Ollama)

```bash
docker compose up --build
```

- Frontend: http://localhost:5173
- API: http://localhost:8000
- Ollama: http://localhost:11434

Pull an LLM model before using the agent:

```bash
docker compose run ollama ollama pull llama3.2:1b
```

---

## Configuration

All configuration is via environment variables. Copy `api/.env.example` to `api/.env` and adjust.

### Required (for AI features)

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | Yes | `ollama` or `openai_compatible` |
| `LLM_BASE_URL` | Yes | LLM endpoint (e.g. `http://localhost:11434` for Ollama) |
| `LLM_MODEL` | Yes | Model name (e.g. `llama3.2`, `llama3.2:1b`) |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///./app.db` | SQLite or Postgres. For Postgres use `postgresql+asyncpg://user:pass@host:5432/dbname`. The app auto-converts `postgres://` to `postgresql+asyncpg://` for compatibility with providers like Render. |

### LLM (OpenAI-compatible)

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | API key for OpenAI, vLLM, LocalAI, etc. (when `LLM_PROVIDER=openai_compatible`) |
| `LLM_TIMEOUT_SECONDS` | Request timeout (default: 60) |
| `LLM_MAX_RETRIES` | Retry count on transient errors (default: 2) |

### Production

| Variable | Required in prod | Description |
|----------|------------------|-------------|
| `API_KEY` | Yes | Required when `ENVIRONMENT=prod`. Clients must send `X-API-Key` header. |
| `ENVIRONMENT` | No | `local`, `dev`, or `prod` |
| `CORS_ALLOWED_ORIGINS` | No | Comma-separated origins for CORS. Default `*` allows all. In prod set to frontend URLs. |

### Redis

| Variable | Description |
|----------|-------------|
| `REDIS_URL` | Redis URL (e.g. `redis://localhost:6379/0`). Disable rate limiting and document cache by omitting. |
| `RATE_LIMIT_PER_MINUTE` | Per-tenant rate limit (default: 120) |

### Agent search

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_PROVIDER` | `duckduckgo` | `duckduckgo` (free, no key) or `tavily` |
| `TAVILY_API_KEY` | — | Required for Tavily. Get key at https://app.tavily.com/sign-in |

### RAG

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `mock` | Currently only `mock` (deterministic hash-based). Future: OpenAI, sentence-transformers. |
| `EMBEDDING_DIMENSION` | 384 | Vector dimension for embeddings |

### Audit

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_AUDIT_RETENTION_DAYS` | 0 | Purge `ai_call_audit` records older than N days. 0 = disabled. Run `python -m app.audit` (e.g. via cron) to perform purge. |

### Other

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `ENABLE_PROMETHEUS` | `true` | Expose `/metrics` |
| `DEFAULT_TENANT_ID` | `default` | Default tenant when `X-Tenant-ID` not provided |
| `TENANT_HEADER_NAME` | `X-Tenant-ID` | Header name for tenant context |
| `APP_NAME` | `AgentHub` | Display name in API and docs |

---

## API Reference

Base path: `/api/v1`. All endpoints except `/health` accept optional `X-Tenant-ID` (default: `default`). When `API_KEY` is set, `X-API-Key` is required.

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check. Returns `environment`, `timestamp`, `db_ok`, `redis_ok`, `llm_ok`. |

### Documents

| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/documents` | `id`, `title`, `text` | Create document. `id` max 64 chars, `title` max 255. |
| POST | `/documents/upload` | multipart: `file`, optional `document_id`, `title` | Upload file (UTF-8 text, max 5 MB). |
| GET | `/documents/{id}` | — | Get document by ID. |

### AI Flows

| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/ai/notary/summarize` | `text`, optional `document_id`, `language` (`nl`/`en`) | Notary-style document summary. |
| POST | `/ai/classify` | `text`, optional `candidate_labels` (max 10) | Zero-shot classification. |
| POST | `/ai/ask` | `question`, `context` | Q&A over context. |
| POST | `/ai/ask/stream` | `question`, `context` | Q&A streaming (SSE). |

### RAG

| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/ai/rag/query` | `query`, optional `document_ids`, `top_k` (1–20, default 5) | RAG query over indexed documents. |
| POST | `/ai/rag/query/stream` | `query` | RAG query streaming (SSE). |
| POST | `/ai/rag/index` | `document_id` | Index document for RAG (chunk, embed, store). |

### Agent

| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/ai/agents/chat` | `message` (1–4000 chars) | ReAct agent chat. Tools: calculator, search, document lookup. |
| POST | `/ai/agents/chat/stream` | `message` | Agent chat streaming (SSE). |

### Metrics

| Method | Path | Description |
|--------|------|-------------|
| GET | `/metrics` | Prometheus metrics (requires `X-API-Key` when `API_KEY` set). |

### Streaming format (SSE)

All streaming endpoints emit Server-Sent Events:

```
data: {"token": "..."}
data: {"token": "..."}
data: {"done": true}
```

On error: `data: {"error": "...", "done": true}`

### Postman

Import `postman/AI-Platform.postman_collection.json` for API requests.

---

## Deployment

### Docker Compose

```bash
docker compose up --build
```

- **Dev**: API with `--reload`, frontend Vite dev server, Postgres, Redis, Ollama.
- **Prod-style**: `docker compose --profile prod up` adds nginx frontend on port 80.

### Kubernetes

```bash
kubectl apply -k k8s/
```

Requires:

- `ai-platform-config` ConfigMap (see `k8s/configmap.yaml`)
- `ai-platform-secrets` Secret (optional)

Images: `ghcr.io/hamoudy41/ai-platform-api:latest`, `ghcr.io/hamoudy41/ai-platform-ui:latest`. Build and push via CI (see `.github/workflows/cd.yml`).

### Docker images

```bash
# API
cd api && docker build -t ai-platform-api .

# Frontend
cd frontend && docker build -t ai-platform-ui .
```

---

## Development

### Backend

```bash
cd api
pip install -e ".[dev]"
pre-commit install   # run once
pytest tests/ -v
alembic upgrade head
alembic revision -m "description" --autogenerate
ruff check app tests --fix
ruff format app tests
```

Tests run from `api/` so `app` is importable. Coverage: `pytest tests/ --cov=app --cov-report=term-missing --cov-fail-under=80`.

### Frontend

```bash
cd frontend
corepack enable
pnpm install
pnpm run dev
pnpm test
pnpm run build
```

Frontend tooling targets Node `20.19+` or `22.12+`. The current Vite/Vitest stack is not reliable on Node 18.

`VITE_API_BASE` overrides API base URL (default: `/api/v1`). `VITE_PROXY_TARGET` overrides proxy target when using Vite dev server.

---

## Project Structure

```
.
├── api/                    # FastAPI backend
│   ├── app/
│   │   ├── agents/         # LangGraph ReAct agent
│   │   ├── documents/      # Document ingest/read domain services
│   │   ├── flows/          # Per-workflow orchestration modules
│   │   ├── http/           # App factory, middleware, routers, SSE helpers
│   │   ├── llm/            # Provider adapters and shared LLM types
│   │   ├── rag/            # Chunking, embeddings, retrieval
│   │   ├── api.py          # Compatibility entrypoint for create_app/app
│   │   └── ...
│   ├── alembic/            # Migrations
│   └── tests/
├── frontend/               # React + Vite
├── k8s/                    # Kubernetes manifests
├── postman/                # API collection
├── ARCHITECTURE.md         # Technical architecture
└── COMPLIANCE.md           # Data, security, audit
```

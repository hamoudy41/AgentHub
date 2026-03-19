# AgentHub

**AgentHub** (also configurable as "AI Platform") is a multi-tenant, agentic document intelligence platform. It combines a ReAct agent with RAG, document management, and AI workflows into a single deployable stack.

---

## What is this?

AgentHub is a self-hostable backend and frontend for:

- **Document management** — Create, upload, and store documents with tenant isolation.
- **RAG (Retrieval-Augmented Generation)** — Index documents, embed chunks, and query them with semantic retrieval. Answers are grounded in your documents.
- **ReAct agent** — A tool-using agent that can compute (calculator), search the web (DuckDuckGo or Tavily), and look up documents by ID. Handles natural-language math (e.g. "average of 1, 2, 5, 6") and falls back to web search when the LLM cannot answer.
- **AI flows** — Notary-style summarization (structured output for legal/notarial docs), zero-shot text classification, and Q&A over provided context.
- **Streaming** — All LLM responses stream via Server-Sent Events for low-latency UX.

It may serve as:

- A **document Q&A system** — Upload docs, index them, ask questions.
- An **agent playground** — Chat with a ReAct agent that uses tools.
- A **notary/legal summarization tool** — Summarize contracts and documents in a structured format.
- A **classification API** — Zero-shot label assignment for text.
- A **foundation for custom agents** — Extend tools, add flows, or integrate with your own LLM pipeline.

**Tech stack:** FastAPI, React, LangGraph, LangChain, SQLite/PostgreSQL, Redis (optional), Ollama or OpenAI-compatible LLMs.

---

## Table of Contents

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
npm install
npm run dev
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
npm install
npm run dev
npm test
npm run build
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
├── QUALITY_ASSESSMENT.md   # Code quality, maintainability, performance review
└── COMPLIANCE.md           # Data, security, audit
```

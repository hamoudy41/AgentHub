# AI Platform

Multi-tenant FastAPI app. `X-Tenant-ID` for tenant context; Ollama or OpenAI-compatible LLM backends; SQLite or Postgres.

## Setup

```bash
# Backend
cd api && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

Configure `LLM_PROVIDER` and `LLM_BASE_URL` in `.env`. Optional `API_KEY` for `X-API-Key` auth.

## Deploy

```bash
# Development: hot reload for API + frontend
docker compose up --build
# Frontend: http://localhost:5173  |  API: http://localhost:8000

# Production: Kubernetes
kubectl apply -k k8s/
```

Compose is for development: api (uvicorn --reload), frontend (Vite dev), Postgres, Redis, Ollama. Code changes reload automatically. Optional: `docker compose --profile prod up` adds production nginx frontend on :80. Pull model: `docker compose run ollama ollama pull llama3.1:8b`.

## API

| Method | Path | Body |
|--------|------|------|
| GET | `/api/v1/health` | — |
| POST | `/api/v1/documents` | `id`, `title`, `text` |
| POST | `/api/v1/documents/upload` | `file` (multipart), optional `document_id`, `title` |
| GET | `/api/v1/documents/{id}` | — |
| POST | `/api/v1/ai/notary/summarize` | `text`, optional `document_id`, `language` |
| POST | `/api/v1/ai/classify` | `text`, optional `candidate_labels` |
| POST | `/api/v1/ai/ask` | `question`, `context` |
| POST | `/api/v1/ai/ask/stream` | `question`, `context` (SSE streaming) |
| POST | `/api/v1/ai/rag/query` | `query`, optional `document_ids`, `top_k` |
| POST | `/api/v1/ai/rag/query/stream` | `query` (SSE streaming) |
| POST | `/api/v1/ai/rag/index` | `document_id` |

Headers: `X-Tenant-ID` (default: `default`), optional `X-API-Key`. Metrics: `/metrics`. Postman: `postman/AI-Platform.postman_collection.json`.

## Dev

```bash
cd api
pip install -e ".[dev]"
pre-commit install                  # lint on commit (run once)
pytest tests/ -v                    # run from api/ so Python finds the app module
alembic upgrade head
alembic revision -m "msg" --autogenerate
ruff check app tests --fix
ruff format app tests
```

`api/` backend, `frontend/` React app, `k8s/` manifests. See `ARCHITECTURE.md`, `COMPLIANCE.md`, `frontend/README.md`.

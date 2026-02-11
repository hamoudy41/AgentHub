# Architecture

Monorepo: `api/` (FastAPI), `frontend/` (React). Multi-tenant with `X-Tenant-ID`; LLM via Ollama or OpenAI-compatible APIs.

## Stack

| Layer      | Tech                                   |
|------------|----------------------------------------|
| API        | FastAPI                                |
| DB         | SQLite / PostgreSQL                    |
| ORM        | SQLAlchemy 2 (async)                   |
| Migrations | Alembic                                |
| LLM        | Ollama, OpenAI-compatible              |
| Auth       | Optional `X-API-Key`                    |
| Metrics    | Prometheus `/metrics`                  |

## Tenancy

`X-Tenant-ID` header (default: `default`). `tenant_id` on all tenant tables; `TenantScopedMixin` for models.

## AI Flows

Notary summarize, Classify, Ask. Each: LLM when configured, fallback otherwise; audits to `ai_call_audit`; retries + timeouts.

## Redis

`REDIS_URL`: rate limiting (120/min per tenant), document cache (5 min TTL).

## Frontend

React + TypeScript, Tailwind. See `frontend/README.md`.

## Deployment

- **ai-platform-api**: FastAPI only. `api/Dockerfile`.
- **ai-platform-ui**: nginx + SPA, proxies `/api` to API. `frontend/Dockerfile`.
- **Compose (dev)**: api (uvicorn --reload), frontend (Vite dev), Postgres, Redis, Ollama. Hot reload on code changes. Optional: `--profile prod` adds nginx frontend on :80.
- **Production**: `kubectl apply -k k8s/` (api + ui deployments)

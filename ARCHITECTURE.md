# Architecture

Technical architecture of AgentHub (AI Platform): components, data flow, design decisions, and implementation details.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Component Diagram](#2-component-diagram)
3. [Backend](#3-backend)
4. [Agent System](#4-agent-system)
5. [RAG Pipeline](#5-rag-pipeline)
6. [AI Flows](#6-ai-flows)
7. [Frontend](#7-frontend)
8. [Data Model](#8-data-model)
9. [Infrastructure](#9-infrastructure)
10. [Security & Middleware](#10-security--middleware)
11. [Roadmap](#11-roadmap)

---

## 1. Overview

| Layer | Technology |
|-------|------------|
| **API** | FastAPI, async SQLAlchemy, Pydantic |
| **Database** | SQLite (dev) or PostgreSQL (prod) |
| **Cache / Rate limit** | Redis (optional) |
| **LLM** | Ollama or OpenAI-compatible (httpx) |
| **Agent** | LangGraph StateGraph, LangChain tools |
| **RAG** | Custom chunking, mock/API embeddings, cosine similarity retrieval |
| **Frontend** | React 19, TypeScript, Vite 7, Tailwind CSS v4 |
| **Streaming** | Server-Sent Events (SSE) |

Design: modular monolith. Single API process; agents, RAG, and AI flows run in-process. Tenant isolation via `X-Tenant-ID` on all data access.

---

## 2. Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Frontend (React)                                                            │
│  Tabs: Health | Documents | RAG | Agents | Classify | Notary | Ask           │
│  Vite dev server proxies /api and /metrics to backend                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │ HTTP/SSE
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FastAPI Application (api/app/http/app.py)                                   │
│  Middleware: Request ID, CORS, Security headers, Prometheus, Rate limit,    │
│              API key (prod), Exception handlers                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────────┐       ┌─────────────────────┐       ┌───────────────────┐
│  Agent (LangGraph) │       │  RAG Pipeline       │       │  AI Flows          │
│  react_agent.py   │       │  rag/pipeline.py    │       │  services_ai_     │
│  Tools: calc,     │       │  chunking.py        │       │  flows/           │
│  search, doc      │       │  embeddings.py      │       │  Notary, Classify, │
│  lookup           │       │  Cosine similarity  │       │  Ask              │
└───────────────────┘       └─────────────────────┘       └───────────────────┘
        │                               │                               │
        └───────────────────────────────┼───────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LLM Client (services_llm.py facade over app/llm/)                           │
│  Ollama adapter and OpenAI-compatible adapter with shared transport policy    │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────┐             ┌─────────────────┐             ┌─────────────────┐
│  PostgreSQL   │             │  Redis           │             │  External       │
│  documents    │             │  Rate limit     │             │  DuckDuckGo /   │
│  document_    │             │  Document cache  │             │  Tavily (search) │
│  chunks       │             │  (300s TTL)     │             │                 │
│  ai_call_audit│             │                 │             │                 │
└───────────────┘             └─────────────────┘             └─────────────────┘
```

---

## 3. Backend

### 3.1 Entry Point

`app/main.py` exports `app = create_app()`. `create_app()` in `api.py` builds the FastAPI instance, configures middleware, and mounts routes under `api_v1_prefix` (default `/api/v1`).

### 3.2 Database

- **Engine**: Async SQLAlchemy with `asyncpg` (Postgres) or `aiosqlite` (SQLite).
- **Session**: `get_db_session` dependency yields `AsyncSession` per request; auto-rollback on exception.
- **Migrations**: Alembic. Run `alembic upgrade head` before start.

### 3.3 Configuration

`app/core/config.py` uses Pydantic Settings. All settings from env; `DATABASE_URL` is normalized (e.g. `postgres://` → `postgresql+asyncpg://`). In `ENVIRONMENT=prod`, `API_KEY` is required.

### 3.4 Directory Layout

```
api/app/
├── agents/           # LangGraph ReAct agent
│   ├── react_agent.py
│   └── tools/
│       ├── calculator.py
│       ├── search.py
│       └── document_lookup.py
├── documents/        # Document domain services and upload normalization
│   └── service.py
├── flows/            # Per-workflow orchestration and audit persistence
│   ├── ask.py
│   ├── classify.py
│   ├── common.py
│   └── notary.py
├── http/             # HTTP assembly and delivery layer
│   ├── app.py
│   ├── middleware.py
│   ├── dependencies.py
│   ├── sse.py
│   └── routers/
│       ├── documents.py
│       ├── workflows.py
│       ├── rag.py
│       ├── agents.py
│       └── health.py
├── llm/              # Provider adapters and shared LLM result/error types
│   ├── errors.py
│   ├── providers.py
│   └── types.py
├── rag/
│   ├── chunking.py
│   ├── embeddings.py
│   └── pipeline.py
├── core/
│   ├── config.py
│   ├── logging.py
│   ├── metrics.py
│   └── redis.py
├── api.py            # Compatibility entrypoint for app/create_app
├── db.py
├── models.py
├── schemas.py
├── services_ai_flows.py  # Compatibility facade over flows/
├── services_llm.py       # Compatibility facade over llm/
├── services_rag.py
└── audit.py
```

---

## 4. Agent System

### 4.1 ReAct Agent

`app/agents/react_agent.py` implements a LangGraph `StateGraph` with:

- **State**: `AgentState` with `messages: Annotated[Sequence[BaseMessage], add_messages]`
- **Nodes**: `agent` (LLM with tools), `tools` (ToolNode)
- **Edges**: `agent` → conditional → `tools` or `END`; `tools` → `agent`

The LLM is bound with tools via `model.bind_tools(tools)`. On each step, the model may emit tool calls; `ToolNode` executes them and appends `ToolMessage`s; control returns to the agent until no more tool calls.

### 4.2 Tools

| Tool | Module | Description |
|------|--------|-------------|
| `calculator_tool` | `tools/calculator.py` | Safe `eval` of math expressions. Supports `average`, `mean`, `sum`, `product` via `_translate_math_intent`. |
| `search_tool` | `tools/search.py` | Web search via DuckDuckGo or Tavily. Dispatched by `SEARCH_PROVIDER`. |
| `document_lookup` | `tools/document_lookup.py` | Tenant-scoped document fetch by ID. Injected per request via `get_document_fn`. |

### 4.3 Math Shortcut

Before invoking the graph, `_translate_math_intent(message)` parses natural-language math (e.g. "average of 1, 2, 5, 6") into an expression. If matched, the calculator runs directly; no LLM call. Supports thousands format (e.g. `1,000`).

### 4.4 Fallback When LLM Fails

If the final answer is malformed (tool-call JSON), empty, or "No response.", the agent:

1. Calls `_search_and_summarize`: search → LLM summarize.
2. If that fails, returns raw search results (first block, truncated to 800 chars).
3. If search fails, returns a generic "couldn't find an answer" message.

### 4.5 Search Query Refinement

`_search_query_from_message` strips question prefixes ("what is", "how does", etc.) and stop words to produce a focused query for the search tool.

---

## 5. RAG Pipeline

### 5.1 Indexing

`RAGPipeline.index_document`:

1. **Chunk**: `chunk_text(text, chunk_size=500, chunk_overlap=50)` in `rag/chunking.py`. Splits on sentence boundaries (`.` or `\n`) when possible.
2. **Embed**: `embedding_service.embed(chunk)` concurrently for indexed chunks. Currently `mock` (deterministic hash-based vectors of `embedding_dimension`).
3. **Store**: Delete existing chunks for the document; insert new `DocumentChunk` rows (tenant_id, document_id, chunk_index, text, embedding as JSON).

### 5.2 Retrieval

`RAGPipeline.retrieve`:

1. Embed the query.
2. Load chunk fields only (optionally filtered by `document_ids`), not full ORM rows.
3. Compute cosine similarity between query embedding and each chunk embedding.
4. Maintain a top-k heap instead of sorting the full candidate set.
5. Return the highest scoring chunks with `text`, `document_id`, `chunk_index`, `score`.

### 5.3 Query Flow

`services_rag.py`:

1. Call `rag_pipeline.retrieve` for top_k chunks.
2. Build context from chunk texts.
3. Call LLM with prompt: question + context.
4. Stream or return answer.

### 5.4 Embedding Model

`rag/embeddings.py`: `EmbeddingService` uses `embedding_model` from config. `mock` produces deterministic vectors via SHA-256 hash; dimension from `embedding_dimension`. Future: OpenAI, sentence-transformers.

---

## 6. AI Flows

### 6.1 Notary Summarization

`run_notary_summarization_flow`: Prompt instructs LLM to summarize in structured form (title, key points, parties, risks). Optional `document_id` loads text from DB. Output parsed into `NotarySummary`; audit logged.

### 6.2 Classification

`run_classify_flow`: Zero-shot classification with `candidate_labels`. LLM returns label + confidence; fallback to first label if parsing fails.

### 6.3 Ask

`run_ask_flow` / `run_ask_flow_stream`: Q&A over provided `context`. Streaming via `llm_client.stream_complete`.

### 6.4 Audit

All AI flows log to `ai_call_audit`: tenant_id, flow_name, success, request_payload, response_payload. Purge via `python -m app.audit` when `AI_AUDIT_RETENTION_DAYS` > 0.

---

## 7. Frontend

### 7.1 Stack

- React 19, TypeScript
- Vite 7, Tailwind CSS v4
- Vitest, jsdom for tests

### 7.2 Structure

```
frontend/src/
├── main.tsx
├── App.tsx
├── api.ts              # Fetch wrappers, streaming
├── tabConfig.ts        # Tab definitions
├── components/
│   ├── Alert.tsx
│   ├── Layout.tsx
│   ├── FileLoadButton.tsx
│   ├── ResultPreview.tsx
│   └── tabs/
│       ├── HealthTab.tsx
│       ├── DocumentsTab.tsx
│       ├── RAGTab.tsx
│       ├── AgentTab.tsx
│       ├── ClassifyTab.tsx
│       ├── NotaryTab.tsx
│       └── AskTab.tsx
└── hooks/
    └── useLoadFile.ts
```

### 7.3 API Client

`api.ts` exports typed functions for each endpoint. `agentChatStream` consumes SSE: validates `ReadableStream`, reads chunks, parses `data: {...}` lines, yields `{ token?, done?, error? }`.

### 7.4 Vite Proxy

`vite.config.ts` proxies `/api` and `/metrics` to `VITE_PROXY_TARGET` (default `http://localhost:8000`). In Docker, `host.docker.internal:8000` so frontend reaches API on host.

---

## 8. Data Model

### 8.1 Tables

| Table | Purpose |
|-------|---------|
| `documents` | Tenant-scoped documents. id (PK), tenant_id, title, text, created_at. |
| `document_chunks` | RAG chunks. id, tenant_id, document_id, chunk_index, text, embedding (JSON). |
| `ai_call_audit` | AI call logs. id, tenant_id, flow_name, request_payload, response_payload, success, created_at. |

### 8.2 Indexes

- `documents`: tenant_id
- `document_chunks`: tenant_id, document_id
- `ai_call_audit`: tenant_id, flow_name

---

## 9. Infrastructure

### 9.1 Docker Compose

- **api**: API with hot reload, Postgres, Redis, Ollama.
- **frontend**: Vite dev server.
- **frontend-prod** (profile): nginx serving built SPA on port 80.
- **migrate**: One-off Alembic upgrade.
- **db**: Postgres 16.
- **redis**: Redis 7.
- **ollama**: Ollama with `OLLAMA_KEEP_ALIVE=-1`.

### 9.2 Kubernetes

`k8s/` contains Kustomize resources: ConfigMap, Postgres, Redis, Deployments (api, ui), Services. Images from GHCR. Init container waits for Postgres before API start.

---

## 10. Security & Middleware

- **Request ID**: UUID in `X-Request-ID`; bound to structlog context.
- **CORS**: Configurable via `CORS_ALLOWED_ORIGINS`.
- **Security headers**: X-Content-Type-Options, X-Frame-Options, Referrer-Policy; HSTS in prod.
- **API key**: When `API_KEY` set, all `/api/v1/*` and `/metrics` require `X-API-Key`. `/health` exempt.
- **Rate limit**: Per-tenant, Redis-backed, configurable requests/minute.
- **Tenant**: `X-Tenant-ID` header; default from config.

---

## 11. Roadmap

Planned enhancements:

| Area | Description |
|------|-------------|
| **Observability** | Langfuse / OpenTelemetry tracing |
| **Evaluations** | Regression suite, LLM-as-judge |
| **Guardrails** | Content filter, PII redaction |
| **Task queue** | Celery/ARQ for async evals, batch indexing |
| **Vector store** | pgvector for native vector search (replace JSON + cosine in Python) |
| **Embeddings** | OpenAI / sentence-transformers integration |

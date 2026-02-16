# AI Agent Engineering Platform — Architecture & Technology Roadmap

This document outlines the architectural changes and technology additions needed to transform the current AI Platform into a **production-grade AI agent engineering showcase** that demonstrates all major capabilities in AI and agentic engineering.

---

## 1. Current State Summary

| Layer | Current | Gaps |
|-------|---------|------|
| **Backend** | FastAPI, async SQLAlchemy, Postgres, Redis | — |
| **AI** | Simple LLM calls (Ollama/OpenAI), no streaming | No agents, tools, RAG, or orchestration |
| **Flows** | 3 basic flows: summarize, classify, ask | No multi-step reasoning, no tool use |
| **Frontend** | React, Vite, Tailwind | No streaming UI, no agent playground |
| **Observability** | Prometheus metrics, basic audit | No tracing, cost tracking, or eval framework |
| **Infra** | Docker Compose, K8s | — |

---

## 2. Target Capabilities to Showcase

### 2.1 Agentic Capabilities
- **ReAct / Tool-calling agents** — agents that decide when to call tools (search, calculator, DB, APIs)
- **Multi-agent orchestration** — planner + researcher + writer agents
- **Human-in-the-loop** — pause for approval before sensitive actions
- **Conversation memory** — session-aware, summarization, context window management
- **Streaming** — token-by-token responses for perceived performance

### 2.2 RAG & Retrieval
- **Vector search** — embeddings + semantic retrieval
- **Hybrid search** — BM25 + vector for better recall
- **Document chunking** — semantic, recursive, or fixed-size
- **Re-ranking** — cross-encoder or LLM-based re-ranking

### 2.3 Production Readiness
- **Evaluations** — regression tests, LLM-as-judge, evals framework
- **Guardrails** — content filtering, output validation, PII redaction
- **Cost & usage tracking** — token counts, cost per tenant/flow
- **Observability** — distributed tracing, span visualization

### 2.4 Advanced AI Patterns
- **Structured output** — JSON mode, function calling, Pydantic extraction
- **Chain-of-thought** — explicit reasoning steps
- **Self-correction** — retry with feedback, validation loops

---

## 3. Architectural Changes

### 3.1 High-Level Architecture (Target)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React)                                     │
│  Health | Documents | RAG | Agents | Evaluations | Playground | Settings          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           API GATEWAY (FastAPI)                                  │
│  Auth | Rate Limit | Tenant | Request ID | CORS | Security Headers               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  Agent Service   │         │   RAG Service    │         │  Eval Service    │
│  LangGraph       │         │  Embeddings +    │         │  Regression +    │
│  Tools/ReAct     │         │  Vector Store    │         │  LLM-as-Judge    │
└──────────────────┘         └──────────────────┘         └──────────────────┘
          │                             │                             │
          └─────────────────────────────┼─────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         LLM GATEWAY (abstraction layer)                          │
│  Ollama | OpenAI | Anthropic | Azure OpenAI | vLLM | LiteLLM (multi-provider)   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  PostgreSQL      │         │  Redis           │         │  Vector Store     │
│  + pgvector      │         │  Cache | Queue   │         │  (pgvector or     │
│  Documents       │         │  Rate Limit      │         │   Qdrant)         │
│  Audit | Memory  │         │  Session         │         │                   │
└──────────────────┘         └──────────────────┘         └──────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY (Langfuse / LangSmith / OpenTelemetry)          │
│  Traces | Spans | Cost | Latency | Token Usage | Eval Results                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Service Decomposition (Optional, for Scale)

For a **showcase** platform, a **modular monolith** is recommended initially. Split into microservices only if needed:

| Service | Responsibility | When to Split |
|---------|----------------|---------------|
| **api** | HTTP, auth, routing, orchestration | — |
| **agent-runtime** | LangGraph graphs, tool execution | If agent load is high |
| **rag-service** | Embeddings, indexing, retrieval | If RAG volume is high |
| **eval-worker** | Async eval jobs | If evals are heavy |

---

## 4. Technology Additions & Changes

### 4.1 Agent Orchestration

| Technology | Purpose | Recommendation |
|------------|---------|----------------|
| **LangGraph** | Graph-based agent orchestration, state, human-in-the-loop | **Add** — industry standard for production agents |
| **LangChain** | Chains, prompts, integrations | **Add** — use for RAG, prompts, loaders; LangGraph for agents |
| **LiteLLM** | Multi-provider LLM routing (OpenAI, Anthropic, Ollama, etc.) | **Add** — simplifies provider switching |

**Migration path:**
- Keep existing `services_llm.py` as fallback; add LiteLLM for new flows
- Introduce `app/agents/` with LangGraph `StateGraph` definitions
- Add tool definitions (search, calculator, document lookup, etc.)

### 4.2 RAG & Vector Store

| Technology | Purpose | Recommendation |
|------------|---------|----------------|
| **pgvector** | Vector search in Postgres | **Add** — you already use Postgres; minimal new infra |
| **Qdrant** | Dedicated vector DB (optional) | Consider if pgvector hits limits (~10M vectors) |
| **sentence-transformers** / **OpenAI embeddings** | Embedding models | **Add** — configurable (local vs API) |
| **LangChain document loaders** | PDF, DOCX, web, etc. | **Add** — for richer ingestion |

**Recommendation:** Start with **pgvector** — same DB, ACID, no new services. Add Qdrant later if needed.

### 4.3 Streaming

| Technology | Purpose | Recommendation |
|------------|---------|----------------|
| **Server-Sent Events (SSE)** | Stream LLM tokens to frontend | **Add** — FastAPI supports `StreamingResponse` |
| **LangChain/LangGraph streaming** | Token streaming from agents | **Add** — `astream_events` or `astream` |

### 4.4 Observability & Tracing

| Technology | Purpose | Recommendation |
|------------|---------|----------------|
| **Langfuse** | Open-source LLM observability (traces, cost, evals) | **Add** — self-hosted or cloud |
| **OpenTelemetry** | Distributed tracing | **Add** — for non-LLM spans |
| **Prometheus** | Metrics (existing) | **Keep** — extend with agent-specific metrics |

### 4.5 Evaluations

| Technology | Purpose | Recommendation |
|------------|---------|----------------|
| **LangSmith / Langfuse evals** | Regression, LLM-as-judge | **Add** — integrate with Langfuse |
| **pytest + fixtures** | Deterministic evals | **Add** — for non-LLM assertions |
| **RAGAS** | RAG-specific metrics (faithfulness, answer relevancy) | **Add** — for RAG flows |

### 4.6 Guardrails & Safety

| Technology | Purpose | Recommendation |
|------------|---------|----------------|
| **NeMo Guardrails** / **Llama Guard** | Content filtering | **Add** — configurable |
| **Pydantic output parsing** | Structured output validation | **Add** — already using Pydantic |
| **Presidio** | PII detection/redaction | **Add** — for compliance |

### 4.7 Task Queue (for long-running agents/evals)

| Technology | Purpose | Recommendation |
|------------|---------|----------------|
| **Celery** + Redis | Async tasks | **Add** — for evals, batch RAG indexing |
| **ARQ** | Lightweight async Redis queue | Alternative — simpler than Celery |
| **Temporal** | Durable workflows | Consider for complex human-in-the-loop |

### 4.8 Frontend Enhancements

| Technology | Purpose | Recommendation |
|------------|---------|----------------|
| **React Query / TanStack Query** | Server state, caching | **Add** — better data fetching |
| **Markdown rendering** | LLM output (code, lists) | **Add** — `react-markdown` |
| **Streaming UI** | Token-by-token display | **Add** — `EventSource` or fetch + ReadableStream |
| **Agent playground** | Interactive agent testing | **Add** — new tab with chat + tool visibility |

---

## 5. Proposed Directory Structure

```
api/
├── app/
│   ├── agents/                    # NEW: LangGraph agents
│   │   ├── __init__.py
│   │   ├── tools/                  # Tool definitions
│   │   │   ├── search.py
│   │   │   ├── calculator.py
│   │   │   └── document_lookup.py
│   │   ├── react_agent.py          # ReAct agent graph
│   │   └── multi_agent.py          # Multi-agent orchestration
│   ├── rag/                        # NEW: RAG pipeline
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── chunking.py
│   │   ├── retrieval.py
│   │   └── pipeline.py
│   ├── evals/                      # NEW: Evaluation framework
│   │   ├── __init__.py
│   │   ├── runners.py
│   │   └── datasets/
│   ├── guardrails/                 # NEW: Content safety
│   │   ├── __init__.py
│   │   └── filters.py
│   ├── api/
│   │   ├── v1/
│   │   │   ├── agents.py           # NEW: Agent endpoints
│   │   │   ├── rag.py              # NEW: RAG endpoints
│   │   │   ├── evals.py            # NEW: Eval endpoints
│   │   │   └── ...
│   │   └── deps.py
│   ├── services_llm.py             # Keep; extend with LiteLLM
│   ├── services_ai_flows.py        # Keep; refactor to use agents
│   └── ...
├── tests/
│   ├── evals/                      # NEW: Eval test datasets
│   └── ...
frontend/
├── src/
│   ├── components/
│   │   ├── AgentPlayground.tsx     # NEW
│   │   ├── RAGTab.tsx              # NEW
│   │   ├── StreamingResponse.tsx   # NEW
│   │   └── ...
│   └── ...
```

---

## 6. Implementation Phases

### Phase 1: Foundation (4–6 weeks)
1. Add **pgvector** to Postgres; create `document_embeddings` table
2. Add **LiteLLM** for multi-provider support
3. Add **streaming** to existing `/ai/ask` and new agent endpoints
4. Add **Langfuse** (or LangSmith) for tracing
5. Frontend: streaming response component, Markdown rendering

### Phase 2: RAG (2–3 weeks)
1. Embedding service (sentence-transformers or OpenAI)
2. Chunking pipeline for documents
3. RAG retrieval + re-ranking
4. New `/ai/rag/query` endpoint with streaming
5. Frontend: RAG tab with document upload + query

### Phase 3: Agents (4–6 weeks)
1. **LangGraph** + **LangChain** setup
2. Tool definitions (search, calculator, document lookup)
3. ReAct agent graph
4. Human-in-the-loop node (optional)
5. New `/ai/agents/chat` endpoint (streaming)
6. Frontend: Agent playground with tool visibility

### Phase 4: Production Hardening (2–3 weeks)
1. **Evaluations** — regression suite, LLM-as-judge
2. **Guardrails** — content filter, PII redaction
3. **Cost tracking** — token usage per tenant/flow
4. **Task queue** — Celery/ARQ for async evals and batch indexing

### Phase 5: Advanced (Ongoing)
1. Multi-agent orchestration
2. Fine-tuning pipeline (optional)
3. Advanced RAG (hybrid search, re-ranking)
4. Self-hosted Langfuse dashboard

---

## 7. Dependency Additions (pyproject.toml)

```toml
# Agent & LLM
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
litellm>=1.50.0

# RAG & Embeddings
sentence-transformers>=3.0.0   # or openai for API embeddings
pgvector>=0.2.0

# Observability
langfuse>=2.0.0                # or opentelemetry-api, opentelemetry-sdk

# Guardrails (optional)
presidio-analyzer>=2.2.0
presidio-anonymizer>=2.2.0

# Task queue (optional)
celery[redis]>=5.4.0
# or arq>=0.25.0
```

---

## 8. Infrastructure Changes

### Docker Compose
- Add **Langfuse** service (or use cloud)
- Add **Qdrant** (optional, if not using pgvector)
- Enable **pgvector** in Postgres image: `pgvector/pgvector:pg16`
- Add **Celery worker** (optional)

### Kubernetes
- New ConfigMaps/Secrets for Langfuse, embedding API keys
- Optional: separate deployment for eval workers

### Environment Variables (New)
```
# LLM
LITELLM_PROVIDER=ollama
OPENAI_API_KEY=...          # if using OpenAI embeddings

# RAG
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE=pgvector       # or qdrant

# Observability
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_HOST=https://...

# Guardrails
ENABLE_CONTENT_FILTER=true
ENABLE_PII_REDACTION=true
```

---

## 9. Summary: What to Add vs Change

| Category | Add | Change |
|----------|-----|--------|
| **Agent framework** | LangGraph, LangChain, LiteLLM | Replace direct httpx LLM calls with LiteLLM |
| **RAG** | pgvector, embeddings, chunking, retrieval | Extend document ingestion |
| **Streaming** | SSE endpoints, streaming UI | Add streaming to ask/agent endpoints |
| **Observability** | Langfuse, OpenTelemetry | Extend Prometheus with agent metrics |
| **Evals** | Langfuse evals, RAGAS, pytest fixtures | New eval framework |
| **Guardrails** | Presidio, content filter | Middleware or pre/post hooks |
| **Task queue** | Celery or ARQ | For async evals, batch indexing |
| **Frontend** | Agent playground, RAG tab, streaming, Markdown | New components |

---

## 10. References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Langfuse](https://langfuse.com/)
- [pgvector](https://github.com/pgvector/pgvector)
- [LiteLLM](https://docs.litellm.ai/)
- [RAGAS - RAG Evaluation](https://docs.ragas.io/)

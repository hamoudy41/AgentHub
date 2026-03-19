# Frontend

React + TypeScript frontend for AgentHub. Tab-based UI for health, documents, RAG, agents, classification, notary summarization, and Q&A.

---

## Prerequisites

- Node.js 20.19+ or 22.12+
- Corepack-enabled pnpm

---

## Development

### Setup

```bash
cd frontend
corepack enable
pnpm install
```

### Run

```bash
pnpm run dev
```

App at http://localhost:5173. Vite proxies `/api` and `/metrics` to the backend (default `http://localhost:8000`).

### Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE` | `/api/v1` | API base path. Used by `api.ts` for fetch URLs. |
| `VITE_API_KEY` | — | Optional API key for `X-API-Key` header. |
| `VITE_PROXY_TARGET` | `http://localhost:8000` | Target for Vite dev server proxy. |

In Docker, set `VITE_PROXY_TARGET=http://host.docker.internal:8000` so the container reaches the API on the host.

---

## Build

```bash
pnpm run build
```

Output in `dist/`. Nginx (or similar) serves `index.html` and static assets. For production, configure the API base URL (e.g. `VITE_API_BASE=https://api.example.com/api/v1` at build time).

---

## Testing

```bash
pnpm test
```

Vitest runs in watch mode by default. For CI:

```bash
pnpm run test:coverage
```

Coverage thresholds: 80% branches, functions, lines, statements. Reports in `coverage/`.

---

## Deployment

### Docker

```bash
docker build -t ai-platform-ui .
docker run -p 80:80 ai-platform-ui
```

Dockerfile uses multi-stage build: node for build, nginx for serve. Nginx serves SPA and proxies `/api` to the API service (configure via env or build args).

---

## Project Structure

```
frontend/
├── src/
│   ├── main.tsx           # Entry
│   ├── App.tsx             # Tab layout
│   ├── api.ts              # API client, streaming
│   ├── api.test.ts
│   ├── tabConfig.ts        # Tab metadata
│   ├── components/
│   │   ├── Alert.tsx
│   │   ├── Layout.tsx
│   │   ├── FileLoadButton.tsx
│   │   ├── ResultPreview.tsx
│   │   └── tabs/           # Per-tab components
│   ├── hooks/
│   │   └── useLoadFile.ts
│   └── test/
│       └── setup.ts
├── index.html
├── vite.config.ts
├── tailwind.config.js
└── package.json
```

---

## API Client

`api.ts` exports typed functions for each endpoint. Streaming endpoints use `fetch` + `ReadableStream`; parse SSE `data:` lines and yield `{ token?, done?, error? }`.

### Streaming

`agentChatStream`, `ask` (stream), `ragQuery` (stream) return `AsyncGenerator`. Consumer iterates with `for await` or `generator.next()`.

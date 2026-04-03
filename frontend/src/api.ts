const API_BASE = import.meta.env.VITE_API_BASE ?? '/api/v1'
const API_KEY = import.meta.env.VITE_API_KEY ?? undefined

interface RequestContext {
  apiKey?: string
  tenantId?: string
}

export interface HealthStatus {
  environment: string
  timestamp: string
  db_ok?: boolean
  redis_ok?: boolean | null
  llm_ok?: boolean
}

export interface DocumentRead {
  id: string
  title: string
  text: string
  created_at: string
}

export interface DocumentCreate {
  id: string
  title: string
  text: string
}

export interface ClassifyResponse {
  label: string
  confidence: number
  model: string
  source: string
  metadata?: Record<string, unknown>
}

export interface NotarySummary {
  title: string
  key_points: string[]
  parties_involved: string[]
  risks_or_warnings: string[]
  raw_summary: string
}

export interface NotarySummarizeResponse {
  document_id: string | null
  summary: NotarySummary
  source: string
  metadata?: Record<string, unknown>
}

export interface AskResponse {
  answer: string
  model: string
  source: string
  metadata?: Record<string, unknown>
}

async function parseError(r: Response): Promise<string> {
  const text = await r.text()
  try {
    const json = JSON.parse(text) as { detail?: string | { msg?: string }[] }
    const d = json.detail
    if (typeof d === 'string') return d
    if (Array.isArray(d) && d[0]?.msg) return d[0].msg
  } catch {
    // JSON may not be valid
  }
  return text || `Request failed (${r.status})`
}

function buildApiUrl(path: string): string {
  return `${API_BASE}${path}`
}

function buildHeaders(
  { apiKey, tenantId }: RequestContext = {},
  contentType: string | null = 'application/json'
): Record<string, string> {
  const resolvedHeaders: Record<string, string> = {
    'X-Tenant-ID': tenantId ?? 'default',
  }

  if (contentType) {
    resolvedHeaders['Content-Type'] = contentType
  }

  const key = apiKey ?? API_KEY
  if (key) {
    resolvedHeaders['X-API-Key'] = key
  }

  return resolvedHeaders
}

async function requestJson<T>(
  path: string,
  init: Omit<RequestInit, 'body' | 'headers'> & { body?: unknown } = {},
  context: RequestContext = {}
): Promise<T> {
  const { body, ...requestInit } = init
  const response = await fetch(buildApiUrl(path), {
    ...requestInit,
    headers: buildHeaders(context),
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!response.ok) throw new Error(await parseError(response))
  return response.json()
}

async function requestForm<T>(
  path: string,
  formData: FormData,
  init: Omit<RequestInit, 'body' | 'headers'> = {},
  context: RequestContext = {}
): Promise<T> {
  const response = await fetch(buildApiUrl(path), {
    ...init,
    headers: buildHeaders(context, null),
    body: formData,
  })
  if (!response.ok) throw new Error(await parseError(response))
  return response.json()
}

async function openStream(
  path: string,
  body: unknown,
  context: RequestContext = {}
): Promise<Response> {
  const response = await fetch(buildApiUrl(path), {
    method: 'POST',
    headers: buildHeaders(context),
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error(await parseError(response))
  return response
}

export async function getHealth(
  apiKey?: string,
  tenantId?: string
): Promise<HealthStatus> {
  return requestJson('/health', {}, { apiKey, tenantId })
}

export async function createDocument(
  payload: DocumentCreate,
  apiKey?: string,
  tenantId?: string
): Promise<DocumentRead> {
  return requestJson('/documents', { method: 'POST', body: payload }, { apiKey, tenantId })
}

export async function uploadDocument(
  file: File,
  options: { documentId?: string; title?: string } = {},
  apiKey?: string,
  tenantId?: string
): Promise<DocumentRead> {
  const form = new FormData()
  form.append('file', file)
  if (options.documentId) form.append('document_id', options.documentId)
  if (options.title) form.append('title', options.title)
  return requestForm('/documents/upload', form, { method: 'POST' }, { apiKey, tenantId })
}

export async function getDocument(
  id: string,
  apiKey?: string,
  tenantId?: string
): Promise<DocumentRead> {
  return requestJson(`/documents/${id}`, {}, { apiKey, tenantId })
}

export async function classify(
  text: string,
  candidateLabels: string[],
  apiKey?: string,
  tenantId?: string
): Promise<ClassifyResponse> {
  return requestJson(
    '/ai/classify',
    {
      method: 'POST',
      body: { text, candidate_labels: candidateLabels },
    },
    { apiKey, tenantId }
  )
}

export async function notarySummarize(
  text: string,
  options: { documentId?: string; language?: 'nl' | 'en' } = {},
  apiKey?: string,
  tenantId?: string
): Promise<NotarySummarizeResponse> {
  const body: Record<string, unknown> = { text }
  if (options.documentId) body.document_id = options.documentId
  if (options.language) body.language = options.language
  return requestJson('/ai/notary/summarize', { method: 'POST', body }, { apiKey, tenantId })
}

export async function ask(
  question: string,
  context: string,
  apiKey?: string,
  tenantId?: string
): Promise<AskResponse> {
  return requestJson(
    '/ai/ask',
    { method: 'POST', body: { question, context } },
    { apiKey, tenantId }
  )
}

export interface RAGQueryResponse {
  answer: string
  sources: Array<{ text: string; document_id: string; score: number }>
  model: string
  metadata?: Record<string, unknown>
}

export interface RAGIndexResponse {
  document_id: string
  chunks_indexed: number
  status: 'indexed'
}

export async function ragQuery(
  query: string,
  options: { documentIds?: string[]; topK?: number } = {},
  apiKey?: string,
  tenantId?: string
): Promise<RAGQueryResponse> {
  const body: Record<string, unknown> = { query }
  if (options.documentIds?.length) body.document_ids = options.documentIds
  if (options.topK != null) body.top_k = options.topK
  return requestJson('/ai/rag/query', { method: 'POST', body }, { apiKey, tenantId })
}

export async function ragIndex(
  documentId: string,
  apiKey?: string,
  tenantId?: string
): Promise<RAGIndexResponse> {
  return requestJson(
    '/ai/rag/index',
    { method: 'POST', body: { document_id: documentId } },
    { apiKey, tenantId }
  )
}

export interface AgentChatResponse {
  answer: string
  tools_used: string[]
  error?: string
}

export async function agentChat(
  message: string,
  apiKey?: string,
  tenantId?: string
): Promise<AgentChatResponse> {
  return requestJson(
    '/ai/agents/chat',
    { method: 'POST', body: { message } },
    { apiKey, tenantId }
  )
}

function getStreamReader(response: Response): ReadableStreamDefaultReader<Uint8Array> {
  if (!response.body) {
    throw new TypeError('Stream error: missing response body')
  }
  if (typeof response.body.getReader !== 'function') {
    throw new TypeError('Stream error: invalid response body')
  }
  try {
    return response.body.getReader()
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    throw new Error(`Stream error: ${msg}`, { cause: err })
  }
}

async function readStreamChunk(
  reader: ReadableStreamDefaultReader<Uint8Array>
): Promise<ReadableStreamReadResult<Uint8Array>> {
  try {
    return await reader.read()
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    throw new Error(`Stream error: ${msg}`, { cause: err })
  }
}

function parseSsePayloads(
  buffer: string
): { nextBuffer: string; payloads: Array<{ token?: string; done?: boolean; error?: string }> } {
  const lines = buffer.split('\n')
  const nextBuffer = lines.pop() ?? ''
  const payloads: Array<{ token?: string; done?: boolean; error?: string }> = []

  for (const line of lines) {
    if (!line.startsWith('data: ')) continue

    const data = line.slice(6)
    if (data === '[DONE]' || !data || data.trim() === '') continue

    try {
      payloads.push(JSON.parse(data) as { token?: string; done?: boolean; error?: string })
    } catch {
      // Ignore malformed chunks and continue streaming.
    }
  }

  return { nextBuffer, payloads }
}

export async function* agentChatStream(
  message: string,
  apiKey?: string,
  tenantId?: string
): AsyncGenerator<{ token?: string; done?: boolean; error?: string }> {
  const r = await openStream('/ai/agents/chat/stream', { message }, { apiKey, tenantId })
  const reader = getStreamReader(r)

  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const readResult = await readStreamChunk(reader)
      const { done, value } = readResult
      if (done) break

      if (!value) continue

      buffer += decoder.decode(value, { stream: true })
      const { nextBuffer, payloads } = parseSsePayloads(buffer)
      buffer = nextBuffer
      for (const payload of payloads) {
        yield payload
      }
    }
  } finally {
    try {
      reader.releaseLock()
    } catch {
      // reader may already be released
    }
  }
}

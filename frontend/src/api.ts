const API_BASE = import.meta.env.VITE_API_BASE ?? '/api/v1'
const API_KEY = import.meta.env.VITE_API_KEY ?? undefined

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
    /* ignore parse errors */
  }
  return text || `Request failed (${r.status})`
}

function headers(apiKey?: string, tenantId?: string): HeadersInit {
  const h: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-Tenant-ID': tenantId ?? 'default',
  }
  const key = apiKey ?? API_KEY
  if (key) h['X-API-Key'] = key
  return h
}

export async function getHealth(
  apiKey?: string,
  tenantId?: string
): Promise<HealthStatus> {
  const r = await fetch(`${API_BASE}/health`, {
    headers: headers(apiKey, tenantId),
  })
  if (!r.ok) throw new Error(await parseError(r))
  return r.json()
}

export async function createDocument(
  payload: DocumentCreate,
  apiKey?: string,
  tenantId?: string
): Promise<DocumentRead> {
  const r = await fetch(`${API_BASE}/documents`, {
    method: 'POST',
    headers: headers(apiKey, tenantId),
    body: JSON.stringify(payload),
  })
  if (!r.ok) throw new Error(await parseError(r))
  return r.json()
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
  const h = headers(apiKey, tenantId) as Record<string, string>
  delete h['Content-Type']
  const r = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    headers: h,
    body: form,
  })
  if (!r.ok) throw new Error(await parseError(r))
  return r.json()
}

export async function getDocument(
  id: string,
  apiKey?: string,
  tenantId?: string
): Promise<DocumentRead> {
  const r = await fetch(`${API_BASE}/documents/${id}`, {
    headers: headers(apiKey, tenantId),
  })
  if (!r.ok) throw new Error(await parseError(r))
  return r.json()
}

export async function classify(
  text: string,
  candidateLabels: string[],
  apiKey?: string,
  tenantId?: string
): Promise<ClassifyResponse> {
  const r = await fetch(`${API_BASE}/ai/classify`, {
    method: 'POST',
    headers: headers(apiKey, tenantId),
    body: JSON.stringify({ text, candidate_labels: candidateLabels }),
  })
  if (!r.ok) throw new Error(await parseError(r))
  return r.json()
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
  const r = await fetch(`${API_BASE}/ai/notary/summarize`, {
    method: 'POST',
    headers: headers(apiKey, tenantId),
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(await parseError(r))
  return r.json()
}

export async function ask(
  question: string,
  context: string,
  apiKey?: string,
  tenantId?: string
): Promise<AskResponse> {
  const r = await fetch(`${API_BASE}/ai/ask`, {
    method: 'POST',
    headers: headers(apiKey, tenantId),
    body: JSON.stringify({ question, context }),
  })
  if (!r.ok) throw new Error(await parseError(r))
  return r.json()
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
  const r = await fetch(`${API_BASE}/ai/rag/query`, {
    method: 'POST',
    headers: headers(apiKey, tenantId),
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(await parseError(r))
  return r.json()
}

export async function ragIndex(
  documentId: string,
  apiKey?: string,
  tenantId?: string
): Promise<RAGIndexResponse> {
  const r = await fetch(`${API_BASE}/ai/rag/index`, {
    method: 'POST',
    headers: headers(apiKey, tenantId),
    body: JSON.stringify({ document_id: documentId }),
  })
  if (!r.ok) throw new Error(await parseError(r))
  return r.json()
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
  const r = await fetch(`${API_BASE}/ai/agents/chat`, {
    method: 'POST',
    headers: headers(apiKey, tenantId),
    body: JSON.stringify({ message }),
  })
  if (!r.ok) throw new Error(await parseError(r))
  return r.json()
}

export async function* agentChatStream(
  message: string,
  apiKey?: string,
  tenantId?: string
): AsyncGenerator<{ token?: string; done?: boolean; error?: string }> {
  const r = await fetch(`${API_BASE}/ai/agents/chat/stream`, {
    method: 'POST',
    headers: headers(apiKey, tenantId),
    body: JSON.stringify({ message }),
  })
  if (!r.ok) throw new Error(await parseError(r))
  
  if (!r.body) {
    throw new Error('Response body is missing from server')
  }
  
  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  
  try {
    while (true) {
      let readResult
      try {
        readResult = await reader.read()
      } catch (error) {
        throw new Error(
          `Failed to read from stream: ${error instanceof Error ? error.message : 'Unknown error'}`
        )
      }
      
      const { done, value } = readResult
      if (done) break
      
      if (!value) {
        continue
      }
      
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6)
          if (data === '[DONE]') continue
          
          if (!data || data.trim() === '') {
            continue
          }
          
          try {
            const parsed = JSON.parse(data) as { token?: string; done?: boolean; error?: string }
            yield parsed
          } catch (error) {
            console.warn('agentChatStream: Skipping invalid JSON chunk:', data, error)
            continue
          }
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

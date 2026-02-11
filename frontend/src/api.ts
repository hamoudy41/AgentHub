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

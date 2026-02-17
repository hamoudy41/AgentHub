import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  agentChat,
  agentChatStream,
  ask,
  classify,
  createDocument,
  getDocument,
  getHealth,
  notarySummarize,
  ragIndex,
  ragQuery,
  uploadDocument,
} from './api'

describe('api', () => {
  const originalFetch = globalThis.fetch

  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve(new Response(JSON.stringify({ ok: true }), { status: 200 }))
      )
    )
  })

  afterEach(() => {
    vi.stubGlobal('fetch', originalFetch as typeof fetch)
    vi.unstubAllGlobals()
  })

  it('getHealth sends correct request and returns data', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({ environment: 'local', timestamp: '2024-01-01', db_ok: true }),
        { status: 200 }
      )
    )
    const result = await getHealth()
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/health'),
      expect.objectContaining({
        headers: expect.objectContaining({ 'X-Tenant-ID': 'default' }),
      })
    )
    expect(result).toEqual({
      environment: 'local',
      timestamp: '2024-01-01',
      db_ok: true,
    })
  })

  it('getHealth with apiKey includes X-API-Key', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ environment: 'local', timestamp: 'x' }), { status: 200 })
    )
    await getHealth('secret-key')
    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({ 'X-API-Key': 'secret-key' }),
      })
    )
  })

  it('getHealth with tenantId uses custom tenant', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ environment: 'local', timestamp: 'x' }), { status: 200 })
    )
    await getHealth(undefined, 'tenant-1')
    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({ 'X-Tenant-ID': 'tenant-1' }),
      })
    )
  })

  it('getHealth throws on non-ok response', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(new Response('Server error', { status: 500 }))
    await expect(getHealth()).rejects.toThrow()
  })

  it('createDocument sends correct payload', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          id: 'doc1',
          title: 'Test',
          text: 'Content',
          created_at: '2024-01-01',
        }),
        { status: 201 }
      )
    )
    const result = await createDocument({
      id: 'doc1',
      title: 'Test',
      text: 'Content',
    })
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/documents'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ id: 'doc1', title: 'Test', text: 'Content' }),
      })
    )
    expect(result.id).toBe('doc1')
  })

  it('createDocument with apiKey includes header', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ id: 'x', title: '', text: '', created_at: '' }), {
        status: 201,
      })
    )
    await createDocument({ id: 'x', title: '', text: '' }, 'key')
    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({ 'X-API-Key': 'key' }),
      })
    )
  })

  it('createDocument throws with API detail for 409 (duplicate ID)', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          detail: "Document with ID 'dup' already exists. Use Get by ID to view, or choose a different ID.",
        }),
        { status: 409, headers: { 'Content-Type': 'application/json' } }
      )
    )
    await expect(
      createDocument({ id: 'dup', title: 'T', text: 'C' })
    ).rejects.toThrow(/already exists/)
  })

  it('getDocument fetches by id', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({ id: 'doc1', title: 'T', text: 'C', created_at: 'x' }),
        { status: 200 }
      )
    )
    const result = await getDocument('doc1')
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/documents/doc1'),
      expect.any(Object)
    )
    expect(result.id).toBe('doc1')
  })

  it('getDocument throws on 404', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(new Response('Not found', { status: 404 }))
    await expect(getDocument('missing')).rejects.toThrow()
  })

  it('uploadDocument sends file and returns document', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          id: 'uploaded',
          title: 'My file.txt',
          text: 'File content',
          created_at: '2024-01-01',
        }),
        { status: 201 }
      )
    )
    const file = new File(['File content'], 'My file.txt', { type: 'text/plain' })
    const result = await uploadDocument(file)
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/documents/upload'),
      expect.objectContaining({
        method: 'POST',
        body: expect.any(FormData),
      })
    )
    expect(result.id).toBe('uploaded')
  })

  it('uploadDocument with documentId and title appends to form', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({ id: 'doc1', title: 'Custom Title', text: 'x', created_at: '' }),
        { status: 201 }
      )
    )
    const file = new File(['x'], 'f.txt')
    await uploadDocument(file, { documentId: 'doc1', title: 'Custom Title' })
    const call = mockFetch.mock.calls[0][1]
    const body = call?.body as FormData
    expect(body.get('document_id')).toBe('doc1')
    expect(body.get('title')).toBe('Custom Title')
  })

  it('uploadDocument throws on error', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(new Response('File too large', { status: 413 }))
    const file = new File(['x'], 'f.txt')
    await expect(uploadDocument(file)).rejects.toThrow()
  })

  it('uploadDocument throws with API detail for 413 (file too large)', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: 'File too large (max 5 MB)' }), {
        status: 413,
        headers: { 'Content-Type': 'application/json' },
      })
    )
    const file = new File(['x'], 'f.txt')
    await expect(uploadDocument(file)).rejects.toThrow('File too large (max 5 MB)')
  })

  it('uploadDocument throws with API detail for 400 (invalid encoding)', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({ detail: 'File could not be decoded as UTF-8 text' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    )
    const file = new File(['x'], 'f.txt')
    await expect(uploadDocument(file)).rejects.toThrow(
      'File could not be decoded as UTF-8 text'
    )
  })

  it('uploadDocument throws with API detail for 409 (duplicate ID)', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          detail: "Document with ID 'doc1' already exists. Use a different ID or Get by ID to view.",
        }),
        { status: 409, headers: { 'Content-Type': 'application/json' } }
      )
    )
    const file = new File(['x'], 'f.txt')
    await expect(uploadDocument(file)).rejects.toThrow(/already exists/)
  })

  it('uploadDocument uses plain text when response is not JSON', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(new Response('Plain error message', { status: 500 }))
    const file = new File(['x'], 'f.txt')
    await expect(uploadDocument(file)).rejects.toThrow('Plain error message')
  })

  it('classify sends text and labels', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          label: 'urgent',
          confidence: 0.9,
          model: 'llm',
          source: 'llm',
        }),
        { status: 200 }
      )
    )
    const result = await classify('Urgent request', ['urgent', 'normal'], 'key')
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/ai/classify'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          text: 'Urgent request',
          candidate_labels: ['urgent', 'normal'],
        }),
      })
    )
    expect(result.label).toBe('urgent')
  })

  it('notarySummarize with text only', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          document_id: null,
          summary: { title: 'S', key_points: [], parties_involved: [], risks_or_warnings: [], raw_summary: 'x' },
          source: 'llm',
        }),
        { status: 200 }
      )
    )
    const result = await notarySummarize('Document text...')
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/ai/notary/summarize'),
      expect.objectContaining({
        body: JSON.stringify({ text: 'Document text...' }),
      })
    )
    expect(result.summary.title).toBe('S')
  })

  it('notarySummarize with documentId and language', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          document_id: 'doc1',
          summary: { title: 'S', key_points: [], parties_involved: [], risks_or_warnings: [], raw_summary: 'x' },
          source: 'llm',
        }),
        { status: 200 }
      )
    )
    const result = await notarySummarize('Text', { documentId: 'doc1', language: 'en' })
    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        body: JSON.stringify({
          text: 'Text',
          document_id: 'doc1',
          language: 'en',
        }),
      })
    )
    expect(result.document_id).toBe('doc1')
  })

  it('notarySummarize throws on error', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(new Response('Bad request', { status: 400 }))
    await expect(notarySummarize('')).rejects.toThrow()
  })

  it('ask sends question and context', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          answer: 'The answer',
          model: 'llm',
          source: 'llm',
        }),
        { status: 200 }
      )
    )
    const result = await ask('What is X?', 'Context about X', 'key', 'tenant')
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/ai/ask'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ question: 'What is X?', context: 'Context about X' }),
        headers: expect.objectContaining({ 'X-Tenant-ID': 'tenant' }),
      })
    )
    expect(result.answer).toBe('The answer')
  })

  it('ask throws on error', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(new Response('Error', { status: 500 }))
    await expect(ask('Q', 'C')).rejects.toThrow()
  })

  it('ragQuery sends query and returns answer', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          answer: 'Paris',
          sources: [],
          model: 'llama3.2',
        }),
        { status: 200 }
      )
    )
    const result = await ragQuery('Capital of France?')
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/ai/rag/query'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ query: 'Capital of France?' }),
      })
    )
    expect(result.answer).toBe('Paris')
  })

  it('ragQuery with documentIds and topK', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({ answer: 'x', sources: [], model: 'llm' }),
        { status: 200 }
      )
    )
    await ragQuery('Q', { documentIds: ['d1', 'd2'], topK: 10 })
    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        body: JSON.stringify({ query: 'Q', document_ids: ['d1', 'd2'], top_k: 10 }),
      })
    )
  })

  it('ragIndex sends document_id and returns chunks_indexed', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          document_id: 'doc1',
          chunks_indexed: 3,
          status: 'indexed',
        }),
        { status: 200 }
      )
    )
    const result = await ragIndex('doc1')
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/ai/rag/index'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ document_id: 'doc1' }),
      })
    )
    expect(result.chunks_indexed).toBe(3)
  })

  it('agentChat sends message and returns answer', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          answer: '42',
          tools_used: ['calculator'],
        }),
        { status: 200 }
      )
    )
    const result = await agentChat('What is 6 * 7?')
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/ai/agents/chat'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ message: 'What is 6 * 7?' }),
      })
    )
    expect(result.answer).toBe('42')
    expect(result.tools_used).toEqual(['calculator'])
  })

  it('agentChatStream yields tokens from SSE', async () => {
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(
          new TextEncoder().encode('data: {"token":"Hello"}\n\ndata: {"token":" world"}\n\ndata: {"done":true}\n\n')
        )
        controller.close()
      },
    })
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(new Response(stream, { status: 200 }))
    const chunks: Array<{ token?: string; done?: boolean }> = []
    for await (const c of agentChatStream('Hi')) {
      chunks.push(c)
    }
    expect(chunks).toEqual([{ token: 'Hello' }, { token: ' world' }, { done: true }])
  })
})

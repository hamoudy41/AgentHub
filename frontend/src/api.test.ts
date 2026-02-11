import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  ask,
  classify,
  createDocument,
  getDocument,
  getHealth,
  notarySummarize,
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
})

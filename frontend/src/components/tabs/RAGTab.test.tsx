import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { RAGTab } from './RAGTab'
import * as api from '../../api'

vi.mock('../../api', () => ({
  ragQuery: vi.fn(),
  ragIndex: vi.fn(),
}))

describe('RAGTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    cleanup()
  })

  it('queries RAG successfully', async () => {
    const user = userEvent.setup()
    vi.mocked(api.ragQuery).mockResolvedValue({
      answer: 'Paris is the capital.',
      sources: [{ text: 'France capital Paris', document_id: 'd1', score: 0.9 }],
      model: 'llama3.2',
    })
    render(<RAGTab />)
    await user.type(screen.getByPlaceholderText(/ask a question/i), 'What is the capital of France?')
    await user.click(screen.getByRole('button', { name: 'Get answer' }))
    await waitFor(() => {
      expect(api.ragQuery).toHaveBeenCalledWith('What is the capital of France?', expect.any(Object))
    })
    expect(screen.getByText(/Paris is the capital/)).toBeInTheDocument()
  })

  it('indexes document successfully', async () => {
    const user = userEvent.setup()
    vi.mocked(api.ragIndex).mockResolvedValue({
      document_id: 'doc1',
      chunks_indexed: 3,
      status: 'indexed',
    })
    render(<RAGTab />)
    await user.type(screen.getByPlaceholderText(/document id to index/i), 'doc1')
    await user.click(screen.getByRole('button', { name: 'Index' }))
    await waitFor(() => {
      expect(api.ragIndex).toHaveBeenCalledWith('doc1')
    })
    expect(screen.getByText(/Indexed 3 chunk/)).toBeInTheDocument()
  })

  it('shows error when index without document ID', async () => {
    const user = userEvent.setup()
    render(<RAGTab />)
    await user.click(screen.getByRole('button', { name: 'Index' }))
    await waitFor(() => {
      expect(screen.getByText(/Document ID is required to index/)).toBeInTheDocument()
    })
    expect(api.ragIndex).not.toHaveBeenCalled()
  })

  it('handles RAG query error', async () => {
    const user = userEvent.setup()
    vi.mocked(api.ragQuery).mockRejectedValue(new Error('RAG failed'))
    render(<RAGTab />)
    await user.type(screen.getByPlaceholderText(/ask a question/i), 'Query')
    await user.click(screen.getByRole('button', { name: 'Get answer' }))
    await waitFor(() => {
      expect(screen.getByText(/RAG failed/)).toBeInTheDocument()
    })
  })

  it('handles RAG index error', async () => {
    const user = userEvent.setup()
    vi.mocked(api.ragIndex).mockRejectedValue(new Error('Index failed'))
    render(<RAGTab />)
    await user.type(screen.getByPlaceholderText(/document id to index/i), 'doc1')
    await user.click(screen.getByRole('button', { name: 'Index' }))
    await waitFor(() => {
      expect(screen.getByText(/Index failed/)).toBeInTheDocument()
    })
  })
})

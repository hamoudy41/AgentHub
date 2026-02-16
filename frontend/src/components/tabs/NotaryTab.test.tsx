import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { NotaryTab } from './NotaryTab'
import * as api from '../../api'

vi.mock('../../api', () => ({
  notarySummarize: vi.fn(),
}))

describe('NotaryTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    cleanup()
  })

  it('summarizes document successfully', async () => {
    const user = userEvent.setup()
    vi.mocked(api.notarySummarize).mockResolvedValue({
      document_id: null,
      summary: {
        title: 'Summary',
        key_points: ['Point 1'],
        parties_involved: [],
        risks_or_warnings: [],
        raw_summary: 'Full summary',
      },
      source: 'llm',
    })
    render(<NotaryTab />)
    await user.type(screen.getByPlaceholderText(/document text/i), 'Deed content...')
    await user.click(screen.getByRole('button', { name: 'Summarize' }))
    await waitFor(() => {
      expect(api.notarySummarize).toHaveBeenCalledWith('Deed content...', { language: 'nl' })
    })
    expect(screen.getByText(/Point 1/)).toBeInTheDocument()
    expect(screen.getByText(/Full summary/)).toBeInTheDocument()
  })

  it('summarizes with document id and language', async () => {
    const user = userEvent.setup()
    vi.mocked(api.notarySummarize).mockResolvedValue({
      document_id: 'doc1',
      summary: {
        title: 'S',
        key_points: [],
        parties_involved: [],
        risks_or_warnings: [],
        raw_summary: 'x',
      },
      source: 'llm',
    })
    render(<NotaryTab />)
    await user.type(screen.getByPlaceholderText(/document text/i), 'Text')
    await user.type(screen.getByPlaceholderText(/document id \(optional\)/i), 'doc1')
    await user.selectOptions(screen.getByRole('combobox'), 'en')
    await user.click(screen.getByRole('button', { name: 'Summarize' }))
    await waitFor(() => {
      expect(api.notarySummarize).toHaveBeenCalledWith('Text', {
        documentId: 'doc1',
        language: 'en',
      })
    })
  })

  it('handles notary error', async () => {
    const user = userEvent.setup()
    vi.mocked(api.notarySummarize).mockRejectedValue(new Error('Summarize failed'))
    render(<NotaryTab />)
    await user.click(screen.getByRole('button', { name: 'Summarize' }))
    await waitFor(() => {
      expect(screen.getByText(/Summarize failed/)).toBeInTheDocument()
    })
  })
})

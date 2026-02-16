import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ClassifyTab } from './ClassifyTab'
import * as api from '../../api'

vi.mock('../../api', () => ({
  classify: vi.fn(),
}))

describe('ClassifyTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    cleanup()
  })

  it('classifies text successfully', async () => {
    const user = userEvent.setup()
    vi.mocked(api.classify).mockResolvedValue({
      label: 'invoice',
      confidence: 0.9,
      model: 'llm',
      source: 'llm',
    })
    render(<ClassifyTab />)
    await user.type(screen.getByPlaceholderText(/text to classify/i), 'Invoice for 100 EUR.')
    await user.click(screen.getByRole('button', { name: 'Classify' }))
    await waitFor(() => {
      expect(api.classify).toHaveBeenCalledWith('Invoice for 100 EUR.', [
        'contract',
        'letter',
        'invoice',
        'report',
        'other',
      ])
    })
    expect(screen.getByText('invoice', { exact: true })).toBeInTheDocument()
  })

  it('loads file and classifies successfully', async () => {
    const user = userEvent.setup()
    vi.mocked(api.classify).mockResolvedValue({
      label: 'contract',
      confidence: 0.85,
      model: 'llm',
      source: 'llm',
    })
    render(<ClassifyTab />)
    const file = new File(['This agreement is between Party A and Party B.'], 'agreement.txt', {
      type: 'text/plain',
    })
    await user.upload(screen.getByTestId('classify-file-upload'), file)
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/text to classify/i)).toHaveValue(
        'This agreement is between Party A and Party B.'
      )
    })
    await user.click(screen.getByRole('button', { name: 'Classify' }))
    await waitFor(() => {
      expect(api.classify).toHaveBeenCalledWith(
        'This agreement is between Party A and Party B.',
        expect.arrayContaining(['contract', 'letter', 'invoice', 'report', 'other'])
      )
    })
    expect(screen.getByText('contract', { exact: true })).toBeInTheDocument()
  })

  it('handles classify error', async () => {
    const user = userEvent.setup()
    vi.mocked(api.classify).mockRejectedValue(new Error('LLM error'))
    render(<ClassifyTab />)
    await user.click(screen.getByRole('button', { name: 'Classify' }))
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(/LLM error/)
    })
  })
})

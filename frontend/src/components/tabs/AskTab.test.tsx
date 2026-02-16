import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { AskTab } from './AskTab'
import * as api from '../../api'

vi.mock('../../api', () => ({
  ask: vi.fn(),
}))

describe('AskTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    cleanup()
  })

  it('asks question successfully', async () => {
    const user = userEvent.setup()
    vi.mocked(api.ask).mockResolvedValue({
      answer: 'The answer is 42',
      model: 'llm',
      source: 'llm',
    })
    render(<AskTab />)
    await user.type(screen.getByPlaceholderText(/context/i), 'Context here')
    await user.type(screen.getByPlaceholderText(/question/i), 'What is the answer?')
    await user.click(screen.getByRole('button', { name: 'Get answer' }))
    await waitFor(() => {
      expect(api.ask).toHaveBeenCalledWith('What is the answer?', 'Context here')
    })
    expect(screen.getByText(/The answer is 42/)).toBeInTheDocument()
  })

  it('handles ask error', async () => {
    const user = userEvent.setup()
    vi.mocked(api.ask).mockRejectedValue(new Error('Ask failed'))
    render(<AskTab />)
    await user.click(screen.getByRole('button', { name: 'Get answer' }))
    await waitFor(() => {
      expect(screen.getByText(/Ask failed/)).toBeInTheDocument()
    })
  })
})

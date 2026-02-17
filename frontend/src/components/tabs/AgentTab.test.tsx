import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { AgentTab } from './AgentTab'
import * as api from '../../api'

vi.mock('../../api', () => ({
  agentChat: vi.fn(),
  agentChatStream: vi.fn(),
}))

describe('AgentTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    cleanup()
  })

  it('renders agent playground heading and input', () => {
    render(<AgentTab />)
    expect(screen.getByRole('heading', { name: /agent playground/i })).toBeInTheDocument()
    expect(screen.getByPlaceholderText(/message/i)).toBeInTheDocument()
  })

  it('sends message and displays answer', async () => {
    const user = userEvent.setup()
    vi.mocked(api.agentChat).mockResolvedValue({
      answer: 'The result is 42',
      tools_used: ['calculator'],
    })
    render(<AgentTab />)
    await user.type(screen.getByPlaceholderText(/message/i), 'What is 6 * 7?')
    await user.click(screen.getByRole('button', { name: /send/i }))
    await waitFor(() => {
      expect(api.agentChat).toHaveBeenCalledWith('What is 6 * 7?')
    })
    expect(screen.getByText(/The result is 42/)).toBeInTheDocument()
  })

  it('displays tools used when present', async () => {
    const user = userEvent.setup()
    vi.mocked(api.agentChat).mockResolvedValue({
      answer: '42',
      tools_used: ['calculator', 'search'],
    })
    render(<AgentTab />)
    await user.type(screen.getByPlaceholderText(/message/i), 'Calculate 6*7')
    await user.click(screen.getByRole('button', { name: /send/i }))
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /tools used/i })).toBeInTheDocument()
    })
    const toolsSection = screen.getByRole('heading', { name: /tools used/i }).parentElement
    expect(toolsSection).toHaveTextContent('calculator')
    expect(toolsSection).toHaveTextContent('search')
  })

  it('handles agent error', async () => {
    const user = userEvent.setup()
    vi.mocked(api.agentChat).mockRejectedValue(new Error('Agent failed'))
    render(<AgentTab />)
    await user.type(screen.getByPlaceholderText(/message/i), 'Hello')
    await user.click(screen.getByRole('button', { name: /send/i }))
    await waitFor(() => {
      expect(screen.getByText(/Agent failed/)).toBeInTheDocument()
    })
  })

  it('disables send when message is empty', () => {
    render(<AgentTab />)
    expect(screen.getByRole('button', { name: /send/i })).toBeDisabled()
  })
})

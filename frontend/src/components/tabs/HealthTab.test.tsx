import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { HealthTab } from './HealthTab'
import * as api from '../../api'

vi.mock('../../api', () => ({
  getHealth: vi.fn(),
}))

describe('HealthTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    cleanup()
  })

  it('renders heading and check health button', () => {
    render(<HealthTab />)
    expect(screen.getByRole('heading', { name: /health/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /check health/i })).toBeInTheDocument()
  })

  it('fetches health on button click and displays result', async () => {
    const user = userEvent.setup()
    vi.mocked(api.getHealth).mockResolvedValue({
      environment: 'local',
      timestamp: '2024-01-01',
      db_ok: true,
      llm_ok: true,
    })
    render(<HealthTab />)
    await user.click(screen.getByRole('button', { name: /check health/i }))
    await waitFor(() => {
      expect(api.getHealth).toHaveBeenCalled()
    })
    expect(screen.getByText(/Health status/)).toBeInTheDocument()
    expect(screen.getByText(/local/)).toBeInTheDocument()
    expect(screen.getByText(/Connected/)).toBeInTheDocument()
    expect(screen.getByText(/Configured/)).toBeInTheDocument()
  })

  it('handles health fetch error', async () => {
    const user = userEvent.setup()
    vi.mocked(api.getHealth).mockRejectedValue(new Error('Network error'))
    render(<HealthTab />)
    await user.click(screen.getByRole('button', { name: /check health/i }))
    await waitFor(() => {
      expect(screen.getByText(/error/)).toBeInTheDocument()
    })
  })
})

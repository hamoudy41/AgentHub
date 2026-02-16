import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import App from './App'

describe('App', () => {
  beforeEach(() => {
    cleanup()
  })

  it('renders and shows health tab by default', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: /health/i })).toBeInTheDocument()
  })

  it('switches tabs and renders correct content', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByTestId('tab-documents'))
    expect(screen.getByRole('heading', { name: /documents/i })).toBeInTheDocument()

    await user.click(screen.getByTestId('tab-classify'))
    expect(screen.getByRole('heading', { name: /classify/i })).toBeInTheDocument()

    await user.click(screen.getByTestId('tab-notary'))
    expect(screen.getByRole('heading', { name: /notary summarize/i })).toBeInTheDocument()

    await user.click(screen.getByTestId('tab-ask'))
    expect(screen.getByRole('heading', { name: /^ask$/i })).toBeInTheDocument()

    await user.click(screen.getByTestId('tab-rag'))
    expect(screen.getByRole('heading', { name: /rag/i })).toBeInTheDocument()

    await user.click(screen.getByTestId('tab-health'))
    expect(screen.getByRole('heading', { name: /health/i })).toBeInTheDocument()
  })
})

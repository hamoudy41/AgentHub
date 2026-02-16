import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Layout } from './Layout'

describe('Layout', () => {
  afterEach(cleanup)
  it('renders children and tab navigation', () => {
    render(
      <Layout tab="health" onTabChange={() => {}}>
        <div>Health content</div>
      </Layout>
    )
    expect(screen.getByText('Health content')).toBeInTheDocument()
    expect(screen.getByTestId('tab-health')).toBeInTheDocument()
    expect(screen.getByTestId('tab-documents')).toBeInTheDocument()
    expect(screen.getByTestId('tab-rag')).toBeInTheDocument()
  })

  it('calls onTabChange when tab is clicked', async () => {
    const user = userEvent.setup()
    const onTabChange = vi.fn()
    render(
      <Layout tab="health" onTabChange={onTabChange}>
        <div>Content</div>
      </Layout>
    )
    const documentsTabs = screen.getAllByTestId('tab-documents')
    await user.click(documentsTabs[0])
    expect(onTabChange).toHaveBeenCalledWith('documents')
  })
})

import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { DocumentsTab } from './DocumentsTab'
import * as api from '../../api'

vi.mock('../../api', () => ({
  createDocument: vi.fn(),
  getDocument: vi.fn(),
}))

describe('DocumentsTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    cleanup()
  })

  it('creates document successfully', async () => {
    const user = userEvent.setup()
    vi.mocked(api.createDocument).mockResolvedValue({
      id: 'd1',
      title: 'Title',
      text: 'Content',
      created_at: '2024-01-01',
    })
    render(<DocumentsTab />)
    await user.type(screen.getByPlaceholderText(/document id/i), 'd1')
    await user.type(screen.getByPlaceholderText(/title/i), 'Title')
    await user.type(screen.getByPlaceholderText(/text/i), 'Content')
    await user.click(screen.getByRole('button', { name: 'Create' }))
    await waitFor(() => {
      expect(api.createDocument).toHaveBeenCalledWith({ id: 'd1', title: 'Title', text: 'Content' })
    })
    expect(screen.getByText(/Title/)).toBeInTheDocument()
    expect(screen.getByText(/ID: d1/)).toBeInTheDocument()
  })

  it('shows error when create without document ID', async () => {
    const user = userEvent.setup()
    render(<DocumentsTab />)
    await user.click(screen.getByRole('button', { name: 'Create' }))
    await waitFor(() => {
      expect(screen.getByText(/Document ID is required/)).toBeInTheDocument()
    })
    expect(api.createDocument).not.toHaveBeenCalled()
  })

  it('shows error when create fails', async () => {
    const user = userEvent.setup()
    vi.mocked(api.createDocument).mockRejectedValue(new Error('Create failed'))
    render(<DocumentsTab />)
    await user.type(screen.getByPlaceholderText(/document id/i), 'd1')
    await user.click(screen.getByRole('button', { name: 'Create' }))
    await waitFor(() => {
      expect(screen.getByText(/Create failed/)).toBeInTheDocument()
    })
  })

  it('shows error when document already exists', async () => {
    const user = userEvent.setup()
    vi.mocked(api.createDocument).mockRejectedValue(
      new Error("Document with ID 'dup' already exists. Use Get by ID to view, or choose a different ID.")
    )
    render(<DocumentsTab />)
    await user.type(screen.getByPlaceholderText(/document id/i), 'dup')
    await user.click(screen.getByRole('button', { name: 'Create' }))
    await waitFor(() => {
      expect(screen.getByText(/already exists/)).toBeInTheDocument()
    })
  })

  it('gets document by id', async () => {
    const user = userEvent.setup()
    vi.mocked(api.getDocument).mockResolvedValue({
      id: 'd1',
      title: 'T',
      text: 'C',
      created_at: 'x',
    })
    render(<DocumentsTab />)
    await user.type(screen.getByPlaceholderText(/document id/i), 'd1')
    await user.click(screen.getByRole('button', { name: 'Get by ID' }))
    await waitFor(() => {
      expect(api.getDocument).toHaveBeenCalledWith('d1')
    })
    expect(screen.getByText(/ID: d1/)).toBeInTheDocument()
  })

  it('shows error when get fails', async () => {
    const user = userEvent.setup()
    vi.mocked(api.getDocument).mockRejectedValue(new Error('Not found'))
    render(<DocumentsTab />)
    await user.type(screen.getByPlaceholderText(/document id/i), 'missing')
    await user.click(screen.getByRole('button', { name: 'Get by ID' }))
    await waitFor(() => {
      expect(screen.getByText(/Not found/)).toBeInTheDocument()
    })
  })

  it('loads file into form and create saves document', async () => {
    const user = userEvent.setup()
    vi.mocked(api.createDocument).mockResolvedValue({
      id: 'mydoc',
      title: 'mydoc.txt',
      text: 'Content from file',
      created_at: '2024-01-01',
    })
    render(<DocumentsTab />)
    const file = new File(['Content from file'], 'mydoc.txt', { type: 'text/plain' })
    await user.upload(screen.getByTestId('file-upload'), file)
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/text/i)).toHaveValue('Content from file')
    })
    await user.click(screen.getByRole('button', { name: 'Create' }))
    await waitFor(() => {
      expect(api.createDocument).toHaveBeenCalledWith({
        id: 'mydoc',
        title: 'mydoc.txt',
        text: 'Content from file',
      })
    })
    expect(screen.getByText(/mydoc\.txt/)).toBeInTheDocument()
    expect(screen.getByText(/ID: mydoc/)).toBeInTheDocument()
  })

  it('shows error when create fails after loading file', async () => {
    const user = userEvent.setup()
    vi.mocked(api.createDocument).mockRejectedValue(new Error('Create failed'))
    render(<DocumentsTab />)
    const file = new File(['x'], 'f.txt')
    await user.upload(screen.getByTestId('file-upload'), file)
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/text/i)).toHaveValue('x')
    })
    await user.click(screen.getByRole('button', { name: 'Create' }))
    await waitFor(() => {
      expect(screen.getByText(/Create failed/)).toBeInTheDocument()
    })
  })

  it('shows error for file too large', async () => {
    const user = userEvent.setup()
    render(<DocumentsTab />)
    const oversized = new File(
      [new Blob(['x'.repeat(1)], { type: 'text/plain' })],
      'big.txt',
      { type: 'text/plain' }
    )
    Object.defineProperty(oversized, 'size', { value: 5 * 1024 * 1024 + 1 })
    await user.upload(screen.getByTestId('file-upload'), oversized)
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(/too large/)
    })
  })

  it('shows error when create fails with duplicate (after loading file)', async () => {
    const user = userEvent.setup()
    vi.mocked(api.createDocument).mockRejectedValue(
      new Error("Document with ID 'mydoc' already exists. Use a different ID or Get by ID to view.")
    )
    render(<DocumentsTab />)
    const file = new File(['content'], 'mydoc.txt')
    await user.upload(screen.getByTestId('file-upload'), file)
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/text/i)).toHaveValue('content')
    })
    await user.click(screen.getByRole('button', { name: 'Create' }))
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(/already exists/)
    })
  })

  it('shows duplicate error with document ID in message', async () => {
    const user = userEvent.setup()
    vi.mocked(api.createDocument).mockRejectedValue(
      new Error("Document with ID 'report' already exists. Use Get by ID to view, or choose a different ID.")
    )
    render(<DocumentsTab />)
    await user.type(screen.getByPlaceholderText(/document id/i), 'report')
    await user.type(screen.getByPlaceholderText(/title/i), 'Report')
    await user.type(screen.getByPlaceholderText(/text/i), 'Content')
    await user.click(screen.getByRole('button', { name: 'Create' }))
    await waitFor(() => {
      const alert = screen.getByRole('alert')
      expect(alert).toHaveTextContent(/already exists/)
      expect(alert).toHaveTextContent(/report/)
    })
  })

  it('allows retry after duplicate: change ID and create succeeds', async () => {
    const user = userEvent.setup()
    vi.mocked(api.createDocument)
      .mockRejectedValueOnce(
        new Error("Document with ID 'dup' already exists. Use Get by ID to view, or choose a different ID.")
      )
      .mockResolvedValueOnce({
        id: 'dup-v2',
        title: 'Report',
        text: 'Content',
        created_at: '2024-01-01',
      })
    render(<DocumentsTab />)
    await user.type(screen.getByPlaceholderText(/document id/i), 'dup')
    await user.type(screen.getByPlaceholderText(/title/i), 'Report')
    await user.type(screen.getByPlaceholderText(/text/i), 'Content')
    await user.click(screen.getByRole('button', { name: 'Create' }))
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(/already exists/)
    })
    await user.clear(screen.getByPlaceholderText(/document id/i))
    await user.type(screen.getByPlaceholderText(/document id/i), 'dup-v2')
    await user.click(screen.getByRole('button', { name: 'Create' }))
    await waitFor(() => {
      expect(screen.getByText(/ID: dup-v2/)).toBeInTheDocument()
    })
  })

  it('dismisses error alert', async () => {
    const user = userEvent.setup()
    vi.mocked(api.createDocument).mockRejectedValue(new Error('Create failed'))
    render(<DocumentsTab />)
    await user.type(screen.getByPlaceholderText(/document id/i), 'd1')
    await user.click(screen.getByRole('button', { name: 'Create' }))
    await waitFor(() => expect(screen.getByText(/Create failed/)).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: 'Dismiss' }))
    await waitFor(() => {
      expect(screen.queryByText(/Create failed/)).not.toBeInTheDocument()
    })
  })
})

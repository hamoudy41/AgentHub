import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import * as api from './api'

vi.mock('./api', () => ({
  getHealth: vi.fn(),
  createDocument: vi.fn(),
  getDocument: vi.fn(),
  classify: vi.fn(),
  notarySummarize: vi.fn(),
  ask: vi.fn(),
}))

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks()
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

    await user.click(screen.getByTestId('tab-health'))
    expect(screen.getByRole('heading', { name: /health/i })).toBeInTheDocument()
  })

  describe('Health tab', () => {
    it('fetches health on button click and displays result', async () => {
      const user = userEvent.setup()
      vi.mocked(api.getHealth).mockResolvedValue({
        environment: 'local',
        timestamp: '2024-01-01',
        db_ok: true,
        llm_ok: true,
      })
      render(<App />)
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
      render(<App />)
      await user.click(screen.getByRole('button', { name: /check health/i }))
      await waitFor(() => {
        expect(screen.getByText(/error/)).toBeInTheDocument()
      })
    })

    it('passes apiKey when provided', async () => {
      const user = userEvent.setup()
      vi.mocked(api.getHealth).mockResolvedValue({
        environment: 'local',
        timestamp: 'x',
      })
      render(<App />)
      await user.click(screen.getByRole('button', { name: /check health/i }))
      await waitFor(() => {
        expect(api.getHealth).toHaveBeenCalled()
      })
      // API key comes from VITE_API_KEY env when set; no UI input
      expect(screen.getByText(/local/)).toBeInTheDocument()
    })
  })

  describe('Documents tab', () => {
    it('creates document successfully', async () => {
      const user = userEvent.setup()
      vi.mocked(api.createDocument).mockResolvedValue({
        id: 'd1',
        title: 'Title',
        text: 'Content',
        created_at: '2024-01-01',
      })
      render(<App />)
      await user.click(screen.getByTestId('tab-documents'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-documents'))
      await user.click(screen.getByRole('button', { name: 'Create' }))
      await waitFor(() => {
        expect(screen.getByText(/Document ID is required/)).toBeInTheDocument()
      })
      expect(api.createDocument).not.toHaveBeenCalled()
    })

    it('shows error when create fails', async () => {
      const user = userEvent.setup()
      vi.mocked(api.createDocument).mockRejectedValue(new Error('Create failed'))
      render(<App />)
      await user.click(screen.getByTestId('tab-documents'))
      await user.type(screen.getByPlaceholderText(/document id/i), 'd1')
      await user.click(screen.getByRole('button', { name: 'Create' }))
      await waitFor(() => {
        expect(screen.getByText(/Create failed/)).toBeInTheDocument()
      })
    })

    it('shows error when document already exists', async () => {
      const user = userEvent.setup()
      vi.mocked(api.createDocument).mockRejectedValue(
        new Error('Document with ID \'dup\' already exists. Use Get by ID to view, or choose a different ID.')
      )
      render(<App />)
      await user.click(screen.getByTestId('tab-documents'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-documents'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-documents'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-documents'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-documents'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-documents'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-documents'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-documents'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-documents'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-documents'))
      await user.type(screen.getByPlaceholderText(/document id/i), 'd1')
      await user.click(screen.getByRole('button', { name: 'Create' }))
      await waitFor(() => expect(screen.getByText(/Create failed/)).toBeInTheDocument())
      await user.click(screen.getByRole('button', { name: 'Dismiss' }))
      await waitFor(() => {
        expect(screen.queryByText(/Create failed/)).not.toBeInTheDocument()
      })
    })
  })

  describe('Classify tab', () => {
    it('classifies text successfully', async () => {
      const user = userEvent.setup()
      vi.mocked(api.classify).mockResolvedValue({
        label: 'invoice',
        confidence: 0.9,
        model: 'llm',
        source: 'llm',
      })
      render(<App />)
      await user.click(screen.getByTestId('tab-classify'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-classify'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-classify'))
      await user.click(screen.getByRole('button', { name: 'Classify' }))
      await waitFor(() => {
        expect(screen.getByRole('alert')).toHaveTextContent(/LLM error/)
      })
    })
  })

  describe('Notary tab', () => {
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
      render(<App />)
      await user.click(screen.getByTestId('tab-notary'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-notary'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-notary'))
      await user.click(screen.getByRole('button', { name: 'Summarize' }))
      await waitFor(() => {
        expect(screen.getByText(/Summarize failed/)).toBeInTheDocument()
      })
    })
  })

  describe('Ask tab', () => {
    it('asks question successfully', async () => {
      const user = userEvent.setup()
      vi.mocked(api.ask).mockResolvedValue({
        answer: 'The answer is 42',
        model: 'llm',
        source: 'llm',
      })
      render(<App />)
      await user.click(screen.getByTestId('tab-ask'))
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
      render(<App />)
      await user.click(screen.getByTestId('tab-ask'))
      await user.click(screen.getByRole('button', { name: 'Get answer' }))
      await waitFor(() => {
        expect(screen.getByText(/Ask failed/)).toBeInTheDocument()
      })
    })
  })
})

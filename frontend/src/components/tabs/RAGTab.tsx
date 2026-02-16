import { useState } from 'react'
import { ragQuery, ragIndex } from '../../api'
import type { RAGQueryResponse, RAGIndexResponse } from '../../api'
import { Alert } from '../Alert'

export function RAGTab() {
  const [query, setQuery] = useState('')
  const [documentId, setDocumentId] = useState('')
  const [result, setResult] = useState<RAGQueryResponse | null>(null)
  const [indexResult, setIndexResult] = useState<RAGIndexResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [indexing, setIndexing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleQuery = async () => {
    setLoading(true)
    setResult(null)
    setError(null)
    try {
      const docIds = documentId.trim() ? documentId.split(/[,\s]+/).filter(Boolean) : undefined
      const r = await ragQuery(query, { documentIds: docIds })
      setResult(r)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const handleIndex = async () => {
    const id = documentId.trim()
    if (!id) {
      setError('Document ID is required to index')
      return
    }
    setIndexing(true)
    setIndexResult(null)
    setError(null)
    try {
      const r = await ragIndex(id)
      setIndexResult(r)
    } catch (e) {
      setError(String(e))
    } finally {
      setIndexing(false)
    }
  }

  return (
    <section>
      <h2 className="mb-4 text-xl font-medium text-slate-800 dark:text-slate-200">RAG (Retrieval-Augmented Generation)</h2>
      <p className="mb-4 text-sm text-slate-600 dark:text-slate-400">
        Index documents for semantic search, then ask questions. Answers are based on retrieved context from your documents.
      </p>

      <div className="mb-6 space-y-4">
        <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">Index document</h3>
        <div className="flex flex-wrap gap-2">
          <input
            placeholder="Document ID to index"
            value={documentId}
            onChange={(e) => setDocumentId(e.target.value)}
            className="rounded border border-slate-300 px-3 py-2 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
          />
          <button
            type="button"
            onClick={handleIndex}
            disabled={indexing}
            className="rounded-md bg-slate-600 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
          >
            {indexing ? 'Indexing…' : 'Index'}
          </button>
        </div>
        {indexResult && (
          <p className="text-sm text-emerald-600 dark:text-emerald-400">
            Indexed {indexResult.chunks_indexed} chunk(s) for document &quot;{indexResult.document_id}&quot;
          </p>
        )}
      </div>

      <div className="space-y-4">
        <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">Query</h3>
        <div className="flex flex-col gap-3">
          <input
            placeholder="Document IDs (comma-separated, optional)"
            value={documentId}
            onChange={(e) => setDocumentId(e.target.value)}
            className="rounded border border-slate-300 px-3 py-2 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
          />
          <textarea
            placeholder="Ask a question"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={2}
            className="rounded border border-slate-300 px-3 py-2 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
          />
          <button
            type="button"
            onClick={handleQuery}
            disabled={loading}
            className="w-fit rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 disabled:opacity-50"
          >
            {loading ? 'Querying…' : 'Get answer'}
          </button>
        </div>
      </div>

      {error && (
        <Alert variant="error" onDismiss={() => setError(null)} className="mt-4">
          {error}
        </Alert>
      )}

      {result && !error && (
        <div className="mt-4 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-800/50">
          <div className="border-b border-slate-200 bg-slate-50 px-4 py-3 dark:border-slate-700 dark:bg-slate-800/80">
            <h3 className="font-medium text-slate-800 dark:text-slate-200">Answer</h3>
          </div>
          <div className="px-4 py-4">
            <p className="whitespace-pre-wrap text-slate-700 dark:text-slate-300">{result.answer}</p>
          </div>
          {result.sources?.length > 0 && (
            <div className="border-t border-slate-200 px-4 py-3 dark:border-slate-700">
              <h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">Sources</h4>
              <ul className="space-y-2 text-sm">
                {result.sources.map((s, i) => (
                  <li key={i} className="rounded bg-slate-100 px-2 py-1 dark:bg-slate-700 dark:text-slate-300">
                    <span className="font-medium text-slate-600 dark:text-slate-400">{s.document_id}</span>
                    <span className="ml-2 text-slate-500">(score: {s.score.toFixed(2)})</span>
                    <p className="mt-1 truncate text-slate-600 dark:text-slate-400">{s.text}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div className="border-t border-slate-200 px-4 py-2 text-xs text-slate-500 dark:border-slate-700 dark:text-slate-400">
            Model: {result.model}
          </div>
        </div>
      )}
    </section>
  )
}

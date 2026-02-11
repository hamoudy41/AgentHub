import { useState } from 'react'
import { ask } from '../../api'
import type { AskResponse } from '../../api'
import { Alert } from '../Alert'
import { AskResult } from '../ResultPreview'

export function AskTab() {
  const [question, setQuestion] = useState('')
  const [context, setContext] = useState('')
  const [result, setResult] = useState<AskResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleAsk = async () => {
    setLoading(true)
    setResult(null)
    setError(null)
    try {
      const r = await ask(question, context)
      setResult(r)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section>
      <h2 className="mb-4 text-xl font-medium text-slate-800 dark:text-slate-200">Ask</h2>
      <p className="mb-4 text-sm text-slate-600 dark:text-slate-400">
        Ask a question based on the provided context.
      </p>
      <div className="flex flex-col gap-3">
        <textarea
          placeholder="Context"
          value={context}
          onChange={(e) => setContext(e.target.value)}
          rows={4}
          className="rounded border border-slate-300 px-3 py-2 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
        />
        <input
          placeholder="Question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          className="rounded border border-slate-300 px-3 py-2 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
        />
        <button
          type="button"
          onClick={handleAsk}
          disabled={loading}
          className="w-fit rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 disabled:opacity-50"
        >
          {loading ? 'Asking…' : 'Get answer'}
        </button>
      </div>
      {error && (
        <Alert variant="error" onDismiss={() => setError(null)} className="mt-4">
          {error}
        </Alert>
      )}
      {result && !error && <AskResult data={result} />}
    </section>
  )
}

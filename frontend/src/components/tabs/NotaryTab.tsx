import { useState } from 'react'
import { notarySummarize } from '../../api'
import type { NotarySummarizeResponse } from '../../api'
import { Alert } from '../Alert'
import { NotaryResult } from '../ResultPreview'

export function NotaryTab() {
  const [text, setText] = useState('')
  const [docId, setDocId] = useState('')
  const [lang, setLang] = useState<'nl' | 'en'>('nl')
  const [result, setResult] = useState<NotarySummarizeResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSummarize = async () => {
    setLoading(true)
    setResult(null)
    setError(null)
    try {
      const opts: { documentId?: string; language?: 'nl' | 'en' } = {}
      if (docId.trim()) opts.documentId = docId.trim()
      opts.language = lang
      const r = await notarySummarize(text, opts)
      setResult(r)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section>
      <h2 className="mb-4 text-xl font-medium text-slate-800 dark:text-slate-200">
        Notary Summarize
      </h2>
      <p className="mb-4 text-sm text-slate-600 dark:text-slate-400">
        Summarize a document for notarial context.
      </p>
      <div className="flex flex-col gap-3">
        <textarea
          placeholder="Document text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          className="rounded border border-slate-300 px-3 py-2 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
        />
        <input
          placeholder="Document ID (optional)"
          value={docId}
          onChange={(e) => setDocId(e.target.value)}
          className="rounded border border-slate-300 px-3 py-2 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
        />
        <select
          value={lang}
          onChange={(e) => setLang(e.target.value as 'nl' | 'en')}
          className="rounded border border-slate-300 px-3 py-2 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
        >
          <option value="nl">Dutch</option>
          <option value="en">English</option>
        </select>
        <button
          type="button"
          onClick={handleSummarize}
          disabled={loading}
          className="w-fit rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 disabled:opacity-50"
        >
          {loading ? 'Summarizing…' : 'Summarize'}
        </button>
      </div>
      {error && (
        <Alert variant="error" onDismiss={() => setError(null)} className="mt-4">
          {error}
        </Alert>
      )}
      {result && !error && <NotaryResult data={result} />}
    </section>
  )
}

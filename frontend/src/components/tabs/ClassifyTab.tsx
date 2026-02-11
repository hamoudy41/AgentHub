import { useState } from 'react'
import { classify } from '../../api'
import type { ClassifyResponse } from '../../api'
import { Alert } from '../Alert'
import { ClassifyResult } from '../ResultPreview'
import { FileLoadButton } from '../FileLoadButton'

const DEFAULT_LABELS = 'contract, letter, invoice, report, other'

export function ClassifyTab() {
  const [text, setText] = useState('')
  const [labels, setLabels] = useState(DEFAULT_LABELS)
  const [result, setResult] = useState<ClassifyResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleClassify = async () => {
    setLoading(true)
    setResult(null)
    setError(null)
    try {
      const labelList = labels.split(',').map((s) => s.trim()).filter(Boolean)
      const r = await classify(text, labelList)
      setResult(r)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section>
      <h2 className="mb-4 text-xl font-medium text-slate-800 dark:text-slate-200">Classify</h2>
      <p className="mb-4 text-sm text-slate-600 dark:text-slate-400">
        Classify text by document type. Default: contract, letter, invoice, report, other. Edit labels to customize.
      </p>
      <div className="flex flex-col gap-3">
        <textarea
          placeholder="Text to classify (or load a file below)"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          className="rounded border border-slate-300 px-3 py-2 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
        />
        <FileLoadButton
          onTextLoaded={(text) => setText(text)}
          onError={(msg) => setError(msg)}
          onClearResult={() => setResult(null)}
          hint=".txt, .md, .json, .csv, .xml, .html (max 5 MB). Then click Classify."
          testId="classify-file-upload"
        />
        <input
          placeholder="Labels (comma-separated)"
          value={labels}
          onChange={(e) => setLabels(e.target.value)}
          className="rounded border border-slate-300 px-3 py-2 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
        />
        <button
          type="button"
          onClick={handleClassify}
          disabled={loading}
          className="w-fit rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 disabled:opacity-50"
        >
          {loading ? 'Classifying…' : 'Classify'}
        </button>
      </div>
      {error && (
        <Alert variant="error" onDismiss={() => setError(null)} className="mt-4">
          {error}
        </Alert>
      )}
      {result && !error && <ClassifyResult data={result} />}
    </section>
  )
}

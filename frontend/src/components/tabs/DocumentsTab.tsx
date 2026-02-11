import { useState } from 'react'
import { createDocument, getDocument } from '../../api'
import type { DocumentRead } from '../../api'
import { Alert } from '../Alert'
import { DocumentResult } from '../ResultPreview'
import { FileLoadButton } from '../FileLoadButton'

export function DocumentsTab() {
  const [docId, setDocId] = useState('')
  const [docTitle, setDocTitle] = useState('')
  const [docText, setDocText] = useState('')
  const [doc, setDoc] = useState<DocumentRead | null>(null)
  const [error, setError] = useState('')

  const handleCreate = async () => {
    setError('')
    setDoc(null)
    const id = docId.trim()
    if (!id) {
      setError('Document ID is required')
      return
    }
    try {
      const created = await createDocument({ id, title: docTitle, text: docText })
      setDoc(created)
    } catch (e) {
      setError(String(e))
    }
  }

  const handleGet = async () => {
    setError('')
    setDoc(null)
    const id = docId.trim()
    if (!id) {
      setError('Document ID is required')
      return
    }
    try {
      const d = await getDocument(id)
      setDoc(d)
    } catch (e) {
      setError(String(e))
    }
  }

  const handleFileLoaded = (text: string, file: File) => {
    const stem = file.name.replace(/\.[^/.]+$/, '') || 'document'
    if (!docId.trim()) setDocId(stem)
    if (!docTitle.trim()) setDocTitle(file.name)
    setDocText(text)
  }

  return (
    <section>
      <h2 className="mb-4 text-xl font-medium text-slate-800 dark:text-slate-200">Documents</h2>
      <div className="flex flex-col gap-3">
        <input
          placeholder="Document ID"
          value={docId}
          onChange={(e) => setDocId(e.target.value)}
          className="rounded border border-slate-300 px-3 py-2 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
        />
        <input
          placeholder="Title"
          value={docTitle}
          onChange={(e) => setDocTitle(e.target.value)}
          className="rounded border border-slate-300 px-3 py-2 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
        />
        <textarea
          placeholder="Text (or upload a file below)"
          value={docText}
          onChange={(e) => setDocText(e.target.value)}
          rows={4}
          className="rounded border border-slate-300 px-3 py-2 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
        />
        <FileLoadButton
          onTextLoaded={handleFileLoaded}
          onError={setError}
          onClearResult={() => setDoc(null)}
          hint=".txt, .md, .json, .csv, .xml, .html (max 5 MB). Then click Create to save."
          testId="file-upload"
        />
        <div className="flex gap-2">
          <button
            type="button"
            onClick={handleCreate}
            className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
          >
            Create
          </button>
          <button
            type="button"
            onClick={handleGet}
            className="rounded-md bg-slate-600 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700"
          >
            Get by ID
          </button>
        </div>
      </div>
      {error && (
        <Alert variant="error" onDismiss={() => setError('')} className="mt-2">
          {error}
        </Alert>
      )}
      {doc && <DocumentResult data={doc} />}
    </section>
  )
}

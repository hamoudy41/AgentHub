import { useState } from 'react'
import { agentChat } from '../../api'
import type { AgentChatResponse } from '../../api'
import { Alert } from '../Alert'

export function AgentTab() {
  const [message, setMessage] = useState('')
  const [result, setResult] = useState<AgentChatResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSend = async () => {
    const msg = message.trim()
    if (!msg) return
    setLoading(true)
    setResult(null)
    setError(null)
    try {
      const r = await agentChat(msg)
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
        Agent Playground
      </h2>
      <p className="mb-4 text-sm text-slate-600 dark:text-slate-400">
        Chat with the ReAct agent. It can use tools: calculator, search, and document lookup.
      </p>

      <div className="flex flex-col gap-3">
        <textarea
          placeholder="Message (e.g. What is 6 * 7?)"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          rows={2}
          className="rounded border border-slate-300 px-3 py-2 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
        />
        <button
          type="button"
          onClick={handleSend}
          disabled={loading || !message.trim()}
          className="w-fit rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 disabled:opacity-50"
        >
          {loading ? 'Sending…' : 'Send'}
        </button>
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
            <p className="whitespace-pre-wrap text-slate-700 dark:text-slate-300">
              {result.answer}
            </p>
          </div>
          {result.tools_used?.length > 0 && (
            <div className="border-t border-slate-200 px-4 py-3 dark:border-slate-700">
              <h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
                Tools used
              </h4>
              <div className="flex flex-wrap gap-2">
                {result.tools_used.map((tool) => (
                  <span
                    key={tool}
                    className="rounded bg-indigo-100 px-2 py-1 text-sm text-indigo-800 dark:bg-indigo-900/50 dark:text-indigo-200"
                  >
                    {tool}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  )
}

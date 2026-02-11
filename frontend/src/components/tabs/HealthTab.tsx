import { useState } from 'react'
import { getHealth } from '../../api'
import type { HealthStatus } from '../../api'
import { Alert } from '../Alert'
import { HealthResult } from '../ResultPreview'

export function HealthTab() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchHealth = async () => {
    setLoading(true)
    setHealth(null)
    setError(null)
    try {
      const h = await getHealth()
      setHealth(h)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section>
      <h2 className="mb-4 text-xl font-medium text-slate-800 dark:text-slate-200">Health</h2>
      <button
        type="button"
        onClick={fetchHealth}
        disabled={loading}
        className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 disabled:opacity-50"
      >
        {loading ? 'Loading…' : 'Check health'}
      </button>
      {error && (
        <Alert variant="error" onDismiss={() => setError(null)} className="mt-4">
          {error}
        </Alert>
      )}
      {health && !error && <HealthResult data={health} />}
    </section>
  )
}

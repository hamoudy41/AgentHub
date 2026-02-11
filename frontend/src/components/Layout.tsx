import type { ReactNode } from 'react'
import { TABS, type TabId } from './tabConfig'

export type { TabId }

interface LayoutProps {
  tab: TabId
  onTabChange: (tab: TabId) => void
  children: ReactNode
}

export function Layout({ tab, onTabChange, children }: LayoutProps) {
  return (
    <div className="mx-auto max-w-3xl px-6 py-8">
      <header className="mb-8">
        <h1 className="mb-4 text-2xl font-semibold text-slate-900 dark:text-slate-100">
          AI Platform
        </h1>
        <nav className="mb-4 flex flex-wrap gap-2">
          {TABS.map((t) => (
            <button
              key={t}
              data-testid={`tab-${t}`}
              type="button"
              onClick={() => onTabChange(t)}
              className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                tab === t
                  ? 'bg-indigo-600 text-white'
                  : 'bg-slate-200 text-slate-700 hover:bg-slate-300 dark:bg-slate-700 dark:text-slate-200 dark:hover:bg-slate-600'
              }`}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>
      <main>{children}</main>
    </div>
  )
}

import type { ReactNode } from 'react'
import { TAB_DEFINITIONS, type TabId } from './tabConfig'

export type { TabId }

interface LayoutProps {
  tab: TabId
  onTabChange: (tab: TabId) => void
  children: ReactNode
}

export function Layout({ tab, onTabChange, children }: LayoutProps) {
  const activeTab = TAB_DEFINITIONS.find((definition) => definition.id === tab)

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-10 overflow-hidden rounded-[28px] border border-white/60 bg-white/75 px-6 py-6 shadow-[0_24px_80px_-40px_rgba(15,23,42,0.45)] backdrop-blur dark:border-slate-800/80 dark:bg-slate-950/70">
        <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-2xl">
            <p className="mb-2 text-xs font-semibold uppercase tracking-[0.28em] text-amber-700 dark:text-amber-300">
              AgentHub Platform
            </p>
            <h1 className="text-3xl font-semibold tracking-tight text-slate-950 dark:text-white">
              Multi-agent document intelligence control plane
            </h1>
            <p className="mt-3 text-sm leading-6 text-slate-600 dark:text-slate-300">
              This workspace is organized around platform capabilities, not demos. Each surface
              tests an operational slice of the stack: ingestion, retrieval, workflows, and the
              agent runtime.
            </p>
          </div>
          {activeTab && (
            <div className="max-w-sm rounded-2xl border border-amber-200/80 bg-amber-50/90 px-4 py-3 text-sm text-amber-950 dark:border-amber-400/20 dark:bg-amber-300/10 dark:text-amber-100">
              <div className="font-medium">{activeTab.label}</div>
              <div className="mt-1 leading-6">{activeTab.description}</div>
            </div>
          )}
        </div>

        <nav className="flex flex-wrap gap-2">
          {TAB_DEFINITIONS.map((tabDefinition) => (
            <button
              key={tabDefinition.id}
              data-testid={`tab-${tabDefinition.id}`}
              type="button"
              onClick={() => onTabChange(tabDefinition.id)}
              className={`rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                tab === tabDefinition.id
                  ? 'bg-slate-950 text-white dark:bg-white dark:text-slate-950'
                  : 'bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800'
              }`}
            >
              {tabDefinition.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="rounded-[28px] border border-white/60 bg-white/80 p-6 shadow-[0_20px_60px_-36px_rgba(15,23,42,0.35)] backdrop-blur dark:border-slate-800/80 dark:bg-slate-950/70">
        {children}
      </main>
    </div>
  )
}

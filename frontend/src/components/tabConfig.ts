export type TabId = 'health' | 'documents' | 'classify' | 'notary' | 'ask' | 'rag' | 'agents'

export interface TabDefinition {
  id: TabId
  label: string
  description: string
}

export const TAB_DEFINITIONS: TabDefinition[] = [
  {
    id: 'health',
    label: 'Ops',
    description: 'Validate platform health, tenant wiring, and control-plane readiness.',
  },
  {
    id: 'documents',
    label: 'Documents',
    description: 'Load tenant-scoped source material that agents and retrieval can actually use.',
  },
  {
    id: 'classify',
    label: 'Classify',
    description: 'Run structured document understanding flows against ad hoc text.',
  },
  {
    id: 'notary',
    label: 'Notary',
    description: 'Exercise the domain-specific summarization workflow for legal document intake.',
  },
  {
    id: 'ask',
    label: 'Ask',
    description: 'Probe grounded question answering against explicit context.',
  },
  {
    id: 'rag',
    label: 'Retrieval',
    description: 'Index documents, inspect semantic recall, and test the RAG answer path.',
  },
  {
    id: 'agents',
    label: 'Agents',
    description: 'Chat with the tool-using agent runtime and inspect the tools it calls.',
  },
]

export const TABS: TabId[] = TAB_DEFINITIONS.map(({ id }) => id)

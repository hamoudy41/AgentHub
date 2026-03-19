import type { ComponentType } from 'react'
import { useState } from 'react'
import {
  AgentTab,
  AskTab,
  ClassifyTab,
  DocumentsTab,
  HealthTab,
  Layout,
  NotaryTab,
  RAGTab,
} from './components'
import type { TabId } from './components'

const TAB_COMPONENTS: Record<TabId, ComponentType> = {
  health: HealthTab,
  documents: DocumentsTab,
  classify: ClassifyTab,
  notary: NotaryTab,
  ask: AskTab,
  rag: RAGTab,
  agents: AgentTab,
}

export default function App() {
  const [tab, setTab] = useState<TabId>('health')
  const ActiveTab = TAB_COMPONENTS[tab]

  return (
    <Layout tab={tab} onTabChange={setTab}>
      <ActiveTab />
    </Layout>
  )
}

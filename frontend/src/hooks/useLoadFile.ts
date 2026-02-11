import { useRef } from 'react'

const MAX_FILE_SIZE = 5 * 1024 * 1024
const FILE_ACCEPT = '.txt,.md,.json,.csv,.xml,.html,.htm,text/plain,text/markdown,application/json'

export function useLoadFile(options: {
  onTextLoaded: (text: string, file: File) => void
  onError: (message: string) => void
  onClearResult?: () => void
  maxSize?: number
  accept?: string
}) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const maxSize = options.maxSize ?? MAX_FILE_SIZE
  const accept = options.accept ?? FILE_ACCEPT

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    options.onError('')
    options.onClearResult?.()
    if (file.size > maxSize) {
      options.onError(`File too large (max ${maxSize / 1024 / 1024} MB)`)
      e.target.value = ''
      return
    }
    const reader = new FileReader()
    reader.onload = () => {
      const text = typeof reader.result === 'string' ? reader.result : ''
      options.onTextLoaded(text, file)
    }
    reader.onerror = () => options.onError('Could not read file')
    reader.readAsText(file, 'utf-8')
    e.target.value = ''
  }

  return { fileInputRef, handleFileSelect, accept }
}

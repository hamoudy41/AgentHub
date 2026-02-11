import { useLoadFile } from '../hooks/useLoadFile'

interface FileLoadButtonProps {
  onTextLoaded: (text: string, file: File) => void
  onError: (message: string) => void
  onClearResult?: () => void
  hint?: string
  buttonLabel?: string
  testId?: string
}

export function FileLoadButton({
  onTextLoaded,
  onError,
  onClearResult,
  hint = '.txt, .md, .json, .csv, .xml, .html (max 5 MB)',
  buttonLabel = 'Load file',
  testId = 'file-upload',
}: FileLoadButtonProps) {
  const { fileInputRef, handleFileSelect, accept } = useLoadFile({
    onTextLoaded,
    onError,
    onClearResult,
  })

  return (
    <div className="flex flex-wrap items-center gap-2">
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={handleFileSelect}
        className="hidden"
        data-testid={testId}
      />
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        className="rounded-md border border-slate-400 bg-slate-100 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-200 dark:border-slate-600 dark:bg-slate-700 dark:text-slate-200 dark:hover:bg-slate-600"
      >
        {buttonLabel}
      </button>
      <span className="text-xs text-slate-500 dark:text-slate-400">{hint}</span>
    </div>
  )
}

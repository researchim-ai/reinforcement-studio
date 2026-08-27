import { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import { FolderOpen } from 'lucide-react'
import { cn } from '@/lib/utils'

interface RunFolderLinkProps {
  runId: string
  /** Backend-reported path — fallback when not in Electron. */
  runDir?: string | null
  /** Icon-only button (sidebar rows). */
  compact?: boolean
  className?: string
}

/** Opens a run's artifact folder in the OS file manager (Electron) or copies
 * the path to the clipboard (browser dev / web build). */
export function RunFolderLink({ runId, runDir, compact, className }: RunFolderLinkProps) {
  const [hostPath, setHostPath] = useState<string | null>(runDir ?? null)

  useEffect(() => {
    if (!window.electronAPI) return
    let cancelled = false
    window.electronAPI.runs.hostPath(runId).then((p) => {
      if (!cancelled) setHostPath(p)
    })
    return () => {
      cancelled = true
    }
  }, [runId])

  const displayPath = hostPath ?? runDir ?? null

  const handleOpen = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation()
    e.preventDefault()
    if (window.electronAPI) {
      try {
        await window.electronAPI.runs.openFolder(runId)
      } catch (err) {
        toast.error(err instanceof Error ? err.message : 'Не удалось открыть папку запуска')
      }
      return
    }
    if (displayPath) {
      try {
        await navigator.clipboard.writeText(displayPath)
        toast.success('Путь к папке скопирован в буфер обмена')
      } catch {
        toast.error('Не удалось скопировать путь')
      }
    }
  }, [runId, displayPath])

  if (!displayPath && !window.electronAPI) return null

  const title = window.electronAPI
    ? 'Открыть папку запуска (model.zip, episode_preview.gif, metrics.json, …)'
    : 'Скопировать путь к папке запуска'

  if (compact) {
    return (
      <button
        type="button"
        onClick={handleOpen}
        title={title}
        className={cn(
          'inline-flex shrink-0 items-center justify-center rounded p-1 text-muted-foreground hover:bg-muted hover:text-primary',
          className,
        )}
      >
        <FolderOpen className="h-3.5 w-3.5" />
      </button>
    )
  }

  return (
    <button
      type="button"
      onClick={handleOpen}
      title={title}
      className={cn(
        'flex max-w-full items-center gap-1 truncate rounded font-mono text-[11px] text-muted-foreground hover:text-primary hover:underline',
        className,
      )}
    >
      <FolderOpen className="h-3 w-3 shrink-0" />
      <span className="truncate">{displayPath ?? runId}</span>
    </button>
  )
}

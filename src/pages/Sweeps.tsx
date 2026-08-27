import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { toast } from 'sonner'
import { FlaskConical, Loader2, Trash2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useSweep, useSweeps } from '@/api/hooks'
import { api } from '@/api/client'
import { CompareRunsPanel } from '@/components/monitor/CompareRunsPanel'

/** List of all sweeps (queued/running/finished) plus a detail view for
 * whichever one is selected — reuses `CompareRunsPanel` to overlay the
 * reward curves of every combination × seed run the sweep queued. */
export function Sweeps() {
  const [searchParams, setSearchParams] = useSearchParams()
  const selectedId = searchParams.get('id')
  const { data: sweepsData, isLoading } = useSweeps()
  const { data: detail, isLoading: detailLoading } = useSweep(selectedId ?? undefined)
  const [deleting, setDeleting] = useState<string | null>(null)

  const sweeps = sweepsData?.sweeps ?? []

  useEffect(() => {
    if (!selectedId && sweeps.length > 0) {
      setSearchParams({ id: sweeps[0].sweep_id }, { replace: true })
    }
  }, [selectedId, sweeps, setSearchParams])

  const handleDelete = async (sweepId: string) => {
    setDeleting(sweepId)
    try {
      const res = await api.deleteSweep(sweepId)
      toast.success(`Sweep удалён, остановлено запусков: ${res.deleted_runs}`)
      if (selectedId === sweepId) {
        setSearchParams((prev) => {
          prev.delete('id')
          return prev
        })
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось удалить sweep')
    } finally {
      setDeleting(null)
    }
  }

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (sweeps.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-muted-foreground">
        <FlaskConical className="h-8 w-8" />
        <p className="text-sm">Пока нет ни одного sweep.</p>
        <p className="text-xs">Запустить sweep можно из Дизайнера экспериментов — кнопка «Запустить как sweep».</p>
      </div>
    )
  }

  return (
    <div className="grid h-full grid-cols-[280px_1fr] gap-4 overflow-hidden p-4">
      <div className="flex flex-col gap-2 overflow-y-auto pr-1">
        {sweeps.map((s) => (
          <Card
            key={s.sweep_id}
            className={`cursor-pointer transition-colors ${selectedId === s.sweep_id ? 'border-primary' : ''}`}
            onClick={() => setSearchParams({ id: s.sweep_id })}
          >
            <CardContent className="flex items-start justify-between gap-2 p-3">
              <div className="min-w-0">
                <p className="truncate text-sm font-medium">{s.name ?? s.sweep_id}</p>
                <p className="text-xs text-muted-foreground">
                  {s.algorithm_id ?? '—'} · {s.environment_id ?? '—'}
                </p>
                <p className="text-xs text-muted-foreground">{s.total} запусков</p>
              </div>
              <Button
                size="icon"
                variant="ghost"
                className="h-6 w-6 shrink-0 text-muted-foreground hover:text-destructive"
                disabled={deleting === s.sweep_id}
                onClick={(e) => {
                  e.stopPropagation()
                  handleDelete(s.sweep_id)
                }}
              >
                {deleting === s.sweep_id ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="overflow-y-auto pr-1">
        {!selectedId ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            Выберите sweep слева
          </div>
        ) : detailLoading || !detail ? (
          <div className="flex h-full items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">
                  {sweeps.find((s) => s.sweep_id === selectedId)?.name ?? selectedId} — {detail.runs.length} запусков,{' '}
                  {detail.runs.filter((r) => r.status === 'completed').length} завершено
                </CardTitle>
              </CardHeader>
            </Card>
            <CompareRunsPanel runs={detail.runs} paramsColumn />
          </div>
        )}
      </div>
    </div>
  )
}

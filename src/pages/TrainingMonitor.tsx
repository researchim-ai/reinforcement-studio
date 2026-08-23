import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { toast } from 'sonner'
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis,
} from 'recharts'
import { Loader2, Square, Trash2, Save, FileText } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Dialog } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useRuns } from '@/api/hooks'
import { api, createMetricsWebSocket } from '@/api/client'
import type { MetricsSnapshot } from '@/api/types'
import { cn, formatDuration } from '@/lib/utils'

const STATUS_VARIANT: Record<string, 'success' | 'secondary' | 'destructive' | 'outline'> = {
  running: 'success',
  completed: 'secondary',
  stopped: 'outline',
  failed: 'destructive',
  interrupted: 'destructive',
}

export function TrainingMonitor() {
  const [params, setParams] = useSearchParams()
  const selectedRunId = params.get('run') ?? undefined
  const { data: runsData, refetch } = useRuns()
  const runs = runsData?.runs ?? []

  const [history, setHistory] = useState<MetricsSnapshot[]>([])
  const [promoteOpen, setPromoteOpen] = useState(false)
  const [promoteName, setPromoteName] = useState('')
  const wsRef = useRef<WebSocket | null>(null)

  const selectedRun = runs.find((r) => r.run_id === selectedRunId)

  useEffect(() => {
    if (!selectedRunId) return
    setHistory([])
    let cancelled = false

    createMetricsWebSocket(selectedRunId, (snapshot) => {
      if (cancelled) return
      setHistory((prev) => [...prev.slice(-499), snapshot])
    }).then((ws) => {
      wsRef.current = ws
    })

    return () => {
      cancelled = true
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [selectedRunId])

  const selectRun = useCallback(
    (runId: string) => setParams({ run: runId }),
    [setParams],
  )

  const handleStop = useCallback(async () => {
    if (!selectedRunId) return
    await api.stopRun(selectedRunId)
    toast.success('Остановка запрошена')
    refetch()
  }, [selectedRunId, refetch])

  const handleDelete = useCallback(async () => {
    if (!selectedRunId) return
    try {
      await api.deleteRun(selectedRunId)
      setParams({})
      refetch()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось удалить запуск')
    }
  }, [selectedRunId, refetch, setParams])

  const handlePromote = useCallback(async () => {
    if (!selectedRunId || !promoteName.trim()) return
    try {
      await api.promoteRun(selectedRunId, promoteName.trim())
      toast.success('Модель сохранена в Model Zoo')
      setPromoteOpen(false)
      setPromoteName('')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось сохранить модель')
    }
  }, [selectedRunId, promoteName])

  const latest = history[history.length - 1] ?? (selectedRun?.metrics as MetricsSnapshot | undefined)
  const isAlphaZero = latest?.kind === 'alphazero' || selectedRun?.kind === 'alphazero'

  return (
    <div className="flex h-full">
      <div className="w-72 shrink-0 border-r border-border">
        <ScrollArea className="h-full p-3">
          <div className="space-y-2">
            {runs.length === 0 && (
              <p className="p-4 text-center text-sm text-muted-foreground">Пока нет запусков</p>
            )}
            {runs.map((run) => (
              <button
                key={run.run_id}
                onClick={() => selectRun(run.run_id)}
                className={cn(
                  'w-full rounded-lg border border-border p-3 text-left transition-colors hover:border-primary/50',
                  selectedRunId === run.run_id && 'border-primary bg-primary/5',
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-sm font-medium">{run.name}</span>
                  <Badge variant={STATUS_VARIANT[run.status] ?? 'outline'} className="shrink-0 text-[10px]">
                    {run.status}
                  </Badge>
                </div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {run.environment_id} · {run.algorithm_id}
                </div>
              </button>
            ))}
          </div>
        </ScrollArea>
      </div>

      <div className="flex-1 overflow-auto p-6">
        {!selectedRun ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            Выбери запуск слева, чтобы увидеть метрики
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold">{selectedRun.name}</h2>
                <p className="text-sm text-muted-foreground">
                  {selectedRun.environment_id} · {selectedRun.algorithm_id}
                </p>
              </div>
              <div className="flex gap-2">
                {selectedRun.running && (
                  <Button variant="outline" size="sm" onClick={handleStop}>
                    <Square className="h-3.5 w-3.5" /> Остановить
                  </Button>
                )}
                {selectedRun.has_model && (
                  <Button variant="outline" size="sm" onClick={() => setPromoteOpen(true)}>
                    <Save className="h-3.5 w-3.5" /> Сохранить в Model Zoo
                  </Button>
                )}
                {!selectedRun.running && (
                  <Button variant="destructive" size="sm" onClick={handleDelete}>
                    <Trash2 className="h-3.5 w-3.5" /> Удалить
                  </Button>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <StatCard label="Шаг / итерация" value={`${latest?.step ?? 0} / ${latest?.total_timesteps ?? '—'}`} />
              {isAlphaZero ? (
                <>
                  <StatCard label="Win-rate vs prev" value={latest?.win_rate_vs_prev != null ? `${(latest.win_rate_vs_prev * 100).toFixed(0)}%` : '—'} />
                  <StatCard label="Loss" value={latest?.loss?.toFixed(3) ?? '—'} />
                  <StatCard label="Buffer" value={String(latest?.buffer_size ?? '—')} />
                </>
              ) : (
                <>
                  <StatCard label="Средняя награда" value={latest?.episode_reward_mean?.toFixed(1) ?? '—'} />
                  <StatCard label="Длина эпизода" value={latest?.episode_length_mean?.toFixed(0) ?? '—'} />
                  <StatCard label="FPS" value={String(latest?.fps ?? '—')} />
                </>
              )}
              <StatCard label="Время" value={latest?.elapsed_seconds ? formatDuration(latest.elapsed_seconds) : '—'} />
            </div>

            {!isAlphaZero && latest?.frame_base64 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Живой просмотр среды</CardTitle>
                </CardHeader>
                <CardContent className="flex justify-center">
                  <img
                    src={`data:image/png;base64,${latest.frame_base64}`}
                    alt="env frame"
                    className="max-h-64 rounded-lg border border-border"
                  />
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">
                  {isAlphaZero ? 'Win-rate по итерациям' : 'Награда по шагам'}
                </CardTitle>
              </CardHeader>
              <CardContent className="h-64">
                {history.length < 2 ? (
                  <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Ждём метрики...
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.3 0 0)" />
                      <XAxis dataKey="step" stroke="oklch(0.65 0 0)" fontSize={11} />
                      <YAxis stroke="oklch(0.65 0 0)" fontSize={11} />
                      <RTooltip
                        contentStyle={{ background: 'oklch(0.17 0 0)', border: '1px solid oklch(0.3 0 0)', fontSize: 12 }}
                      />
                      {isAlphaZero ? (
                        <Line type="monotone" dataKey="win_rate_vs_prev" stroke="oklch(0.7 0.15 260)" dot={false} strokeWidth={2} />
                      ) : (
                        <Line type="monotone" dataKey="episode_reward_mean" stroke="oklch(0.7 0.15 260)" dot={false} strokeWidth={2} />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>

            {latest?.error && (
              <Card className="border-destructive/50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-sm text-destructive">
                    <FileText className="h-4 w-4" /> Ошибка
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-xs text-muted-foreground">{latest.error}</CardContent>
              </Card>
            )}
          </div>
        )}
      </div>

      <Dialog open={promoteOpen} onClose={() => setPromoteOpen(false)} title="Сохранить модель">
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            Модель попадёт в Model Zoo под этим именем и переживёт удаление запуска.
          </p>
          <Input
            value={promoteName}
            onChange={(e) => setPromoteName(e.target.value)}
            placeholder="Название модели"
            autoFocus
          />
          <Button className="w-full" onClick={handlePromote} disabled={!promoteName.trim()}>
            Сохранить
          </Button>
        </div>
      </Dialog>
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="p-3">
        <div className="text-lg font-semibold">{value}</div>
        <div className="text-xs text-muted-foreground">{label}</div>
      </CardContent>
    </Card>
  )
}

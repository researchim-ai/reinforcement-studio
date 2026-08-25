import { useCallback, useEffect, useMemo, useRef, useState, memo } from 'react'
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
import { AlgorithmDiagram, type AlgorithmDiagramNetwork } from '@/components/AlgorithmDiagram'
import type { MetricsSnapshot, SpaceInfo } from '@/api/types'
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

  // The backend only records a fresh episode/game every couple thousand
  // steps (rendering + encoding is expensive) — most metrics snapshots
  // simply omit `episode_gif_base64`/`board_history`. Reading those straight
  // off `latest` made the "Живой просмотр среды" card mount/unmount on every
  // snapshot that lacked them, i.e. exactly the "то исчезает то появляется"
  // flicker. Instead, carry the last non-empty one forward until a newer
  // one shows up — the GIF/board-replay component below then plays that
  // whole recorded episode/game on a loop until it's replaced.
  const { lastGif, lastBoardHistory, lastBoardWinner } = useMemo(() => {
    let gif = selectedRun?.metrics?.episode_gif_base64
    let boardHistory = selectedRun?.metrics?.board_history
    let boardWinner = selectedRun?.metrics?.board_winner
    for (const snap of history) {
      if (snap.episode_gif_base64) gif = snap.episode_gif_base64
      if (snap.board_history) boardHistory = snap.board_history
      if (snap.board_winner != null) boardWinner = snap.board_winner
    }
    return { lastGif: gif, lastBoardHistory: boardHistory, lastBoardWinner: boardWinner }
  }, [history, selectedRun])

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

            {!isAlphaZero && (
              latest?.exploration_epsilon != null
              || latest?.rnd_bonus_mean != null
              || latest?.rnd_predictor_loss != null
            ) && (
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                {latest?.exploration_epsilon != null && Number(latest.hyperparams?.action_exploration ?? 0) === 0 && (
                  <StatCard label="Текущий ε" value={latest.exploration_epsilon.toFixed(3)} />
                )}
                {latest?.episode_extrinsic_reward_mean != null && (
                  <StatCard label="Extrinsic reward" value={latest.episode_extrinsic_reward_mean.toFixed(2)} />
                )}
                {latest?.episode_intrinsic_reward_mean != null && (
                  <StatCard label="Intrinsic reward" value={latest.episode_intrinsic_reward_mean.toFixed(2)} />
                )}
                {latest?.rnd_predictor_loss != null && latest.rnd_predictor_loss > 0 && (
                  <StatCard label="RND predictor loss" value={latest.rnd_predictor_loss.toFixed(4)} />
                )}
              </div>
            )}

            {((!isAlphaZero && lastGif) || (isAlphaZero && lastBoardHistory && lastBoardHistory.length > 0)) && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Живой просмотр среды</CardTitle>
                  <p className="text-xs text-muted-foreground">
                    {isAlphaZero
                      ? 'Самая свежая self-play партия: проигрывается один раз и остаётся на финальной позиции до следующей итерации.'
                      : 'Самый свежий записанный эпизод: проигрывается один раз и остаётся на финальном кадре до следующей записи.'}
                  </p>
                </CardHeader>
                <CardContent className="flex flex-col items-center gap-2">
                  {!isAlphaZero && lastGif && (
                    <img
                      key={lastGif.slice(0, 32)}
                      src={`data:image/gif;base64,${lastGif}`}
                      alt="env episode replay"
                      className="max-h-64 rounded-lg border border-border"
                    />
                  )}
                  {isAlphaZero && lastBoardHistory && lastBoardHistory.length > 0 && (
                    <BoardReplay history={lastBoardHistory} winner={lastBoardWinner} />
                  )}
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Схема алгоритма</CardTitle>
              </CardHeader>
              <CardContent>
                <AlgorithmDiagram
                  algorithmId={selectedRun.algorithm_id}
                  kind={latest?.kind ?? selectedRun.kind}
                  hyperparams={latest?.hyperparams}
                  network={buildDiagramNetwork(latest)}
                />
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {latest?.hyperparams && Object.keys(latest.hyperparams).length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Гиперпараметры алгоритма</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <KeyValueGrid entries={Object.entries(latest.hyperparams)} />
                  </CardContent>
                </Card>
              )}

              {(latest?.total_params != null || latest?.network) && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Конфигурация модели</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <KeyValueGrid entries={buildNetworkEntries(latest)} />
                  </CardContent>
                </Card>
              )}

              {!isAlphaZero && latest?.wrappers && latest.wrappers.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Wrappers среды</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-1.5">
                      {latest.wrappers.map((w, i) => (
                        <Badge key={i} variant="secondary" className="text-[10px]">{w.type}</Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">
                  {isAlphaZero ? 'Win-rate по итерациям' : 'Награда по шагам (реальная и intrinsic отдельно)'}
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
                        <>
                          <Line name="Суммарная" type="monotone" dataKey="episode_reward_mean" stroke="oklch(0.7 0.15 260)" dot={false} strokeWidth={2} />
                          <Line name="Extrinsic" type="monotone" dataKey="episode_extrinsic_reward_mean" stroke="oklch(0.72 0.16 145)" dot={false} strokeWidth={1.5} connectNulls />
                          <Line name="Intrinsic" type="monotone" dataKey="episode_intrinsic_reward_mean" stroke="oklch(0.72 0.16 55)" dot={false} strokeWidth={1.5} connectNulls />
                        </>
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

function formatValue(value: unknown): string {
  if (value == null) return '—'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
  if (typeof value === 'boolean') return value ? 'да' : 'нет'
  return String(value)
}

function formatSpace(space?: SpaceInfo): string {
  if (!space) return '—'
  // `Discrete` spaces have an empty `shape` ([]) and carry their size in
  // `n` instead — `space.shape ? ... : ...` treated `[]` as truthy (empty
  // arrays are truthy in JS) and never fell through to the `n=` branch,
  // showing a blank "Discrete []" instead of "Discrete n=2".
  const shape = space.shape && space.shape.length > 0
    ? `[${space.shape.join(', ')}]`
    : space.n != null ? `n=${space.n}` : ''
  return `${space.type} ${shape}`.trim()
}

function spaceToShape(space?: SpaceInfo): number[] | undefined {
  if (!space) return undefined
  if (space.shape && space.shape.length > 0) return space.shape
  if (space.n != null) return [space.n]
  return undefined
}

/** Adapts whatever a live run's `MetricsSnapshot` happens to carry (AlphaZero's
 * `network{}` dims vs. gym runs' flat `layers`/spaces) into the one shape
 * `AlgorithmDiagram` understands — mirrors the InspectPanel's Designer-side
 * adapter for the exact same `InspectNetwork` data, just sourced from a live
 * run instead of a pre-run inspect call. */
function buildDiagramNetwork(latest: MetricsSnapshot | undefined): AlgorithmDiagramNetwork | null {
  if (!latest) return null
  if (latest.total_params == null && !latest.network && !latest.layers?.length) return null
  return {
    policy: latest.policy,
    layers: latest.layers,
    totalParams: latest.total_params,
    inputShape: spaceToShape(latest.observation_space),
    outputShape: spaceToShape(latest.action_space),
    channels: latest.network?.channels,
    numBlocks: latest.network?.num_blocks,
    actionSize: latest.network?.action_size,
    inputPlanes: latest.network?.input_planes,
    rows: latest.network?.rows,
    cols: latest.network?.cols,
  }
}

function buildNetworkEntries(latest: MetricsSnapshot): [string, string][] {
  const entries: [string, string][] = []
  if (latest.policy) entries.push(['Policy', latest.policy])
  if (latest.network) {
    entries.push(['Каналы / блоки', `${latest.network.channels} / ${latest.network.num_blocks}`])
    entries.push(['Доска', `${latest.network.rows}×${latest.network.cols}`])
    entries.push(['Размер действия', String(latest.network.action_size)])
    entries.push(['Входные плоскости', String(latest.network.input_planes)])
  }
  if (latest.observation_space) entries.push(['Observation space', formatSpace(latest.observation_space)])
  if (latest.action_space) entries.push(['Action space', formatSpace(latest.action_space)])
  if (latest.total_params != null) entries.push(['Параметры сети', latest.total_params.toLocaleString('ru-RU')])
  if (latest.device) entries.push(['Устройство', latest.device.toUpperCase()])
  if (latest.seed != null) entries.push(['Seed', String(latest.seed)])
  return entries
}

function KeyValueGrid({ entries }: { entries: [string, unknown][] }) {
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-3">
      {entries.map(([key, value]) => (
        <div key={key} className="flex flex-col">
          <span className="text-muted-foreground">{key}</span>
          <span className="font-mono font-medium">{formatValue(value)}</span>
        </div>
      ))}
    </div>
  )
}

/** Steps through the latest recorded self-play game once and remains on its
 * final position. A newer game replaces `history` and restarts playback. */
const BoardReplay = memo(function BoardReplay({ history, winner }: { history: number[][][]; winner: number | null | undefined }) {
  const [moveIndex, setMoveIndex] = useState(0)

  useEffect(() => {
    setMoveIndex(0)
    if (history.length <= 1) return
    let cancelled = false
    let timeoutId: ReturnType<typeof setTimeout>
    const scheduleNext = (idx: number) => {
      const isLast = idx === history.length - 1
      if (isLast) return
      timeoutId = setTimeout(() => {
        if (cancelled) return
        const next = idx + 1
        setMoveIndex(next)
        scheduleNext(next)
      }, 550)
    }
    scheduleNext(0)
    return () => {
      cancelled = true
      clearTimeout(timeoutId)
    }
  }, [history])

  const board = history[Math.min(moveIndex, history.length - 1)]
  const isLastMove = moveIndex === history.length - 1

  return (
    <div className="flex flex-col items-center gap-2">
      <BoardPreview board={board} />
      <p className="text-xs text-muted-foreground">
        Ход {moveIndex} / {history.length - 1}
        {isLastMove && winner != null && (
          winner === 0 ? ' · ничья' : ` · победил игрок ${winner === 1 ? '●' : '○'}`
        )}
      </p>
    </div>
  )
})

function BoardPreview({ board }: { board: number[][] }) {
  const cols = board[0]?.length ?? 0
  return (
    <div className="inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
      {board.map((rowVals, row) =>
        rowVals.map((cell, col) => (
          <div
            key={`${row}-${col}`}
            className={cn(
              'flex h-8 w-8 items-center justify-center rounded-md border border-border text-sm font-bold',
              cell === 0 ? 'bg-muted/40' : 'bg-muted',
            )}
          >
            {cell === 1 && <span className="text-primary">●</span>}
            {cell === -1 && <span className="text-warning">○</span>}
          </div>
        )),
      )}
    </div>
  )
}

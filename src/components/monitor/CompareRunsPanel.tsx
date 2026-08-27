import { useMemo } from 'react'
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip as RTooltip, Legend, XAxis, YAxis,
} from 'recharts'
import { Loader2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { RunFolderLink } from '@/components/monitor/RunFolderLink'
import { useMultiRunHistory } from '@/api/hooks'
import type { MetricsSnapshot, RunSummary } from '@/api/types'

// Distinct, readable-on-dark hues (oklch) — cycles if there are more runs
// than colors, same palette style as the rest of the app's charts.
const PALETTE = [
  'oklch(0.7 0.15 260)', 'oklch(0.72 0.16 145)', 'oklch(0.72 0.16 55)',
  'oklch(0.72 0.16 30)', 'oklch(0.7 0.18 320)', 'oklch(0.75 0.14 200)',
  'oklch(0.68 0.2 15)', 'oklch(0.78 0.12 100)',
]

function colorFor(index: number): string {
  return PALETTE[index % PALETTE.length]
}

/** Overlays every selected run's reward curve on one chart (each `<Line>`
 * carries its own `data`, since runs rarely share the same `step` values —
 * recharts happily draws several independently-keyed series on one shared
 * axis this way) plus a compact results table, sortable by final reward so
 * the best run in a batch/sweep jumps out immediately. Shared by the
 * Training Monitor's "Сравнение" mode and the Sweeps results view. */
export function CompareRunsPanel({ runs, paramsColumn }: { runs: RunSummary[]; paramsColumn?: boolean }) {
  const runIds = useMemo(() => runs.map((r) => r.run_id), [runs])
  const { data: histories, isLoading } = useMultiRunHistory(runIds)

  const series = useMemo(() => {
    return runs.map((run, i) => {
      const history = histories?.[run.run_id] ?? []
      const sorted = [...history].sort((a, b) => a.step - b.step)
      const last = sorted[sorted.length - 1] as MetricsSnapshot | undefined
      const rewards = sorted.map((s) => s.episode_reward_mean).filter((v): v is number => v != null)
      const best = rewards.length ? Math.max(...rewards) : null
      return {
        run,
        color: colorFor(i),
        data: sorted,
        finalReward: last?.episode_reward_mean ?? null,
        bestReward: best,
        finalStep: last?.step ?? 0,
      }
    })
  }, [runs, histories])

  const sortedByReward = useMemo(
    () => [...series].sort((a, b) => (b.finalReward ?? -Infinity) - (a.finalReward ?? -Infinity)),
    [series],
  )
  const bestRunId = sortedByReward[0]?.finalReward != null ? sortedByReward[0].run.run_id : null

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Сравнение наград</CardTitle>
        </CardHeader>
        <CardContent className="h-72">
          {isLoading ? (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Загружаем историю...
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart>
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.3 0 0)" />
                <XAxis dataKey="step" type="number" domain={['dataMin', 'dataMax']} stroke="oklch(0.65 0 0)" fontSize={11} />
                <YAxis stroke="oklch(0.65 0 0)" fontSize={11} />
                <RTooltip contentStyle={{ background: 'oklch(0.17 0 0)', border: '1px solid oklch(0.3 0 0)', fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {series.map((s) => (
                  <Line
                    key={s.run.run_id}
                    name={s.run.name}
                    data={s.data}
                    type="linear"
                    dataKey="episode_reward_mean"
                    stroke={s.color}
                    dot={false}
                    strokeWidth={2}
                    connectNulls
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Итоги</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border text-left text-muted-foreground">
                  <th className="py-1.5 pr-3 font-medium">Запуск</th>
                  <th className="py-1.5 pr-3 font-medium">Папка</th>
                  <th className="py-1.5 pr-3 font-medium">Статус</th>
                  {paramsColumn && <th className="py-1.5 pr-3 font-medium">Параметры</th>}
                  <th className="py-1.5 pr-3 font-medium">Шаг</th>
                  <th className="py-1.5 pr-3 font-medium">Финальная награда</th>
                  <th className="py-1.5 pr-3 font-medium">Лучшая награда</th>
                </tr>
              </thead>
              <tbody>
                {sortedByReward.map((s) => (
                  <tr key={s.run.run_id} className="border-b border-border/50">
                    <td className="py-1.5 pr-3">
                      <span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
                      {s.run.name}
                      {s.run.run_id === bestRunId && (
                        <span className="ml-1.5 rounded bg-success/15 px-1 py-0.5 text-[10px] text-success">лучший</span>
                      )}
                    </td>
                    <td className="py-1.5 pr-3">
                      <RunFolderLink runId={s.run.run_id} runDir={s.run.run_dir} compact />
                    </td>
                    <td className="py-1.5 pr-3 text-muted-foreground">{s.run.status}</td>
                    {paramsColumn && (
                      <td className="py-1.5 pr-3 font-mono text-[10px] text-muted-foreground">
                        {s.run.sweep?.params && Object.keys(s.run.sweep.params).length > 0
                          ? Object.entries(s.run.sweep.params).map(([k, v]) => `${k}=${v}`).join(', ')
                          : '—'}
                        {s.run.sweep?.seed != null && ` · seed=${s.run.sweep.seed}`}
                      </td>
                    )}
                    <td className="py-1.5 pr-3">{s.finalStep}</td>
                    <td className="py-1.5 pr-3 font-medium">{s.finalReward != null ? s.finalReward.toFixed(2) : '—'}</td>
                    <td className="py-1.5 pr-3">{s.bestReward != null ? s.bestReward.toFixed(2) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

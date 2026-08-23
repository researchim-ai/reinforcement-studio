import { AlertTriangle, Boxes, Cpu, Loader2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { formatNumber } from '@/lib/utils'
import type { InspectResult, SpaceInfo } from '@/api/types'

function formatSpace(space?: SpaceInfo | null): string {
  if (!space) return '—'
  if (space.n !== undefined) return `${space.type}(n=${space.n})`
  if (space.shape && space.shape.length > 0) return `${space.type}[${space.shape.join('×')}]`
  return space.type
}

export interface InspectPanelProps {
  data: InspectResult | undefined
  isFetching: boolean
}

/** Live, read-only preview docked over the Designer canvas — shows the
 * effective observation/action space of the env and the resulting network's
 * input/output dims, layer summary and parameter count, recomputed on the
 * backend (without training) whenever the env/wrappers/algorithm/hyperparams
 * change. Lets you track "сколько входов/выходов у сети и у среды" while
 * still designing. */
export function InspectPanel({ data, isFetching }: InspectPanelProps) {
  const env = data?.environment
  const net = data?.network
  const obsChangedByWrappers =
    !!env?.raw_observation_space &&
    !!env?.observation_space &&
    JSON.stringify(env.raw_observation_space) !== JSON.stringify(env.observation_space)

  return (
    <div className="absolute right-4 top-4 z-10 w-72 space-y-3">
      <Card className="bg-card/95 shadow-md backdrop-blur">
        <CardHeader className="flex-row items-center gap-2 space-y-0 p-3">
          <Boxes className="h-4 w-4 text-primary" />
          <CardTitle className="text-xs font-semibold uppercase tracking-wide">Среда</CardTitle>
          {isFetching && <Loader2 className="ml-auto h-3.5 w-3.5 animate-spin text-muted-foreground" />}
        </CardHeader>
        <CardContent className="space-y-1.5 p-3 pt-0 text-xs">
          {!env ? (
            <p className="text-muted-foreground">Выберите среду</p>
          ) : (
            <>
              <div className="flex justify-between gap-2">
                <span className="text-muted-foreground">Наблюдения (вход)</span>
                <span className="text-right font-mono">{formatSpace(env.observation_space)}</span>
              </div>
              {obsChangedByWrappers && (
                <div className="flex justify-between gap-2 text-[10px] text-muted-foreground/70">
                  <span>до wrapper&apos;ов</span>
                  <span className="text-right font-mono line-through">{formatSpace(env.raw_observation_space)}</span>
                </div>
              )}
              <div className="flex justify-between gap-2">
                <span className="text-muted-foreground">Действия (выход)</span>
                <span className="text-right font-mono">{formatSpace(env.action_space)}</span>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      <Card className="bg-card/95 shadow-md backdrop-blur">
        <CardHeader className="flex-row items-center gap-2 space-y-0 p-3">
          <Cpu className="h-4 w-4 text-primary" />
          <CardTitle className="text-xs font-semibold uppercase tracking-wide">Нейросеть</CardTitle>
          {net && <Badge variant="outline" className="ml-auto text-[9px]">{net.policy}</Badge>}
        </CardHeader>
        <CardContent className="space-y-1.5 p-3 pt-0 text-xs">
          {data?.error && (
            <div className="flex items-start gap-1.5 text-warning">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span className="text-[11px]">{data.error}</span>
            </div>
          )}
          {!net ? (
            !data?.error && <p className="text-muted-foreground">—</p>
          ) : (
            <>
              <div className="flex justify-between gap-2">
                <span className="text-muted-foreground">Вход</span>
                <span className="font-mono">[{net.input_shape.join('×')}]</span>
              </div>
              <div className="flex justify-between gap-2">
                <span className="text-muted-foreground">Выход</span>
                <span className="font-mono">[{net.output_shape.join('×')}]</span>
              </div>
              <div className="flex justify-between gap-2">
                <span className="text-muted-foreground">Параметры</span>
                <span className="font-mono">{formatNumber(net.total_params)}</span>
              </div>
              {net.channels !== undefined && (
                <div className="flex justify-between gap-2">
                  <span className="text-muted-foreground">Conv-блоки</span>
                  <span className="font-mono">{net.channels} ch × {net.num_blocks}</span>
                </div>
              )}
              {net.note && <p className="text-[11px] text-muted-foreground">{net.note}</p>}
              {net.layers.length > 0 && (
                <div className="max-h-40 space-y-0.5 overflow-auto rounded-md border border-border/60 bg-background/40 p-1.5 font-mono text-[10px] text-muted-foreground">
                  {net.layers.map((l, i) => (
                    <div key={i}>{l}</div>
                  ))}
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

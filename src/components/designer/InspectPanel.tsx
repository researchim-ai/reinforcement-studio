import { memo } from 'react'
import { AlertTriangle, Boxes, Cpu, Loader2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { AlgorithmDiagram } from '@/components/AlgorithmDiagram'
import { formatSpace, spaceSize } from '@/lib/spaceInfo'
import type { InspectResult } from '@/api/types'

export interface InspectPanelProps {
  data: InspectResult | undefined
  isFetching: boolean
  sceneAgentCount?: number | null
  sceneTeamCount?: number | null
}

/** Live, read-only preview docked over the Designer canvas — shows the
 * effective observation/action space of the env and the resulting network's
 * input/output dims, layer summary and parameter count, recomputed on the
 * backend (without training) whenever the env/wrappers/algorithm/hyperparams
 * change. Lets you track "сколько входов/выходов у сети и у среды" while
 * still designing. */
/** Memoized so dragging a node — which re-renders the whole Designer page
 * on every pointer-move frame to update that node's position — doesn't also
 * re-render (and repaint the `backdrop-blur`/shadow on) this floating panel
 * dozens of times a second when its own props haven't actually changed.
 * That repeated repaint of a blurred, elevated card is what read as a
 * strong flicker while dragging. */
export const InspectPanel = memo(function InspectPanel({ data, isFetching, sceneAgentCount, sceneTeamCount }: InspectPanelProps) {
  const env = data?.environment
  const net = data?.network
  const obsChangedByWrappers =
    !!env?.raw_observation_space &&
    !!env?.observation_space &&
    JSON.stringify(env.raw_observation_space) !== JSON.stringify(env.observation_space)

  return (
    <div className="absolute right-4 top-4 z-10 w-96 space-y-3">
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
              {sceneAgentCount != null && sceneAgentCount > 0 && (
                <p className="rounded-md bg-primary/10 px-2 py-1 text-[10px] text-primary">
                  Общий мир: {sceneAgentCount} агент(ов) в одном мире, общая политика (parameter sharing).
                  Поле num_envs в узле обучения игнорируется.
                  {sceneTeamCount != null && (
                    sceneTeamCount >= 2
                      ? ` Команд: ${sceneTeamCount} — доступны IPPO (под несколько команд) и, для дискретных `
                        + 'действий, QMIX.'
                      : ' Команда одна — IPPO не показывается (нужно 2+ команды), но QMIX доступен и для одной '
                        + 'команды из 2+ агентов (это его собственный классический случай — общий Q, '
                        + 'разложенный по агентам). Для непрерывных действий — только обычные PPO/DQN/...'
                  )}
                </p>
              )}
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-muted-foreground">Входов (наблюдения)</span>
                <span className="text-right">
                  <span className="font-mono text-sm font-semibold text-foreground">
                    {spaceSize(env.observation_space) ?? '—'}
                  </span>
                  <span className="ml-1.5 font-mono text-[10px] text-muted-foreground">
                    {formatSpace(env.observation_space)}
                  </span>
                </span>
              </div>
              {obsChangedByWrappers && (
                <div className="flex justify-between gap-2 text-[10px] text-muted-foreground/70">
                  <span>до wrapper&apos;ов</span>
                  <span className="text-right font-mono line-through">{formatSpace(env.raw_observation_space)}</span>
                </div>
              )}
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-muted-foreground">Выходов (действия)</span>
                <span className="text-right">
                  <span className="font-mono text-sm font-semibold text-foreground">
                    {spaceSize(env.action_space) ?? '—'}
                  </span>
                  <span className="ml-1.5 font-mono text-[10px] text-muted-foreground">
                    {formatSpace(env.action_space)}
                  </span>
                </span>
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
            <AlgorithmDiagram
              show={{ loop: false, network: true }}
              hideTitles
              network={{
                policy: net.policy,
                layers: net.layers,
                components: net.architecture_components,
                totalParams: net.total_params,
                trainableParams: net.trainable_params,
                inputShape: net.input_shape,
                outputShape: net.output_shape,
                channels: net.channels,
                numBlocks: net.num_blocks,
                actionSize: net.action_size,
                inputPlanes: net.input_planes,
                rows: net.rows,
                cols: net.cols,
                note: net.note,
              }}
            />
          )}
        </CardContent>
      </Card>
    </div>
  )
})

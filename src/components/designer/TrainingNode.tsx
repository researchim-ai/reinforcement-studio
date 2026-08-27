import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Rocket, Loader2 } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { NumericInput } from '@/components/ui/numeric-input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import type { EnvKind } from '@/api/types'

export interface TrainingNodeData {
  kind: EnvKind
  name: string
  totalTimesteps: number
  recommendedTotalTimesteps?: number | null
  numEnvs: number
  recommendedNumEnvs?: number | null
  numIterations: number
  seed: number
  useGpu: boolean
  starting: boolean
  disabled: boolean
  onChangeName: (v: string) => void
  onChangeTotalTimesteps: (v: number) => void
  onChangeNumEnvs: (v: number) => void
  onChangeNumIterations: (v: number) => void
  onChangeSeed: (v: number) => void
  onChangeUseGpu: (v: boolean) => void
  onRun: () => void
}

// See AlgorithmNode.tsx for why this is memoized against `data` only —
// otherwise the node re-renders (and visibly flickers) on every drag frame.
export const TrainingNode = memo(function TrainingNode({ data }: NodeProps & { data: TrainingNodeData }) {
  return (
    <div className="w-72 rounded-xl border border-border bg-card shadow-md">
      <div className="flex items-center gap-2 rounded-t-xl border-b border-border bg-success/10 px-3 py-2">
        <Rocket className="h-4 w-4 text-success" />
        <span className="text-xs font-semibold uppercase tracking-wide text-success">Training</span>
      </div>
      <div className="space-y-3 p-3">
        <div className="space-y-1">
          <Label className="text-[11px] text-muted-foreground">Название запуска</Label>
          <Input
            value={data.name}
            onChange={(e) => data.onChangeName(e.target.value)}
            placeholder="Мой эксперимент"
            className="h-7 text-xs"
          />
        </div>

        {data.kind === 'gym' ? (
          <div className="space-y-1">
            <Label className="text-[11px] text-muted-foreground">
              Total timesteps <span className="text-muted-foreground/60">(без ограничений)</span>
            </Label>
            <NumericInput
              integer
              value={data.totalTimesteps}
              onChange={data.onChangeTotalTimesteps}
              className="h-7 text-xs"
            />
            <p className="text-[10px] text-muted-foreground/60">
              Общая длина обучения — не путать с гиперпараметрами алгоритма слева (например, «Replay buffer size» у DQN).
            </p>
            {!!data.recommendedTotalTimesteps && data.totalTimesteps < data.recommendedTotalTimesteps && (
              <p className="text-[10px] text-warning">
                Эта среда сложная: рекомендуем ≥{data.recommendedTotalTimesteps.toLocaleString('ru-RU')} шагов,
                иначе прогон скорее всего закончится на стадии «пока ничего не выучил».{' '}
                <button
                  type="button"
                  className="underline"
                  onClick={() => data.onChangeTotalTimesteps(data.recommendedTotalTimesteps!)}
                >
                  Поставить {data.recommendedTotalTimesteps.toLocaleString('ru-RU')}
                </button>
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-1">
            <Label className="text-[11px] text-muted-foreground">Self-play итераций</Label>
            <NumericInput
              integer
              value={data.numIterations}
              onChange={data.onChangeNumIterations}
              className="h-7 text-xs"
            />
          </div>
        )}

        {data.kind === 'gym' && (
          <div className="space-y-1">
            <Label className="text-[11px] text-muted-foreground">Параллельных сред (num_envs)</Label>
            <NumericInput
              integer
              value={data.numEnvs}
              onChange={data.onChangeNumEnvs}
              className="h-7 text-xs"
            />
            <p className="text-[10px] text-muted-foreground/60">
              Сколько независимых копий среды крутятся параллельно (отдельный
              процесс на каждую) — больше опыта за шаг и быстрее сбор данных на
              CPU. 1 — одна среда, как раньше.
            </p>
            {!!data.recommendedNumEnvs && data.numEnvs < data.recommendedNumEnvs && (
              <p className="text-[10px] text-warning">
                Эта среда выигрывает от параллелизма: рекомендуем ≥{data.recommendedNumEnvs}.{' '}
                <button
                  type="button"
                  className="underline"
                  onClick={() => data.onChangeNumEnvs(data.recommendedNumEnvs!)}
                >
                  Поставить {data.recommendedNumEnvs}
                </button>
              </p>
            )}
          </div>
        )}

        <div className="space-y-1">
          <Label className="text-[11px] text-muted-foreground">Seed</Label>
          <NumericInput
            integer
            value={data.seed}
            onChange={data.onChangeSeed}
            className="h-7 text-xs"
          />
        </div>

        <div className="flex items-center justify-between">
          <Label className="text-[11px] text-muted-foreground">Использовать GPU</Label>
          <Switch checked={data.useGpu} onCheckedChange={data.onChangeUseGpu} />
        </div>
        {data.useGpu && (
          <p className="text-[10px] text-muted-foreground/60">
            Нужна CUDA-версия PyTorch — включите GPU в Настройках (Settings), иначе прогон тихо пойдёт на CPU.
          </p>
        )}

        <Button className="w-full" size="sm" onClick={data.onRun} disabled={data.disabled || data.starting}>
          {data.starting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Rocket className="h-3.5 w-3.5" />}
          Запустить обучение
        </Button>
      </div>
      <Handle type="target" position={Position.Left} className="!bg-success" />
    </div>
  )
}, (prev, next) => prev.data === next.data)

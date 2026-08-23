import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Rocket, Loader2 } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import type { EnvKind } from '@/api/types'

export interface TrainingNodeData {
  kind: EnvKind
  name: string
  totalTimesteps: number
  numIterations: number
  seed: number
  useGpu: boolean
  starting: boolean
  disabled: boolean
  onChangeName: (v: string) => void
  onChangeTotalTimesteps: (v: number) => void
  onChangeNumIterations: (v: number) => void
  onChangeSeed: (v: number) => void
  onChangeUseGpu: (v: boolean) => void
  onRun: () => void
}

export function TrainingNode({ data }: NodeProps & { data: TrainingNodeData }) {
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
            <Input
              type="number"
              min={1}
              value={data.totalTimesteps}
              onChange={(e) => data.onChangeTotalTimesteps(Math.max(1, Number(e.target.value) || 0))}
              className="h-7 text-xs"
            />
            <p className="text-[10px] text-muted-foreground/60">
              Общая длина обучения — не путать с гиперпараметрами алгоритма слева (например, «Replay buffer size» у DQN).
            </p>
          </div>
        ) : (
          <div className="space-y-1">
            <Label className="text-[11px] text-muted-foreground">Self-play итераций</Label>
            <Input
              type="number"
              value={data.numIterations}
              onChange={(e) => data.onChangeNumIterations(Number(e.target.value))}
              className="h-7 text-xs"
            />
          </div>
        )}

        <div className="space-y-1">
          <Label className="text-[11px] text-muted-foreground">Seed</Label>
          <Input
            type="number"
            value={data.seed}
            onChange={(e) => data.onChangeSeed(Number(e.target.value))}
            className="h-7 text-xs"
          />
        </div>

        <div className="flex items-center justify-between">
          <Label className="text-[11px] text-muted-foreground">Использовать GPU</Label>
          <Switch checked={data.useGpu} onCheckedChange={data.onChangeUseGpu} />
        </div>

        <Button className="w-full" size="sm" onClick={data.onRun} disabled={data.disabled || data.starting}>
          {data.starting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Rocket className="h-3.5 w-3.5" />}
          Запустить обучение
        </Button>
      </div>
      <Handle type="target" position={Position.Left} className="!bg-success" />
    </div>
  )
}

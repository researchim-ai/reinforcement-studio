import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Cpu } from 'lucide-react'
import { Select } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import type { AlgorithmSpec } from '@/api/types'

export interface AlgorithmNodeData {
  algorithms: AlgorithmSpec[]
  selectedId: string
  hyperparams: Record<string, number>
  onChangeAlgo: (id: string) => void
  onChangeHyperparam: (key: string, value: number) => void
}

export function AlgorithmNode({ data }: NodeProps & { data: AlgorithmNodeData }) {
  const selected = data.algorithms.find((a) => a.id === data.selectedId)

  return (
    <div className="w-72 rounded-xl border border-border bg-card shadow-md">
      <div className="flex items-center gap-2 rounded-t-xl border-b border-border bg-primary/10 px-3 py-2">
        <Cpu className="h-4 w-4 text-primary" />
        <span className="text-xs font-semibold uppercase tracking-wide text-primary">Algorithm</span>
        {selected?.is_custom && <Badge variant="secondary" className="ml-auto text-[9px]">custom</Badge>}
      </div>
      <div className="space-y-3 p-3">
        <Select
          value={data.selectedId}
          onChange={(e) => data.onChangeAlgo(e.target.value)}
          options={data.algorithms.map((a) => ({ value: a.id, label: a.name }))}
        />
        {selected && (
          <>
            <p className="text-xs text-muted-foreground">{selected.description}</p>
            <div className="max-h-56 space-y-2 overflow-auto pr-1">
              {selected.hyperparams.map((hp) => (
                <div key={hp.key} className="space-y-1">
                  <div className="flex items-baseline justify-between gap-2">
                    <Label className="text-[11px] text-muted-foreground">{hp.label}</Label>
                    {(hp.min != null || hp.max != null) && (
                      <span className="text-[10px] text-muted-foreground/60">
                        {hp.min ?? '−∞'}–{hp.max ?? '∞'}
                      </span>
                    )}
                  </div>
                  <Input
                    type="number"
                    step={hp.type === 'int' ? 1 : 'any'}
                    min={hp.min}
                    max={hp.max}
                    value={data.hyperparams[hp.key] ?? hp.default}
                    onChange={(e) => {
                      const raw = Number(e.target.value)
                      // Browsers don't clamp typed (non-spinner) input to
                      // min/max on their own — without this a stray extra
                      // digit here (e.g. DQN's "Replay buffer size", which
                      // happens to share its 50 000 default with "Total
                      // timesteps" on the Training node) silently sails past
                      // its declared bound instead of the user's actually
                      // intended field.
                      const clamped = hp.max != null ? Math.min(raw, hp.max) : raw
                      data.onChangeHyperparam(hp.key, hp.min != null ? Math.max(clamped, hp.min) : clamped)
                    }}
                    className="h-7 text-xs"
                  />
                </div>
              ))}
            </div>
          </>
        )}
      </div>
      <Handle type="target" position={Position.Left} className="!bg-primary" />
      <Handle type="source" position={Position.Right} className="!bg-primary" />
    </div>
  )
}

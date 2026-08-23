import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Cpu } from 'lucide-react'
import { Select } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
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
                  <Label className="text-[11px] text-muted-foreground">{hp.label}</Label>
                  <Input
                    type="number"
                    step={hp.type === 'int' ? 1 : 'any'}
                    min={hp.min}
                    max={hp.max}
                    value={data.hyperparams[hp.key] ?? hp.default}
                    onChange={(e) => data.onChangeHyperparam(hp.key, Number(e.target.value))}
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

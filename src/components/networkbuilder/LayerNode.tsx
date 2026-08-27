import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { ArrowDown, ArrowUp, Lock, X } from 'lucide-react'
import { Select } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { NumericInput } from '@/components/ui/numeric-input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ACTIVATION_LABELS, LAYER_LABELS, LAYER_TYPES } from '@/lib/networkBuilder'
import type { ActivationFn, NetworkLayer, NetworkLayerType } from '@/api/types'

export interface LayerNodeData {
  layer: NetworkLayer
  outShape: number[] | null
  error: string | null
  locked?: boolean // last layer of a head — type/out_features are fixed (auto-sized by the env)
  removable?: boolean
  hasHandleIn?: boolean
  hasHandleOut?: boolean
  onChange: (layer: NetworkLayer) => void
  onRemove?: () => void
  onMoveUp?: () => void
  onMoveDown?: () => void
}

// See designer/AlgorithmNode.tsx for why this is memoized against `data`
// only — otherwise the node re-renders (and visibly flickers) on every drag frame.
export const LayerNode = memo(function LayerNode({ data }: NodeProps & { data: LayerNodeData }) {
  const { layer, outShape, error, locked } = data

  const setType = (type: NetworkLayerType) => {
    if (type === layer.type) return
    const base: NetworkLayer = { type }
    if (type === 'linear') base.out_features = 64
    if (type === 'conv2d') Object.assign(base, { out_channels: 32, kernel_size: 3, stride: 1, padding: 0 })
    if (type === 'maxpool2d') Object.assign(base, { kernel_size: 2, stride: 2 })
    if (type === 'activation') base.fn = 'relu'
    if (type === 'dropout') base.p = 0.1
    data.onChange(base)
  }

  return (
    <div
      className={`w-52 rounded-lg border shadow-sm ${error ? 'border-destructive' : 'border-border'} bg-card`}
    >
      <div className="flex items-center gap-1 rounded-t-lg border-b border-border bg-muted/60 px-2 py-1.5">
        {locked ? (
          <span className="flex items-center gap-1 text-[11px] font-semibold text-muted-foreground">
            <Lock className="h-3 w-3" /> {LAYER_LABELS[layer.type]}
          </span>
        ) : (
          <Select
            value={layer.type}
            onChange={(e) => setType(e.target.value as NetworkLayerType)}
            options={LAYER_TYPES.map((t) => ({ value: t, label: LAYER_LABELS[t] }))}
            className="nodrag h-6 flex-1 text-[11px]"
          />
        )}
        <div className="ml-auto flex items-center gap-0.5">
          {data.onMoveUp && (
            <Button variant="ghost" size="icon" className="nodrag h-5 w-5" onClick={data.onMoveUp}>
              <ArrowUp className="h-3 w-3" />
            </Button>
          )}
          {data.onMoveDown && (
            <Button variant="ghost" size="icon" className="nodrag h-5 w-5" onClick={data.onMoveDown}>
              <ArrowDown className="h-3 w-3" />
            </Button>
          )}
          {data.removable && data.onRemove && (
            <Button variant="ghost" size="icon" className="nodrag h-5 w-5" onClick={data.onRemove}>
              <X className="h-3 w-3" />
            </Button>
          )}
        </div>
      </div>
      <div className="space-y-1.5 p-2">
        {layer.type === 'linear' && (
          <div className="space-y-0.5">
            <span className="text-[10px] text-muted-foreground">out_features</span>
            {locked ? (
              <Input disabled value="авто (по среде)" className="nodrag h-6 text-[11px]" />
            ) : (
              <NumericInput
                integer
                value={layer.out_features ?? 1}
                onChange={(v) => data.onChange({ ...layer, out_features: v })}
                className="nodrag h-6 text-[11px]"
              />
            )}
          </div>
        )}
        {layer.type === 'conv2d' && (
          <div className="grid grid-cols-2 gap-1">
            <LabeledNum label="channels" value={layer.out_channels} onChange={(v) => data.onChange({ ...layer, out_channels: v })} />
            <LabeledNum label="kernel" value={layer.kernel_size} onChange={(v) => data.onChange({ ...layer, kernel_size: v })} />
            <LabeledNum label="stride" value={layer.stride} onChange={(v) => data.onChange({ ...layer, stride: v })} />
            <LabeledNum label="padding" value={layer.padding} onChange={(v) => data.onChange({ ...layer, padding: v })} integer />
          </div>
        )}
        {layer.type === 'maxpool2d' && (
          <div className="grid grid-cols-2 gap-1">
            <LabeledNum label="kernel" value={layer.kernel_size} onChange={(v) => data.onChange({ ...layer, kernel_size: v })} />
            <LabeledNum label="stride" value={layer.stride} onChange={(v) => data.onChange({ ...layer, stride: v })} />
          </div>
        )}
        {layer.type === 'activation' && (
          <Select
            value={layer.fn ?? 'relu'}
            onChange={(e) => data.onChange({ ...layer, fn: e.target.value as ActivationFn })}
            options={Object.entries(ACTIVATION_LABELS).map(([value, label]) => ({ value, label }))}
            className="nodrag h-6 text-[11px]"
          />
        )}
        {layer.type === 'dropout' && (
          <LabeledNum label="p" value={layer.p} onChange={(v) => data.onChange({ ...layer, p: v })} />
        )}

        <div className="flex items-center justify-between pt-0.5">
          <span className="font-mono text-[10px] text-muted-foreground">
            {outShape ? `→ [${outShape.join('×')}]` : '\u00A0'}
          </span>
        </div>
        {error && <Badge variant="destructive" className="w-full justify-center text-[9px] leading-tight">{error}</Badge>}
      </div>
      {data.hasHandleIn !== false && <Handle type="target" position={Position.Top} className="!bg-primary" />}
      {data.hasHandleOut !== false && <Handle type="source" position={Position.Bottom} className="!bg-primary" />}
    </div>
  )
}, (prev, next) => prev.data === next.data)

function LabeledNum({
  label, value, onChange, integer = true,
}: { label: string; value: number | null | undefined; onChange: (v: number) => void; integer?: boolean }) {
  return (
    <div className="space-y-0.5">
      <span className="text-[10px] text-muted-foreground">{label}</span>
      <NumericInput
        integer={integer}
        value={value ?? 0}
        onChange={onChange}
        className="nodrag h-6 text-[11px]"
      />
    </div>
  )
}

import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'

export interface LabelNodeData {
  title: string
  subtitle?: string
  variant?: 'input' | 'head'
  error?: string | null
}

// See designer/AlgorithmNode.tsx for why this is memoized against `data`
// only — otherwise the node re-renders (and visibly flickers) on every drag frame.
export const LabelNode = memo(function LabelNode({ data }: NodeProps & { data: LabelNodeData }) {
  const isInput = data.variant === 'input'
  return (
    <div
      className={`w-52 rounded-lg border px-3 py-2 text-center shadow-sm ${
        data.error ? 'border-destructive bg-destructive/10' : isInput ? 'border-primary/40 bg-primary/10' : 'border-border bg-muted/60'
      }`}
    >
      <div className="text-xs font-semibold">{data.title}</div>
      {data.subtitle && <div className="font-mono text-[10px] text-muted-foreground">{data.subtitle}</div>}
      {data.error && <div className="mt-0.5 text-[9px] text-destructive">{data.error}</div>}
      {!isInput && <Handle type="target" position={Position.Top} className="!bg-primary" />}
      <Handle type="source" position={Position.Bottom} className="!bg-primary" />
    </div>
  )
}, (prev, next) => prev.data === next.data)

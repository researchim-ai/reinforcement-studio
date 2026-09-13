import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Lock } from 'lucide-react'
import { cn, formatNumber } from '@/lib/utils'
import type { CompositeGraphNodeData, CompositeNodeKind } from '@/lib/compositeGraph'

const KIND_CLASS: Record<CompositeNodeKind, string> = {
  input: 'border-primary/40 bg-primary/10',
  encoder: 'border-border bg-card',
  component: 'border-border bg-card',
  fixed: 'border-primary/25 bg-card',
  head: 'border-border bg-muted/60',
  ema: 'border-dashed border-muted-foreground/40 bg-muted/20',
}

const HEADER_CLASS: Record<CompositeNodeKind, string> = {
  input: 'border-primary/20 bg-primary/10',
  encoder: 'border-border bg-muted/60',
  component: 'border-border bg-muted/60',
  fixed: 'border-primary/20 bg-primary/10',
  head: 'border-border bg-muted/40',
  ema: 'border-transparent bg-transparent',
}

export const CompositeGraphNode = memo(function CompositeGraphNode({
  data,
}: NodeProps & { data: CompositeGraphNodeData }) {
  const showLock = data.kind === 'input' || data.kind === 'head' || data.kind === 'ema'
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={data.onSelect}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          data.onSelect?.()
        }
      }}
      className={cn(
        'w-56 rounded-lg border text-left shadow-sm transition-shadow',
        KIND_CLASS[data.kind],
        data.selected && 'ring-2 ring-primary shadow-md',
        data.onSelect && 'cursor-pointer hover:border-primary/50',
      )}
    >
      <div className={cn('flex items-center gap-1.5 rounded-t-lg border-b px-2 py-1.5', HEADER_CLASS[data.kind])}>
        {showLock && <Lock className="h-3 w-3 shrink-0 text-muted-foreground" />}
        <span className="min-w-0 flex-1 truncate text-[11px] font-semibold">{data.title}</span>
        {data.params != null && (
          <span className="shrink-0 font-mono text-[9px] text-muted-foreground">{formatNumber(data.params)}</span>
        )}
      </div>
      {(data.subtitle || data.detail) && (
        <div className="space-y-0.5 px-2 py-1.5">
          {data.subtitle && (
            <div className="font-mono text-[10px] leading-tight text-muted-foreground">{data.subtitle}</div>
          )}
          {data.detail && (
            <div className="truncate font-mono text-[9px] text-muted-foreground/80">{data.detail}</div>
          )}
        </div>
      )}
      {data.hasHandleIn !== false && <Handle type="target" position={Position.Top} className="!bg-primary" />}
      {data.hasHandleOut !== false && <Handle type="source" position={Position.Bottom} className="!bg-primary" />}
    </div>
  )
}, (prev, next) => prev.data === next.data)

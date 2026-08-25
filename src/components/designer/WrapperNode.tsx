import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Layers, X } from 'lucide-react'
import { Select } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import type { WrapperSpec } from '@/api/types'

export interface WrapperNodeData {
  catalog: WrapperSpec[]
  type: string
  onChange: (type: string) => void
  onRemove: () => void
}

// See AlgorithmNode.tsx for why this is memoized against `data` only —
// otherwise the node re-renders (and visibly flickers) on every drag frame.
export const WrapperNode = memo(function WrapperNode({ data }: NodeProps & { data: WrapperNodeData }) {
  const spec = data.catalog.find((w) => w.type === data.type)

  return (
    <div className="w-56 rounded-xl border border-border bg-card shadow-md">
      <div className="flex items-center gap-2 rounded-t-xl border-b border-border bg-muted px-3 py-2">
        <Layers className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Wrapper</span>
        {spec?.is_custom && <Badge variant="secondary" className="ml-auto text-[9px]">custom</Badge>}
        <Button variant="ghost" size="icon" className={`h-5 w-5 ${spec?.is_custom ? '' : 'ml-auto'}`} onClick={data.onRemove}>
          <X className="h-3 w-3" />
        </Button>
      </div>
      <div className="space-y-2 p-3">
        <Select
          value={data.type}
          onChange={(e) => data.onChange(e.target.value)}
          options={data.catalog.map((w) => ({ value: w.type, label: w.label }))}
        />
        {spec && Object.keys(spec.params).length > 0 && (
          <div className="space-y-1 text-[11px] text-muted-foreground">
            {Object.entries(spec.params).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span>{k}</span>
                <span className="font-mono">{String(v)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <Handle type="target" position={Position.Left} className="!bg-muted-foreground" />
      <Handle type="source" position={Position.Right} className="!bg-muted-foreground" />
    </div>
  )
}, (prev, next) => prev.data === next.data)

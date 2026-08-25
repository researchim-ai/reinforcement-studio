import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'

export interface AddNodeData {
  onAdd: () => void
}

/** Floating "+" button dropped into a gap of any of our ReactFlow chains
 * (Network Builder layers, Designer wrappers) so a new node can be
 * inserted at that exact spot instead of only ever being appended at the
 * end. Not part of the `edges` list — it's a UI control, not a graph node.
 *
 * Memoized against `data` only — see designer/AlgorithmNode.tsx for why
 * (otherwise it re-renders, and visibly flickers, on every drag frame of
 * any node on the canvas, not just its own). */
export const AddNode = memo(function AddNode({ data }: NodeProps & { data: AddNodeData }) {
  return (
    <div className="flex items-center justify-center">
      <Button variant="outline" size="icon" className="nodrag h-8 w-8 rounded-full border-dashed" onClick={data.onAdd}>
        <Plus className="h-4 w-4" />
      </Button>
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
    </div>
  )
}, (prev, next) => prev.data === next.data)

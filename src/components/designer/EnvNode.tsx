import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Boxes, LogIn, LogOut } from 'lucide-react'
import { Select } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { spaceSize } from '@/lib/spaceInfo'
import type { EnvSpec, SpaceInfo } from '@/api/types'

const CATEGORY_SHORT: Record<string, string> = {
  classic_control: 'Classic',
  toy_text: 'Toy',
  box2d: 'Box2D',
  mujoco: 'MuJoCo',
  atari: 'Atari',
  pomdp: 'POMDP',
  board_game: 'Board',
}

export interface EnvNodeData {
  environments: EnvSpec[]
  selectedId: string
  // Live-inspected (env + wrappers applied, no algorithm/network needed —
  // see rl_core/inspect.py::_io_shapes) — undefined while the inspect
  // request is still in flight or hasn't run yet for this selection.
  observationSpace?: SpaceInfo | null
  actionSpace?: SpaceInfo | null
  onChange: (id: string) => void
  // Set while "Дообучить" is active — the environment (and its observation/
  // action space) can't change once weights are loaded into a fixed-shape
  // network, see AlgorithmNodeData's `locked`.
  disabled?: boolean
}

// See AlgorithmNode.tsx for why this is memoized against `data` only —
// otherwise the node re-renders (and visibly flickers) on every drag frame.
export const EnvNode = memo(function EnvNode({ data }: NodeProps & { data: EnvNodeData }) {
  const selected = data.environments.find((e) => e.id === data.selectedId)

  return (
    <div className="w-64 rounded-xl border border-border bg-card shadow-md">
      <div className="flex items-center gap-2 rounded-t-xl border-b border-border bg-primary/10 px-3 py-2">
        <Boxes className="h-4 w-4 text-primary" />
        <span className="text-xs font-semibold uppercase tracking-wide text-primary">Environment</span>
      </div>
      <div className="space-y-2 p-3">
        <Select
          value={data.selectedId}
          onChange={(e) => data.onChange(e.target.value)}
          options={data.environments.map((e) => ({
            value: e.id,
            label: `${CATEGORY_SHORT[e.category] ?? e.category} · ${e.name}${e.available ? '' : ' (недоступна)'}`,
          }))}
          disabled={data.disabled}
        />
        {selected && (
          <>
            <p className="text-xs text-muted-foreground">{selected.description}</p>
            <div className="flex gap-1">
              <Badge variant="outline" className="text-[10px]">{selected.category}</Badge>
              <Badge variant="outline" className="text-[10px]">{selected.action_kind}</Badge>
            </div>
            <div className="flex items-center justify-between gap-2 rounded-md border border-border/60 bg-background/40 px-2 py-1.5 text-[11px]">
              <span className="flex items-center gap-1 text-muted-foreground">
                <LogIn className="h-3 w-3" /> Входов
              </span>
              <span className="font-mono font-semibold">{spaceSize(data.observationSpace) ?? '—'}</span>
              <span className="mx-1 h-3 w-px bg-border" />
              <span className="flex items-center gap-1 text-muted-foreground">
                <LogOut className="h-3 w-3" /> Выходов
              </span>
              <span className="font-mono font-semibold">{spaceSize(data.actionSpace) ?? '—'}</span>
            </div>
          </>
        )}
      </div>
      <Handle type="source" position={Position.Right} className="!bg-primary" />
    </div>
  )
}, (prev, next) => prev.data === next.data)

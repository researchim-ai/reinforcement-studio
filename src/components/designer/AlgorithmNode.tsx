import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Link } from 'react-router-dom'
import { Cpu, Network } from 'lucide-react'
import { Select } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { AlgorithmDiagram } from '@/components/AlgorithmDiagram'
import { FAMILY_LABELS } from '@/lib/networkBuilder'
import type { AlgorithmSpec, NetworkFamily, NetworkMeta } from '@/api/types'

export interface AlgorithmNodeData {
  algorithms: AlgorithmSpec[]
  selectedId: string
  hyperparams: Record<string, number>
  networks: NetworkMeta[]
  networkSpecId: string | null
  onChangeAlgo: (id: string) => void
  onChangeHyperparam: (key: string, value: number) => void
  onChangeNetworkSpecId: (id: string | null) => void
}

/** Only built-in algorithms know how to accept a hand-designed spec net
 * (see rl_core/algorithms/native/{ppo,a2c,dqn}.py and
 * rl_core/alphazero/base.py::build_network) — custom plugins ignore
 * `network_spec` unless they opt in themselves, so hide the picker for
 * them rather than implying it does something it doesn't. */
function requiredFamilyFor(algorithmId: string, kind: 'gym' | 'alphazero'): NetworkFamily | null {
  if (kind === 'alphazero') return algorithmId === 'alphazero' ? 'alphazero' : null
  if (algorithmId === 'dqn') return 'q_network'
  if (algorithmId === 'rainbow_dqn') return 'dueling_q'
  if (algorithmId === 'ppo' || algorithmId === 'a2c') return 'actor_critic'
  return null
}

/** React Flow re-renders a node's component on every position update while
 * it's being dragged (it passes positionAbsoluteX/Y/dragging as props), even
 * though none of that affects what we render here — the actual on-screen
 * placement is handled by the wrapper div ReactFlow renders around us. Left
 * unmemoized, that meant this fairly heavy node (selects, diagram) fully
 * re-rendered 60 times/sec while being dragged, which showed up as visible
 * flicker/jank. Comparing only on `data` skips all of that. */
// Memory sub-hyperparams (hidden size / layers / BPTT length) only matter
// once `memory_type` is actually enabled — hiding them otherwise avoids
// cluttering the panel with knobs that don't do anything yet. A
// hand-designed network (Network Builder) doesn't support memory at all
// (see rl_core/algorithms/native/{ppo,a2c,dqn,rainbow_dqn}.py — the
// `network_spec` branch takes priority over `memory_type` in every one of
// them), so the whole group is hidden then too, with a note near the
// network picker instead of a confusingly-inert one.
function conditionsMatch(
  conditions: NonNullable<AlgorithmSpec['hyperparams'][number]['visibleWhen']>,
  values: Record<string, number>,
): boolean {
  return conditions.every((condition) => {
    const value = Number(values[condition.key] ?? 0)
    if (condition.eq != null && value !== condition.eq) return false
    if (condition.gte != null && value < condition.gte) return false
    if (condition.lte != null && value > condition.lte) return false
    return true
  })
}

export const AlgorithmNode = memo(function AlgorithmNode({ data }: NodeProps & { data: AlgorithmNodeData }) {
  const selected = data.algorithms.find((a) => a.id === data.selectedId)
  const requiredFamily = selected ? requiredFamilyFor(selected.id, selected.kind) : null
  const compatibleNetworks = requiredFamily ? data.networks.filter((n) => n.family === requiredFamily && !n.broken) : []
  const hasCustomNetwork = requiredFamily != null && data.networkSpecId != null
  const visibleHyperparams = selected?.hyperparams.filter((hp) => {
    if (hasCustomNetwork && (hp.key === 'memory_type' || hp.key.startsWith('memory_'))) return false
    if (hp.visibleWhen && !conditionsMatch(hp.visibleWhen, data.hyperparams)) return false
    return true
  })

  return (
    <div className="w-80 rounded-xl border border-border bg-card shadow-md">
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
              {visibleHyperparams!.map((hp) => (
                <div key={hp.key} className="space-y-1">
                  <div className="flex items-baseline justify-between gap-2">
                    <Label className="text-[11px] text-muted-foreground">{hp.label}</Label>
                    {!hp.options && (hp.min != null || hp.max != null) && (
                      <span className="text-[10px] text-muted-foreground/60">
                        {hp.min ?? '−∞'}–{hp.max ?? '∞'}
                      </span>
                    )}
                  </div>
                  {hp.options ? (
                    <Select
                      value={String(data.hyperparams[hp.key] ?? hp.default)}
                      onChange={(e) => data.onChangeHyperparam(hp.key, Number(e.target.value))}
                      options={hp.options
                        .filter((opt) => !(hasCustomNetwork && hp.key === 'action_exploration' && opt.value === 1))
                        .map((opt) => ({ value: String(opt.value), label: opt.label }))}
                      className="h-7 text-xs"
                    />
                  ) : (
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
                  )}
                </div>
              ))}
            </div>
            {requiredFamily && (
              <div className="space-y-1 border-t border-border pt-2">
                <div className="flex items-baseline justify-between gap-2">
                  <Label className="text-[11px] text-muted-foreground">Архитектура сети</Label>
                  <Link to="/network-builder" className="flex items-center gap-1 text-[10px] text-primary hover:underline">
                    <Network className="h-2.5 w-2.5" /> своя
                  </Link>
                </div>
                <Select
                  value={data.networkSpecId ?? ''}
                  onChange={(e) => data.onChangeNetworkSpecId(e.target.value || null)}
                  options={[
                    { value: '', label: `Стандартная (${FAMILY_LABELS[requiredFamily]})` },
                    ...compatibleNetworks.map((n) => ({ value: n.slug, label: n.name })),
                  ]}
                  className="h-7 text-xs"
                />
                {hasCustomNetwork && (
                  <p className="text-[10px] italic text-muted-foreground">
                    Своя архитектура не поддерживает встроенную память и NoisyNet. RND совместим, поскольку работает отдельно от Q-сети.
                  </p>
                )}
              </div>
            )}
            <div className="border-t border-border pt-2">
              <AlgorithmDiagram
                algorithmId={selected.id}
                kind={selected.kind}
                hyperparams={data.hyperparams}
                show={{ loop: true, network: false }}
              />
            </div>
          </>
        )}
      </div>
      <Handle type="target" position={Position.Left} className="!bg-primary" />
      <Handle type="source" position={Position.Right} className="!bg-primary" />
    </div>
  )
}, (prev, next) => prev.data === next.data)

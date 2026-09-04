import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Link } from 'react-router-dom'
import { Brain, Cpu, Network } from 'lucide-react'
import { Select } from '@/components/ui/select'
import { NumericInput } from '@/components/ui/numeric-input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Tooltip } from '@/components/ui/tooltip'
import { AlgorithmDiagram } from '@/components/AlgorithmDiagram'
import { QuickNetworkEditor } from '@/components/designer/QuickNetworkEditor'
import {
  FAMILY_LABELS,
  defaultHiddenSizesForFamily,
  isCompositeFamily,
  networkFamilyFor,
  requiredFamilyFor,
} from '@/lib/networkBuilder'
import { WORLD_MODEL_TYPE_LABELS } from '@/lib/worldModels'
import { useSettingsStore } from '@/stores/settingsStore'
import type { AlgorithmSpec, NetworkMeta, WorldModelMeta } from '@/api/types'

export interface AlgorithmNodeData {
  algorithms: AlgorithmSpec[]
  selectedId: string
  hyperparams: Record<string, number>
  networks: NetworkMeta[]
  networkSpecId: string | null
  quickHiddenLayers: number[] | null
  worldModels: WorldModelMeta[]
  worldModelId: string | null
  onChangeAlgo: (id: string) => void
  onChangeHyperparam: (key: string, value: number) => void
  onChangeNetworkSpecId: (id: string | null) => void
  onChangeQuickLayers: (layers: number[] | null) => void
  onChangeWorldModelId: (id: string | null) => void
  // Set while "Дообучить" (resume/fine-tune — see ExperimentDesigner.tsx's
  // `resumeFrom` state) is active: the architecture/hyperparams are fixed
  // by the source run/checkpoint's saved weights (loading a state_dict
  // into a differently-shaped network fails outright), so every control
  // here goes read-only instead of just being pre-filled.
  locked?: boolean
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
  const language = useSettingsStore((s) => s.language)
  const selected = data.algorithms.find((a) => a.id === data.selectedId)
  const requiredFamily = selected ? networkFamilyFor(selected.id, selected.kind) : null
  // Narrower than `requiredFamily` — excludes `alphazero` (which already has
  // its own quick architecture knobs, `channels`/`num_blocks`, right in the
  // hyperparams list above) since the quick *layer* editor only understands
  // plain MLP trunks.
  const quickFamily = selected ? requiredFamilyFor(selected.id, selected.kind) : null
  const compatibleNetworks = requiredFamily ? data.networks.filter((n) => n.family === requiredFamily && !n.broken) : []
  const requiredWorldModelType = selected?.world_model_type ?? null
  const compatibleWorldModels = requiredWorldModelType
    ? data.worldModels.filter((w) => w.type === requiredWorldModelType && !w.broken)
    : []
  const hasSavedNetwork = requiredFamily != null && data.networkSpecId != null
  const hasCustomNetwork = hasSavedNetwork || (quickFamily != null && data.quickHiddenLayers != null)
  const visibleHyperparams = selected?.hyperparams.filter((hp) => {
    if (hasCustomNetwork && (hp.key === 'memory_type' || hp.key.startsWith('memory_'))) return false
    if (
      hasSavedNetwork
      && requiredFamily
      && isCompositeFamily(requiredFamily)
      && [
        'latent_dim', 'hidden_dim', 'proj_dim', 'embed_dim', 'num_layers',
        'num_heads', 'dropout', 'rotary_emb',
      ].includes(hp.key)
    ) return false
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
          disabled={data.locked}
        />
        {data.locked && (
          <p className="rounded-md bg-primary/10 px-2 py-1.5 text-[10px] text-primary">
            Дообучение: алгоритм, гиперпараметры и архитектура сети наследуются от исходной модели
            и не могут быть изменены.
          </p>
        )}
        {selected && (
          <>
            <p className="text-xs text-muted-foreground">{selected.description}</p>
            <div className="max-h-56 space-y-2 overflow-auto pr-1">
              {visibleHyperparams!.map((hp) => (
                <div key={hp.key} className="space-y-1">
                  <div className="flex items-baseline justify-between gap-2">
                    {hp.desc ? (
                      <Tooltip content={hp.desc[language]} side="right">
                        <Label className="cursor-help text-[11px] text-muted-foreground underline decoration-dotted decoration-muted-foreground/50 underline-offset-2">
                          {hp.label}
                        </Label>
                      </Tooltip>
                    ) : (
                      <Label className="text-[11px] text-muted-foreground">{hp.label}</Label>
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
                      disabled={data.locked}
                    />
                  ) : (
                    <NumericInput
                      integer={hp.type === 'int'}
                      value={data.hyperparams[hp.key] ?? hp.default}
                      onChange={(v) => data.onChangeHyperparam(hp.key, v)}
                      className="h-7 text-xs"
                      disabled={data.locked}
                    />
                  )}
                </div>
              ))}
            </div>
            {requiredFamily && !data.locked && (
              <div className="space-y-1 border-t border-border pt-2">
                <div className="flex items-baseline justify-between gap-2">
                  <Label className="text-[11px] text-muted-foreground">Архитектура сети</Label>
                  <Link to={`/network-builder?family=${requiredFamily}`} className="flex items-center gap-1 text-[10px] text-primary hover:underline">
                    <Network className="h-2.5 w-2.5" /> своя
                  </Link>
                </div>
                <Select
                  value={data.networkSpecId ?? ''}
                  onChange={(e) => data.onChangeNetworkSpecId(e.target.value || null)}
                  options={[
                    {
                      value: '',
                      label: data.quickHiddenLayers != null
                        ? `Быстрая настройка (${FAMILY_LABELS[requiredFamily]})`
                        : `Стандартная (${FAMILY_LABELS[requiredFamily]})`,
                    },
                    ...compatibleNetworks.map((n) => ({ value: n.slug, label: n.name })),
                  ]}
                  className="h-7 text-xs"
                />
                {hasCustomNetwork && (
                  <p className="text-[10px] italic text-muted-foreground">
                    {requiredFamily && isCompositeFamily(requiredFamily)
                      ? 'Размерности и компоненты берутся из сохранённой composite-архитектуры; выходы автоматически согласованы со средой.'
                      : 'Своя архитектура не поддерживает встроенную память и NoisyNet. RND совместим, поскольку работает отдельно от Q-сети.'}
                  </p>
                )}
              </div>
            )}
            {quickFamily && !hasSavedNetwork && !data.locked && (
              <QuickNetworkEditor
                defaultSizes={defaultHiddenSizesForFamily(quickFamily)}
                hiddenLayers={data.quickHiddenLayers}
                onChange={data.onChangeQuickLayers}
              />
            )}
            {requiredWorldModelType && !data.locked && (
              <div className="space-y-1 border-t border-border pt-2">
                <div className="flex items-baseline justify-between gap-2">
                  <Label className="text-[11px] text-muted-foreground">World Model ({WORLD_MODEL_TYPE_LABELS[requiredWorldModelType]})</Label>
                  <Link to="/world-models" className="flex items-center gap-1 text-[10px] text-primary hover:underline">
                    <Brain className="h-2.5 w-2.5" /> создать
                  </Link>
                </div>
                <Select
                  value={data.worldModelId ?? ''}
                  onChange={(e) => data.onChangeWorldModelId(e.target.value || null)}
                  options={[
                    { value: '', label: 'Свежая модель (стандартная конфигурация)' },
                    ...compatibleWorldModels.map((w) => ({
                      value: w.slug,
                      label: `${w.name}${w.trained ? ' ✓' : ''}`,
                    })),
                  ]}
                  className="h-7 text-xs"
                />
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

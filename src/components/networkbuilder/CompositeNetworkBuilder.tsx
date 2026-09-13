import { useCallback, useEffect, useMemo, useState } from 'react'
import { Background, Controls, ReactFlow } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { ArrowDown, ArrowUp, Plus, Trash2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select } from '@/components/ui/select'
import { CompositeGraphNode } from '@/components/networkbuilder/CompositeGraphNode'
import { useNodePositions } from '@/lib/useNodePositions'
import { buildCompositeGraph, selectableGraphNode } from '@/lib/compositeGraph'
import type {
  CompositeEncoderSpec,
  CompositeMlpSpec,
  CompositeNetworkSpec,
  NetworkLayer,
  NetworkLayerType,
  NetworkPreviewResult,
} from '@/api/types'
import { ACTIVATION_LABELS, LAYER_LABELS, defaultLayer } from '@/lib/networkBuilder'

interface Props {
  spec: CompositeNetworkSpec
  preview?: NetworkPreviewResult
  onChange: (spec: CompositeNetworkSpec) => void
}

const nodeTypes = { composite: CompositeGraphNode }

const DIMENSION_LABELS: Record<string, string> = {
  latent_dim: 'Размер latent',
  hidden_dim: 'Скрытый размер',
  proj_dim: 'Размер SimSiam projection',
  embed_dim: 'Размер token embedding',
  num_layers: 'Transformer-блоков',
  num_heads: 'Attention heads',
  ffn_multiplier: 'FFN multiplier',
  dropout: 'Transformer dropout',
  rotary_emb: 'RoPE',
  stoch_variables: 'Категориальных переменных',
  stoch_classes: 'Классов на переменную',
}

const COMPONENT_LABELS: Record<string, string> = {
  representation: 'Representation projection',
  dynamics: 'Dynamics trunk',
  prediction: 'Prediction trunk',
  tokenizer: 'Tokenizer projection',
  heads: 'Reward / Value / Policy trunk',
  projector: 'SimSiam projector',
  predictor: 'SimSiam predictor',
  stochastic_prior: 'Stochastic prior',
  stochastic_posterior: 'Observation posterior',
  state_feature: 'Deterministic + stochastic state feature',
  reward: 'Symlog reward head',
  continue: 'Continue head',
  actor: 'Imagination actor / search prior',
  critic: 'Online / EMA critic',
  ensemble_prior: 'Prior ensemble',
  ensemble_reward: 'Reward ensemble',
}

function encoderTemplate(kind: CompositeEncoderSpec['kind']): NetworkLayer[] {
  if (kind === 'vector_mlp') {
    return [
      { type: 'linear', out_features: 128 },
      { type: 'activation', fn: 'relu' },
    ]
  }
  if (kind === 'image_cnn') {
    return [
      { type: 'conv2d', out_channels: 32, kernel_size: 3, stride: 2, padding: 1 },
      { type: 'activation', fn: 'relu' },
      { type: 'conv2d', out_channels: 64, kernel_size: 3, stride: 2, padding: 1 },
      { type: 'activation', fn: 'relu' },
      { type: 'flatten' },
    ]
  }
  if (kind === 'custom') {
    return [
      { type: 'flatten' },
      { type: 'linear', out_features: 128 },
      { type: 'activation', fn: 'relu' },
    ]
  }
  return []
}

function LayerFields({
  layer,
  onChange,
}: {
  layer: NetworkLayer
  onChange: (layer: NetworkLayer) => void
}) {
  const numberField = (key: keyof NetworkLayer, label: string, min = 0) => (
    <label className="space-y-1 text-[10px] text-muted-foreground">
      <span>{label}</span>
      <Input
        type="number"
        min={min}
        value={String(layer[key] ?? '')}
        onChange={(event) => onChange({ ...layer, [key]: Number(event.target.value) })}
        className="h-7 text-xs"
      />
    </label>
  )
  return (
    <div className="grid flex-1 grid-cols-2 gap-2">
      {layer.type === 'linear' && numberField('out_features', 'Выход', 1)}
      {layer.type === 'conv2d' && (
        <>
          {numberField('out_channels', 'Каналы', 1)}
          {numberField('kernel_size', 'Kernel', 1)}
          {numberField('stride', 'Stride', 1)}
          {numberField('padding', 'Padding', 0)}
        </>
      )}
      {layer.type === 'maxpool2d' && (
        <>
          {numberField('kernel_size', 'Kernel', 1)}
          {numberField('stride', 'Stride', 1)}
        </>
      )}
      {layer.type === 'activation' && (
        <label className="space-y-1 text-[10px] text-muted-foreground">
          <span>Функция</span>
          <Select
            value={layer.fn ?? 'relu'}
            onChange={(event) => onChange({ ...layer, fn: event.target.value as NetworkLayer['fn'] })}
            options={Object.entries(ACTIVATION_LABELS).map(([value, label]) => ({ value, label }))}
            className="h-7 text-xs"
          />
        </label>
      )}
      {layer.type === 'dropout' && numberField('p', 'Вероятность', 0)}
    </div>
  )
}

function EncoderEditor({
  encoder,
  onChange,
}: {
  encoder: CompositeEncoderSpec
  onChange: (encoder: CompositeEncoderSpec) => void
}) {
  const setKind = (kind: CompositeEncoderSpec['kind']) => {
    onChange({ kind, layers: kind === 'auto' ? [] : encoderTemplate(kind) })
  }
  const updateLayer = (index: number, layer: NetworkLayer) => {
    onChange({ ...encoder, layers: encoder.layers.map((item, i) => (i === index ? layer : item)) })
  }
  const moveLayer = (index: number, direction: -1 | 1) => {
    const target = index + direction
    if (target < 0 || target >= encoder.layers.length) return
    const layers = [...encoder.layers]
    ;[layers[index], layers[target]] = [layers[target], layers[index]]
    onChange({ ...encoder, layers })
  }
  return (
    <div className="space-y-3">
      <Select
        value={encoder.kind}
        onChange={(event) => setKind(event.target.value as CompositeEncoderSpec['kind'])}
        options={[
          { value: 'auto', label: 'Auto — проверенный CNN/MLP по типу среды' },
          { value: 'vector_mlp', label: 'Vector MLP' },
          { value: 'image_cnn', label: 'Image CNN' },
          { value: 'custom', label: 'Custom chain' },
        ]}
      />
      {encoder.kind !== 'auto' && (
        <div className="space-y-2">
          {encoder.layers.map((layer, index) => (
            <div key={index} className="flex items-start gap-2 rounded-md border p-2">
              <Select
                value={layer.type}
                onChange={(event) => updateLayer(index, defaultLayer(event.target.value as NetworkLayerType))}
                options={Object.entries(LAYER_LABELS)
                  .filter(([value]) => value !== 'batchnorm')
                  .map(([value, label]) => ({ value, label }))}
                className="h-7 w-28 text-xs"
              />
              <LayerFields layer={layer} onChange={(next) => updateLayer(index, next)} />
              <div className="flex flex-col">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-5 w-7"
                  disabled={index === 0}
                  onClick={() => moveLayer(index, -1)}
                >
                  <ArrowUp className="h-3 w-3" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-5 w-7"
                  disabled={index === encoder.layers.length - 1}
                  onClick={() => moveLayer(index, 1)}
                >
                  <ArrowDown className="h-3 w-3" />
                </Button>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7 text-destructive"
                onClick={() => onChange({ ...encoder, layers: encoder.layers.filter((_, i) => i !== index) })}
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </div>
          ))}
          <Button
            variant="outline"
            size="sm"
            onClick={() => onChange({ ...encoder, layers: [...encoder.layers, defaultLayer('linear')] })}
          >
            <Plus className="h-3.5 w-3.5" /> Добавить слой
          </Button>
        </div>
      )}
    </div>
  )
}

function formatHiddenSizes(sizes: number[]): string {
  return sizes.join(', ')
}

/** Parses a live "128, 64," draft. Returns null while a token is incomplete
 * (trailing comma, empty slot, non-integer) so the input can keep the comma
 * instead of snapping back to the last committed list. */
export function parseHiddenSizesDraft(text: string): number[] | null {
  const trimmed = text.trim()
  if (trimmed === '') return []
  if (/[^\d,\s]/.test(text) || /,\s*,/.test(text) || /,\s*$/.test(text)) return null
  const sizes: number[] = []
  for (const token of text.split(',').map((part) => part.trim()).filter(Boolean)) {
    if (!/^\d+$/.test(token)) return null
    const size = Number(token)
    if (!Number.isFinite(size) || size <= 0) return null
    sizes.push(size)
  }
  return sizes
}

function HiddenSizesInput({
  value,
  onChange,
}: {
  value: number[]
  onChange: (value: number[]) => void
}) {
  const [focused, setFocused] = useState(false)
  const [draft, setDraft] = useState(() => formatHiddenSizes(value))

  useEffect(() => {
    if (!focused) setDraft(formatHiddenSizes(value))
  }, [value, focused])

  return (
    <Input
      value={focused ? draft : formatHiddenSizes(value)}
      onFocus={() => {
        setFocused(true)
        setDraft(formatHiddenSizes(value))
      }}
      onChange={(event) => {
        const next = event.target.value
        if (/[^\d,\s]/.test(next)) return
        setDraft(next)
        const parsed = parseHiddenSizesDraft(next)
        if (parsed !== null) onChange(parsed)
      }}
      onBlur={() => {
        setFocused(false)
        const parsed = parseHiddenSizesDraft(draft)
        if (parsed !== null) {
          onChange(parsed)
          setDraft(formatHiddenSizes(parsed))
        } else {
          const committed = parseHiddenSizesDraft(draft.replace(/,+\s*$/, ''))
          if (committed !== null) {
            onChange(committed)
            setDraft(formatHiddenSizes(committed))
          } else {
            setDraft(formatHiddenSizes(value))
          }
        }
      }}
      className="h-8 text-xs"
      placeholder="128, 128"
    />
  )
}

function MlpEditor({
  name,
  config,
  onChange,
}: {
  name: string
  config: CompositeMlpSpec
  onChange: (config: CompositeMlpSpec) => void
}) {
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-1 gap-2">
        <label className="space-y-1">
          <Label className="text-[10px]">Hidden sizes через запятую</Label>
          <HiddenSizesInput
            value={config.hidden_sizes}
            onChange={(hidden_sizes) => onChange({ ...config, hidden_sizes })}
          />
        </label>
        <label className="space-y-1">
          <Label className="text-[10px]">Активация</Label>
          <Select
            value={config.activation}
            onChange={(event) => onChange({ ...config, activation: event.target.value as CompositeMlpSpec['activation'] })}
            options={[
              ...Object.entries(ACTIVATION_LABELS).map(([value, label]) => ({ value, label })),
              { value: 'elu', label: 'ELU' },
            ]}
            className="h-8 text-xs"
          />
        </label>
        <label className="space-y-1">
          <Label className="text-[10px]">Dropout</Label>
          <Input
            type="number"
            min={0}
            max={0.99}
            step={0.05}
            value={config.dropout}
            onChange={(event) => onChange({ ...config, dropout: Number(event.target.value) })}
            className="h-8 text-xs"
          />
        </label>
        {(name === 'projector' || name === 'predictor') && (
          <label className="flex items-center gap-2 pb-1 text-xs">
            <input type="checkbox" checked disabled />
            BatchNorm (обязательно)
          </label>
        )}
      </div>
    </div>
  )
}

function DimensionFields({
  spec,
  keys,
  onChange,
}: {
  spec: CompositeNetworkSpec
  keys: string[]
  onChange: (spec: CompositeNetworkSpec) => void
}) {
  return (
    <div className="grid grid-cols-1 gap-2">
      {keys.filter((key) => key in spec.dimensions).map((key) => (
        <label key={key} className="space-y-1">
          <Label className="text-[10px]">{DIMENSION_LABELS[key] ?? key}</Label>
          {key === 'rotary_emb' ? (
            <Select
              value={String(spec.dimensions[key])}
              onChange={(event) => onChange({
                ...spec,
                dimensions: { ...spec.dimensions, [key]: Number(event.target.value) },
              })}
              options={[{ value: '1', label: 'Включён' }, { value: '0', label: 'Выключен' }]}
              className="h-8 text-xs"
            />
          ) : (
            <Input
              type="number"
              min={key === 'dropout' ? 0 : 1}
              max={key === 'dropout' ? 0.99 : undefined}
              step={key === 'dropout' ? 0.05 : 1}
              value={spec.dimensions[key]}
              onChange={(event) => onChange({
                ...spec,
                dimensions: { ...spec.dimensions, [key]: Number(event.target.value) },
              })}
              className="h-8 text-xs"
            />
          )}
        </label>
      ))}
    </div>
  )
}

function Inspector({
  spec,
  selectedId,
  onChange,
}: {
  spec: CompositeNetworkSpec
  selectedId: string | null
  onChange: (spec: CompositeNetworkSpec) => void
}) {
  const selected = selectedId ? selectableGraphNode(selectedId, spec.family) : undefined
  if (!selected) {
    return (
      <div className="space-y-3 p-4">
        <div>
          <h3 className="text-sm font-semibold">Каркас алгоритма</h3>
          <p className="mt-1 text-[11px] text-muted-foreground">
            Кликните блок на графе, чтобы настроить его. Связи, search-протокол и размеры выходов
            зафиксированы — меняются только безопасные внутренние блоки.
          </p>
        </div>
        <DimensionFields spec={spec} keys={Object.keys(spec.dimensions)} onChange={onChange} />
      </div>
    )
  }

  if (selected.kind === 'encoder') {
    return (
      <div className="space-y-3 p-4">
        <div>
          <h3 className="text-sm font-semibold">Входной энкодер</h3>
          <p className="mt-1 text-[11px] text-muted-foreground">
            Вход задаёт среда. Здесь — преобразование изображения или вектора в признаки.
          </p>
        </div>
        <EncoderEditor encoder={spec.encoder} onChange={(encoder) => onChange({ ...spec, encoder })} />
      </div>
    )
  }

  if (selected.componentKey && spec.components[selected.componentKey]) {
    return (
      <div className="space-y-3 p-4">
        <div>
          <h3 className="text-sm font-semibold">{COMPONENT_LABELS[selected.componentKey] ?? selected.title}</h3>
          <p className="mt-1 text-[11px] text-muted-foreground">Внутренний MLP этого блока. Выход алгоритма по-прежнему auto.</p>
        </div>
        {selected.dimKeys && selected.dimKeys.length > 0 && (
          <DimensionFields spec={spec} keys={selected.dimKeys} onChange={onChange} />
        )}
        <MlpEditor
          name={selected.componentKey}
          config={spec.components[selected.componentKey]}
          onChange={(next) => onChange({
            ...spec,
            components: { ...spec.components, [selected.componentKey!]: next },
          })}
        />
      </div>
    )
  }

  if (selected.dimKeys && selected.dimKeys.length > 0) {
    return (
      <div className="space-y-3 p-4">
        <div>
          <h3 className="text-sm font-semibold">{selected.title}</h3>
          <p className="mt-1 text-[11px] text-muted-foreground">Размерности этого блока. Топология графа не меняется.</p>
        </div>
        <DimensionFields spec={spec} keys={selected.dimKeys} onChange={onChange} />
      </div>
    )
  }

  return (
    <div className="space-y-3 p-4">
      <div>
        <h3 className="text-sm font-semibold">{selected.title}</h3>
        <p className="mt-1 text-[11px] text-muted-foreground">
          Этот блок зафиксирован алгоритмом: его выход и связи нельзя разобрать, не сломав search/train.
        </p>
      </div>
    </div>
  )
}

export function CompositeNetworkBuilder({ spec, preview, onChange }: Props) {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const onSelect = useCallback((id: string) => {
    setSelectedId((current) => (current === id ? null : id))
  }, [])
  const { nodes: rawNodes, edges } = useMemo(
    () => buildCompositeGraph(spec, preview, selectedId, onSelect),
    [spec, preview, selectedId, onSelect],
  )
  const { applyPositions, onNodesChange } = useNodePositions()
  const nodes = applyPositions(rawNodes)

  return (
    <div className="flex h-full min-h-0">
      <div className="min-w-0 flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onPaneClick={() => setSelectedId(null)}
          nodesConnectable={false}
          elementsSelectable={false}
          minZoom={0.25}
          fitView
          fitViewOptions={{ padding: 0.25 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
      <aside className="w-80 shrink-0 overflow-auto border-l border-border bg-card">
        <Inspector spec={spec} selectedId={selectedId} onChange={onChange} />
        {preview?.fixed_outputs && (
          <div className="border-t border-border p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Фиксированные выходы</h3>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {Object.entries(preview.fixed_outputs).map(([name, shape]) => (
                <span key={name} className="rounded-md bg-muted px-2 py-1 font-mono text-[10px]">
                  {name} → [{shape.join('×')}]
                </span>
              ))}
            </div>
          </div>
        )}
      </aside>
    </div>
  )
}

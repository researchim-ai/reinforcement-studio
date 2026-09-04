import { ArrowDown, ArrowRight, ArrowUp, Plus, Trash2 } from 'lucide-react'

import { AlgorithmDiagram } from '@/components/AlgorithmDiagram'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select } from '@/components/ui/select'
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
}

const COMPONENT_LABELS: Record<string, string> = {
  representation: 'Representation projection',
  dynamics: 'Dynamics trunk',
  prediction: 'Prediction trunk',
  tokenizer: 'Tokenizer projection',
  heads: 'Reward / Value / Policy trunk',
  projector: 'SimSiam projector',
  predictor: 'SimSiam predictor',
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
    <div className="grid flex-1 grid-cols-2 gap-2 lg:grid-cols-4">
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
    <section className="space-y-3 rounded-lg border bg-card p-3">
      <div>
        <h3 className="text-sm font-semibold">Входной энкодер</h3>
        <p className="text-[11px] text-muted-foreground">
          Вход задаёт среда. Здесь настраивается преобразование изображения/вектора в latent-признаки.
        </p>
      </div>
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
                className="h-7 w-32 text-xs"
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
    </section>
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
    <section className="space-y-2 rounded-lg border bg-card p-3">
      <h3 className="text-sm font-semibold">{COMPONENT_LABELS[name] ?? name}</h3>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-4">
        <label className="space-y-1">
          <Label className="text-[10px]">Hidden sizes через запятую</Label>
          <Input
            value={config.hidden_sizes.join(', ')}
            onChange={(event) => onChange({
              ...config,
              hidden_sizes: event.target.value
                .split(',')
                .map((value) => Number(value.trim()))
                .filter((value) => Number.isFinite(value) && value > 0),
            })}
            className="h-8 text-xs"
            placeholder="128, 128"
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
          <label className="flex items-end gap-2 pb-1 text-xs">
            <input type="checkbox" checked disabled />
            BatchNorm (обязательно)
          </label>
        )}
      </div>
    </section>
  )
}

export function CompositeNetworkBuilder({ spec, preview, onChange }: Props) {
  const skeleton = spec.family === 'efficientzero'
    ? ['Observation encoder', 'Representation', 'Dynamics + value-prefix LSTM', 'Prediction heads']
    : ['Observation tokenizer + Action embedding', 'Causal Transformer + KV-cache', 'Reward / Value / Policy / Latent heads', 'EMA target']
  return (
    <div className="h-full overflow-auto p-4">
      <div className="mx-auto max-w-6xl space-y-4">
        <section className="rounded-lg border border-dashed bg-muted/20 p-3">
          <h3 className="mb-2 text-sm font-semibold">Фиксированный корректный каркас</h3>
          <div className="flex flex-wrap items-center gap-2">
            {skeleton.map((label, index) => (
              <div key={label} className="contents">
                <span className="rounded-md border bg-background px-2 py-1 text-[11px]">{label}</span>
                {index < skeleton.length - 1 && <ArrowRight className="h-3.5 w-3.5 text-muted-foreground" />}
              </div>
            ))}
          </div>
          <p className="mt-2 text-[10px] text-muted-foreground">
            Связи, search-протокол и auto-sized выходы защищены от редактирования; меняются только безопасные внутренние блоки.
          </p>
        </section>
        <section className="rounded-lg border bg-card p-3">
          <h3 className="mb-3 text-sm font-semibold">Размерности каркаса</h3>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            {Object.entries(spec.dimensions).map(([key, value]) => (
              <label key={key} className="space-y-1">
                <Label className="text-[10px]">{DIMENSION_LABELS[key] ?? key}</Label>
                {key === 'rotary_emb' ? (
                  <Select
                    value={String(value)}
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
                    value={value}
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
        </section>

        <EncoderEditor encoder={spec.encoder} onChange={(encoder) => onChange({ ...spec, encoder })} />
        {Object.entries(spec.components).map(([name, config]) => (
          <MlpEditor
            key={name}
            name={name}
            config={config}
            onChange={(next) => onChange({ ...spec, components: { ...spec.components, [name]: next } })}
          />
        ))}

        {preview?.fixed_outputs && (
          <section className="rounded-lg border border-dashed p-3">
            <h3 className="text-sm font-semibold">Фиксированные выходы алгоритма</h3>
            <div className="mt-2 flex flex-wrap gap-2">
              {Object.entries(preview.fixed_outputs).map(([name, shape]) => (
                <span key={name} className="rounded-md bg-muted px-2 py-1 font-mono text-[10px]">
                  {name} → [{shape.join('×')}] · auto
                </span>
              ))}
            </div>
          </section>
        )}

        {preview?.components && preview.components.length > 0 && (
          <section className="rounded-lg border p-3">
            <h3 className="mb-3 text-sm font-semibold">Реальная собранная архитектура</h3>
            <AlgorithmDiagram
              show={{ loop: false, network: true }}
              hideTitles
              network={{
                components: preview.components,
                totalParams: preview.total_params ?? undefined,
                trainableParams: preview.trainable_params ?? undefined,
                inputShape: preview.input_shape,
                outputShape: preview.output_shape,
              }}
            />
          </section>
        )}
      </div>
    </div>
  )
}

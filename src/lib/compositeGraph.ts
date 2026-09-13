import type { Edge, Node } from '@xyflow/react'

import type {
  ArchitectureComponent,
  CompositeNetworkFamily,
  CompositeNetworkSpec,
  NetworkPreviewResult,
} from '@/api/types'
import { ACTIVATION_LABELS } from '@/lib/networkBuilder'

export type CompositeNodeKind = 'input' | 'encoder' | 'component' | 'fixed' | 'head' | 'ema'

export interface CompositeGraphNodeData {
  title: string
  subtitle?: string
  detail?: string
  params?: number
  kind: CompositeNodeKind
  componentKey?: string
  dimKeys?: string[]
  selected?: boolean
  onSelect?: () => void
  hasHandleIn?: boolean
  hasHandleOut?: boolean
}

interface GraphSpecNode {
  id: string
  title: string
  kind: CompositeNodeKind
  x: number
  y: number
  componentKey?: string
  previewName?: string
  dimKeys?: string[]
}

interface GraphSpecEdge {
  source: string
  target: string
  label?: string
  dashed?: boolean
}

const COL = 260
const ROW = 140
const CX = 480

function encoderSubtitle(spec: CompositeNetworkSpec): string {
  if (spec.encoder.kind === 'auto') return 'Auto CNN / MLP по среде'
  if (spec.encoder.kind === 'vector_mlp') return 'Vector MLP'
  if (spec.encoder.kind === 'image_cnn') return 'Image CNN'
  return 'Custom chain'
}

function mlpSubtitle(spec: CompositeNetworkSpec, key: string): string | undefined {
  const mlp = spec.components[key]
  if (!mlp) return undefined
  const sizes = mlp.hidden_sizes.length ? mlp.hidden_sizes.join(' → ') : 'линейный'
  const act = ACTIVATION_LABELS[mlp.activation as keyof typeof ACTIVATION_LABELS] ?? mlp.activation.toUpperCase()
  const extras = [
    mlp.dropout > 0 ? `drop ${mlp.dropout}` : null,
    mlp.batch_norm ? 'BN' : null,
  ].filter(Boolean)
  return extras.length ? `${sizes} · ${act} · ${extras.join(' · ')}` : `${sizes} · ${act}`
}

function transformerSubtitle(spec: CompositeNetworkSpec): string {
  const d = spec.dimensions
  const parts = [
    d.num_layers != null ? `L=${d.num_layers}` : null,
    d.num_heads != null ? `H=${d.num_heads}` : null,
    d.embed_dim != null ? `d=${d.embed_dim}` : d.latent_dim != null ? `z=${d.latent_dim}` : null,
    d.rotary_emb ? 'RoPE' : null,
  ].filter(Boolean)
  return parts.join(' · ')
}

function dimSubtitle(spec: CompositeNetworkSpec, keys: string[]): string | undefined {
  const parts = keys
    .map((key) => {
      const value = spec.dimensions[key]
      if (value == null) return null
      if (key === 'rotary_emb') return value ? 'RoPE' : null
      if (key === 'latent_dim') return `z=${value}`
      if (key === 'hidden_dim') return `h=${value}`
      if (key === 'proj_dim') return `proj=${value}`
      if (key === 'embed_dim') return `d=${value}`
      if (key === 'stoch_variables') return `${value} vars`
      if (key === 'stoch_classes') return `${value} cls`
      return `${key}=${value}`
    })
    .filter(Boolean)
  return parts.length ? parts.join(' · ') : undefined
}

function previewByName(preview: NetworkPreviewResult | undefined): Map<string, ArchitectureComponent> {
  const map = new Map<string, ArchitectureComponent>()
  for (const component of preview?.components ?? []) {
    map.set(component.name, component)
  }
  return map
}

function headOutputKey(nodeId: string, family: CompositeNetworkFamily): string | undefined {
  if (nodeId === 'policy') return 'policy'
  if (nodeId === 'value') return 'value'
  if (nodeId === 'reward') return family === 'efficientzero' ? 'value_prefix' : 'reward'
  if (nodeId === 'latent') return 'next_token'
  return undefined
}

function shapeLabel(shape: (number | string)[] | undefined): string | undefined {
  if (!shape?.length) return undefined
  return `→ [${shape.join('×')}]`
}

function nodeSubtitle(
  spec: CompositeNetworkSpec,
  node: GraphSpecNode,
  preview: NetworkPreviewResult | undefined,
): string | undefined {
  if (node.kind === 'input') {
    if (node.id === 'obs' && preview?.input_shape?.length) return `[${preview.input_shape.join('×')}]`
    return undefined
  }
  if (node.kind === 'head') {
    return shapeLabel(preview?.fixed_outputs?.[headOutputKey(node.id, spec.family) ?? ''])
  }
  if (node.kind === 'encoder') return encoderSubtitle(spec)
  if (node.id === 'transformer' || (node.kind === 'fixed' && node.dimKeys?.includes('num_layers'))) {
    return transformerSubtitle(spec)
  }
  if (node.componentKey) return mlpSubtitle(spec, node.componentKey)
  if (node.dimKeys?.length) return dimSubtitle(spec, node.dimKeys)
  return undefined
}

function graphForFamily(family: CompositeNetworkFamily): { nodes: GraphSpecNode[]; edges: GraphSpecEdge[] } {
  if (family === 'efficientzero') {
    return {
      nodes: [
        { id: 'obs', title: 'Наблюдение', kind: 'input', x: CX, y: 0 },
        { id: 'encoder', title: 'Энкодер', kind: 'encoder', x: CX, y: ROW },
        { id: 'representation', title: 'Representation', kind: 'component', x: CX, y: ROW * 2, componentKey: 'representation', previewName: 'representation', dimKeys: ['latent_dim'] },
        { id: 'dynamics', title: 'Dynamics + LSTM', kind: 'component', x: CX, y: ROW * 3, componentKey: 'dynamics', previewName: 'dynamics', dimKeys: ['hidden_dim'] },
        { id: 'prediction', title: 'Prediction trunk', kind: 'component', x: CX, y: ROW * 4, componentKey: 'prediction', previewName: 'prediction' },
        { id: 'policy', title: 'Policy', kind: 'head', x: CX - COL, y: ROW * 5.2 },
        { id: 'value', title: 'Value', kind: 'head', x: CX, y: ROW * 5.2 },
        { id: 'reward', title: 'Reward', kind: 'head', x: CX + COL, y: ROW * 5.2 },
        { id: 'projector', title: 'SimSiam projector', kind: 'component', x: CX + COL * 1.5, y: ROW * 2, componentKey: 'projector', previewName: 'projector', dimKeys: ['proj_dim'] },
        { id: 'predictor', title: 'SimSiam predictor', kind: 'component', x: CX + COL * 1.5, y: ROW * 3, componentKey: 'predictor', previewName: 'predictor' },
      ],
      edges: [
        { source: 'obs', target: 'encoder' },
        { source: 'encoder', target: 'representation' },
        { source: 'representation', target: 'dynamics' },
        { source: 'dynamics', target: 'prediction' },
        { source: 'prediction', target: 'policy' },
        { source: 'prediction', target: 'value' },
        { source: 'prediction', target: 'reward' },
        { source: 'representation', target: 'projector', label: 'consistency' },
        { source: 'projector', target: 'predictor' },
      ],
    }
  }

  const withSimSiam = family === 'researchimzero' || family === 'latentimzero'
  const latent = family === 'latentimzero'
  return {
    nodes: [
      { id: 'obs', title: 'Наблюдение', kind: 'input', x: CX - COL, y: 0 },
      { id: 'action', title: 'Действие', kind: 'input', x: CX + COL, y: 0 },
      { id: 'encoder', title: 'Энкодер', kind: 'encoder', x: CX - COL, y: ROW },
      { id: 'tokenizer', title: 'Tokenizer', kind: 'component', x: CX - COL, y: ROW * 2, componentKey: 'tokenizer', previewName: 'tokenizer' },
      { id: 'action_embed', title: 'Action embedding', kind: 'fixed', x: CX + COL, y: ROW * 2, previewName: 'action_embed' },
      { id: 'transformer', title: 'Causal Transformer', kind: 'fixed', x: CX, y: ROW * 3.2, previewName: 'transformer', dimKeys: ['embed_dim', 'num_layers', 'num_heads', 'ffn_multiplier', 'dropout', 'rotary_emb'] },
      { id: 'heads', title: 'Reward / Value / Policy', kind: 'component', x: CX, y: ROW * 4.4, componentKey: 'heads', previewName: 'heads' },
      { id: 'policy', title: 'Policy', kind: 'head', x: CX - COL * 1.5, y: ROW * 5.6 },
      { id: 'value', title: 'Value', kind: 'head', x: CX - COL * 0.5, y: ROW * 5.6 },
      { id: 'reward', title: 'Reward', kind: 'head', x: CX + COL * 0.5, y: ROW * 5.6 },
      { id: 'latent', title: 'Next latent', kind: 'head', x: CX + COL * 1.5, y: ROW * 5.6 },
      { id: 'ema_tokenizer', title: 'EMA tokenizer', kind: 'ema', x: CX + COL * 2.6, y: ROW * 2, previewName: 'target_tokenizer' },
      { id: 'ema_transformer', title: 'EMA Transformer', kind: 'ema', x: CX + COL * 2.6, y: ROW * 3.2, previewName: 'target_transformer' },
      { id: 'ema_heads', title: 'EMA heads', kind: 'ema', x: CX + COL * 2.6, y: ROW * 4.4, previewName: 'target_heads' },
      ...(withSimSiam
        ? [
            { id: 'projector', title: 'SimSiam projector', kind: 'component' as const, x: CX - COL * 2.6, y: ROW * 3.2, componentKey: 'projector', previewName: 'projector', dimKeys: ['proj_dim'] },
            { id: 'predictor', title: 'SimSiam predictor', kind: 'component' as const, x: CX - COL * 2.6, y: ROW * 4.4, componentKey: 'predictor', previewName: 'predictor' },
          ]
        : []),
      ...(latent
        ? [
            { id: 'uncertainty_probe', title: 'Uncertainty probe', kind: 'fixed' as const, x: CX - COL * 2.6, y: ROW * 5.6, previewName: 'uncertainty_probe' },
            { id: 'planner', title: 'Gumbel planner', kind: 'fixed' as const, x: CX, y: ROW * 6.8 },
          ]
        : []),
    ],
    edges: [
      { source: 'obs', target: 'encoder' },
      { source: 'encoder', target: 'tokenizer' },
      { source: 'action', target: 'action_embed' },
      { source: 'tokenizer', target: 'transformer' },
      { source: 'action_embed', target: 'transformer' },
      { source: 'transformer', target: 'heads' },
      { source: 'heads', target: 'policy' },
      { source: 'heads', target: 'value' },
      { source: 'heads', target: 'reward' },
      { source: 'heads', target: 'latent' },
      { source: 'tokenizer', target: 'ema_tokenizer', dashed: true, label: 'EMA' },
      { source: 'transformer', target: 'ema_transformer', dashed: true },
      { source: 'heads', target: 'ema_heads', dashed: true },
      ...(withSimSiam
        ? [
            { source: 'transformer', target: 'projector', label: 'consistency' },
            { source: 'projector', target: 'predictor' },
          ]
        : []),
      ...(latent
        ? [
            { source: 'heads', target: 'uncertainty_probe', dashed: true, label: 'ensemble' },
            { source: 'heads', target: 'planner', label: 'search' },
          ]
        : []),
    ],
  }
}

export function buildCompositeGraph(
  spec: CompositeNetworkSpec,
  preview: NetworkPreviewResult | undefined,
  selectedId: string | null,
  onSelect: (id: string) => void,
): { nodes: Node[]; edges: Edge[] } {
  const { nodes: specNodes, edges: specEdges } = graphForFamily(spec.family)
  const previews = previewByName(preview)
  const sources = new Set(specEdges.map((edge) => edge.source))
  const targets = new Set(specEdges.map((edge) => edge.target))
  const nodes: Node[] = specNodes.map((node) => {
    const component = node.previewName ? previews.get(node.previewName) : undefined
    return {
      id: node.id,
      type: 'composite',
      position: { x: node.x, y: node.y },
      data: {
        title: node.title,
        subtitle: nodeSubtitle(spec, node, preview),
        detail: component?.type?.replace(/^_/, ''),
        params: component?.params,
        kind: node.kind,
        componentKey: node.componentKey,
        dimKeys: node.dimKeys,
        selected: selectedId === node.id,
        onSelect: () => onSelect(node.id),
        hasHandleIn: targets.has(node.id),
        hasHandleOut: sources.has(node.id),
      } satisfies CompositeGraphNodeData,
    }
  })
  const edges: Edge[] = specEdges.map((edge, index) => ({
    id: `e-${edge.source}-${edge.target}-${index}`,
    source: edge.source,
    target: edge.target,
    label: edge.label,
    animated: !edge.dashed && (edge.source === 'transformer' || edge.source === 'prediction' || edge.source === 'heads'),
    style: edge.dashed ? { strokeDasharray: '6 4' } : undefined,
    labelStyle: { fontSize: 9, fill: 'var(--muted-foreground)' },
  }))

  return { nodes, edges }
}

export function selectableGraphNode(id: string, family: CompositeNetworkFamily): GraphSpecNode | undefined {
  return graphForFamily(family).nodes.find((node) => node.id === id)
}

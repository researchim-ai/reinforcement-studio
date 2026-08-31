import type { ActivationFn, NetworkFamily, NetworkHead, NetworkLayer, NetworkLayerType, NetworkSpec } from '@/api/types'

export const FAMILY_LABELS: Record<NetworkFamily, string> = {
  actor_critic: 'Actor-Critic (PPO / A2C)',
  q_network: 'Q-Network (DQN)',
  dueling_q: 'Dueling Q-Network (Rainbow DQN)',
  alphazero: 'AlphaZero (policy + value)',
}

export const FAMILY_HEADS: Record<NetworkFamily, string[]> = {
  actor_critic: ['action', 'value'],
  q_network: ['q'],
  dueling_q: ['advantage', 'value'],
  alphazero: ['policy', 'value'],
}

export const HEAD_LABELS: Record<string, string> = {
  action: 'Action (логиты / mu)',
  value: 'Value (оценка состояния)',
  q: 'Q-values',
  advantage: 'Advantage (по действиям)',
  policy: 'Policy (логиты ходов)',
}

export const LAYER_LABELS: Record<NetworkLayerType, string> = {
  linear: 'Linear',
  conv2d: 'Conv2D',
  maxpool2d: 'MaxPool2D',
  flatten: 'Flatten',
  activation: 'Активация',
  dropout: 'Dropout',
  batchnorm: 'BatchNorm',
}

export const ACTIVATION_LABELS: Record<ActivationFn, string> = {
  relu: 'ReLU',
  tanh: 'Tanh',
  sigmoid: 'Sigmoid',
  gelu: 'GELU',
  leaky_relu: 'LeakyReLU',
}

export const LAYER_TYPES: NetworkLayerType[] = ['linear', 'conv2d', 'maxpool2d', 'flatten', 'activation', 'dropout', 'batchnorm']

export function defaultLayer(type: NetworkLayerType): NetworkLayer {
  switch (type) {
    case 'linear':
      return { type: 'linear', out_features: 64 }
    case 'conv2d':
      return { type: 'conv2d', out_channels: 32, kernel_size: 3, stride: 1, padding: 0 }
    case 'maxpool2d':
      return { type: 'maxpool2d', kernel_size: 2, stride: 2 }
    case 'flatten':
      return { type: 'flatten' }
    case 'activation':
      return { type: 'activation', fn: 'relu' }
    case 'dropout':
      return { type: 'dropout', p: 0.1 }
    case 'batchnorm':
      return { type: 'batchnorm' }
  }
}

function head(name: string): NetworkHead {
  return { name, layers: [{ type: 'linear', out_features: null }] }
}

export function defaultSpecForFamily(family: NetworkFamily): NetworkSpec {
  if (family === 'q_network') {
    return {
      trunk: [
        { type: 'linear', out_features: 128 },
        { type: 'activation', fn: 'relu' },
        { type: 'linear', out_features: 128 },
        { type: 'activation', fn: 'relu' },
      ],
      heads: FAMILY_HEADS.q_network.map(head),
    }
  }
  if (family === 'dueling_q') {
    return {
      trunk: [
        { type: 'linear', out_features: 128 },
        { type: 'activation', fn: 'relu' },
      ],
      heads: FAMILY_HEADS.dueling_q.map(head),
    }
  }
  if (family === 'alphazero') {
    return {
      trunk: [
        { type: 'conv2d', out_channels: 32, kernel_size: 3, stride: 1, padding: 1 },
        { type: 'activation', fn: 'relu' },
        { type: 'conv2d', out_channels: 32, kernel_size: 3, stride: 1, padding: 1 },
        { type: 'activation', fn: 'relu' },
        { type: 'flatten' },
        { type: 'linear', out_features: 128 },
        { type: 'activation', fn: 'relu' },
      ],
      heads: FAMILY_HEADS.alphazero.map(head),
    }
  }
  return {
    trunk: [
      { type: 'linear', out_features: 128 },
      { type: 'activation', fn: 'tanh' },
      { type: 'linear', out_features: 128 },
      { type: 'activation', fn: 'tanh' },
    ],
    heads: FAMILY_HEADS.actor_critic.map(head),
  }
}

/** Re-shapes an existing spec's heads to match a newly picked family,
 * preserving the trunk and any head bodies whose name is still relevant. */
export function reconcileHeadsForFamily(spec: NetworkSpec, family: NetworkFamily): NetworkSpec {
  const byName = new Map(spec.heads.map((h) => [h.name, h]))
  return { trunk: spec.trunk, heads: FAMILY_HEADS[family].map((name) => byName.get(name) ?? head(name)) }
}

export function layerSummary(layer: NetworkLayer): string {
  switch (layer.type) {
    case 'linear':
      return layer.out_features ? `${layer.out_features}` : 'авто'
    case 'conv2d':
      return `${layer.out_channels}ch k${layer.kernel_size ?? 3}s${layer.stride ?? 1}`
    case 'maxpool2d':
      return `k${layer.kernel_size ?? 2}s${layer.stride ?? layer.kernel_size ?? 2}`
    case 'activation':
      return ACTIVATION_LABELS[layer.fn ?? 'relu']
    case 'dropout':
      return `p=${layer.p ?? 0.1}`
    case 'flatten':
    case 'batchnorm':
      return ''
  }
}

export function shapeLabel(shape: number[] | undefined): string {
  if (!shape || shape.length === 0) return '?'
  return `[${shape.join('×')}]`
}

// ------------------------------------------------- Quick (inline) layer editor
//
// Below the full Network Builder page (trunk + heads + arbitrary layer
// types), the Designer's Algorithm node offers a much smaller "quick
// editor" — just the trunk's hidden `Linear`+activation sizes, for
// algorithms that already accept a hand-designed `NetworkSpec`
// (`ppo`/`a2c`/`dqn`/`rainbow_dqn` — see `rl_core/algorithms/native/*.py`).
// It never touches heads (still exactly `FAMILY_HEADS[family]`, auto-sized
// to the env at build time, same as `defaultSpecForFamily`) — just how
// many hidden layers the shared trunk has and how wide each one is.

/** Only built-in algorithms know how to accept a hand-designed spec net
 * (see rl_core/algorithms/native/{ppo,a2c,dqn,rainbow_dqn}.py and
 * rl_core/alphazero/base.py::build_network) — custom plugins ignore
 * `network_spec` unless they opt in themselves. AlphaZero already has its
 * own quick architecture knobs (`channels`/`num_blocks` hyperparams), so
 * it's deliberately excluded here — the quick *layer* editor only applies
 * to the plain-MLP-trunk families. */
export function requiredFamilyFor(algorithmId: string, kind: 'gym' | 'alphazero'): NetworkFamily | null {
  if (kind === 'alphazero') return null
  if (algorithmId === 'dqn') return 'q_network'
  if (algorithmId === 'rainbow_dqn') return 'dueling_q'
  if (algorithmId === 'ppo' || algorithmId === 'a2c') return 'actor_critic'
  return null
}

/** Same family used for the full Network Builder's "своя" architecture
 * picker (includes `alphazero`, unlike `requiredFamilyFor` above). */
export function networkFamilyFor(algorithmId: string, kind: 'gym' | 'alphazero'): NetworkFamily | null {
  if (kind === 'alphazero') return algorithmId === 'alphazero' ? 'alphazero' : null
  return requiredFamilyFor(algorithmId, kind)
}

export const FAMILY_DEFAULT_ACTIVATION: Record<NetworkFamily, ActivationFn> = {
  actor_critic: 'tanh',
  q_network: 'relu',
  dueling_q: 'relu',
  alphazero: 'relu',
}

/** The trunk hidden-layer sizes shown before the user has touched
 * anything — matches `defaultSpecForFamily`'s own trunk exactly, so
 * switching the quick editor's first field is a no-op until actually
 * edited. */
export function defaultHiddenSizesForFamily(family: NetworkFamily): number[] {
  return defaultSpecForFamily(family)
    .trunk.filter((layer) => layer.type === 'linear')
    .map((layer) => layer.out_features ?? 128)
}

/** Builds a full `NetworkSpec` from just a list of trunk hidden-layer
 * sizes — heads are always the family's defaults (a single auto-sized
 * Linear straight off the trunk, exactly like `defaultSpecForFamily`),
 * since the quick editor never lets you touch head shape. */
export function quickSpecForFamily(family: NetworkFamily, hiddenSizes: number[]): NetworkSpec {
  const activation = FAMILY_DEFAULT_ACTIVATION[family]
  const trunk: NetworkLayer[] = []
  for (const size of hiddenSizes) {
    trunk.push({ type: 'linear', out_features: size })
    trunk.push({ type: 'activation', fn: activation })
  }
  return { trunk, heads: FAMILY_HEADS[family].map(head) }
}

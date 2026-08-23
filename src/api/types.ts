export type ActionKind = 'discrete' | 'continuous'
export type EnvKind = 'gym' | 'alphazero'

export interface EnvSpec {
  id: string
  name: string
  category: string
  description: string
  action_kind: ActionKind
  compatible_algorithms: string[]
  extra_requirement?: string | null
  kind: EnvKind
  available: boolean
  preview_url?: string | null
  preview_thumb_url?: string | null
}

export interface WrapperSpec {
  type: string
  label: string
  params: Record<string, number | string>
  is_custom?: boolean
}

export interface HyperparamSpec {
  key: string
  label: string
  type: 'float' | 'int'
  default: number
  min: number
  max: number
}

export interface AlgorithmSpec {
  id: string
  name: string
  kind: EnvKind
  description: string
  hyperparams: HyperparamSpec[]
  is_custom?: boolean
  supported_action_kinds?: ActionKind[]
}

export type PluginKind = 'gym-algorithms' | 'alphazero-algorithms' | 'reward-functions'

export interface PluginScriptMeta {
  id: string
  slug: string
  name: string
  kind: string
  description: string
  hyperparams: HyperparamSpec[]
  is_custom: true
  broken?: boolean
  error?: string
}

export interface PluginTemplate {
  id: string
  label: string
  kind: 'gym-algorithm' | 'alphazero-algorithm' | 'reward-function'
  code: string
}

export interface ValidateResult {
  ok: boolean
  error: string | null
  traceback: string | null
}

export interface WrapperNode {
  type: string
  params: Record<string, number | string>
}

export interface ExperimentConfig {
  kind: EnvKind
  name?: string
  environment: {
    id: string
    wrappers: WrapperNode[]
  }
  algorithm: {
    id: string
    hyperparams: Record<string, number>
  }
  training: {
    total_timesteps?: number
    num_iterations?: number
    seed?: number
    use_gpu?: boolean
  }
}

export interface SpaceInfo {
  type: string
  shape: number[] | null
  n?: number
}

export interface NetworkInfo {
  channels: number
  num_blocks: number
  rows: number
  cols: number
  action_size: number
  input_planes: number
}

export interface InspectEnvironment {
  raw_observation_space: SpaceInfo | null
  raw_action_space: SpaceInfo | null
  observation_space: SpaceInfo | null
  action_space: SpaceInfo | null
}

export interface InspectNetwork {
  policy: string
  layers: string[]
  total_params: number
  trainable_params: number
  input_shape: number[]
  output_shape: number[]
  note?: string
  channels?: number
  num_blocks?: number
  rows?: number
  cols?: number
  action_size?: number
  input_planes?: number
}

export interface InspectResult {
  environment: InspectEnvironment | null
  network: InspectNetwork | null
  error: string | null
}

export interface MetricsSnapshot {
  run_id: string
  kind: EnvKind
  status: 'running' | 'completed' | 'stopped' | 'failed'
  algo: string
  env_id: string
  step: number
  total_timesteps: number
  episode_reward_mean?: number | null
  episode_length_mean?: number | null
  fps?: number
  elapsed_seconds?: number
  frame_base64?: string
  loss?: number
  win_rate_vs_prev?: number
  accepted?: boolean
  buffer_size?: number
  arena?: { wins_a: number; wins_b: number; draws: number; games: number }
  error?: string
  // Static per-run facts, merged into every snapshot (see MetricsCallback /
  // alphazero train.py): effective hyperparams + model/network shape.
  hyperparams?: Record<string, number | string | boolean>
  policy?: string
  device?: string
  total_params?: number
  seed?: number | null
  observation_space?: SpaceInfo
  action_space?: SpaceInfo
  wrappers?: { type: string; params: Record<string, number | string> }[]
  network?: NetworkInfo
  board?: number[][] | null
  board_winner?: number | null
}

export interface RunSummary {
  run_id: string
  name: string
  kind: EnvKind
  environment_id: string
  algorithm_id: string
  status: string
  running: boolean
  metrics: MetricsSnapshot | Record<string, never>
  has_model: boolean
  created_at?: string
}

export interface ModelInfo {
  source: 'run' | 'checkpoint'
  id: string
  label: string
  kind: EnvKind
  environment_id: string
  algorithm_id: string
  status?: string
  size_bytes: number
  modified_at: string
  metrics: Record<string, unknown>
}

export interface GameInfo {
  id: string
  name: string
  rows: number
  cols: number
}

export interface ArenaOpponent {
  id: string
  source: 'builtin' | 'run' | 'checkpoint'
  label: string
}

export interface ArenaState {
  session_id: string
  board: number[][]
  rows: number
  cols: number
  current_player: number
  human_player: number
  done: boolean
  winner: number | null
  legal_actions: number[]
  ai_info: {
    action: number | null
    visit_probs?: Record<string, number>
  } | null
}

export interface SystemInfo {
  platform: string
  cpu_count: number
  torch_cuda_available: boolean
  gpus: { id: number; name: string; memory_used_mb: number; memory_total_mb: number; utilization: number | null }[]
}

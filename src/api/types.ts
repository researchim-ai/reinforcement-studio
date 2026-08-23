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

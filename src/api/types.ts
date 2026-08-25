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
  // Per-env overrides merged on top of the algorithm's global hyperparam
  // defaults (see ALGORITHM_CATALOG on the backend) — e.g. Gomoku needs a
  // much bigger MCTS/network budget than Tic-Tac-Toe to actually learn.
  default_hyperparams?: Record<string, number> | null
}

export interface WrapperSpec {
  type: string
  label: string
  params: Record<string, number | string>
  is_custom?: boolean
}

export interface HyperparamOption {
  value: number
  label: string
}

export interface HyperparamCondition {
  key: string
  eq?: number
  gte?: number
  lte?: number
}

export interface HyperparamSpec {
  key: string
  label: string
  type: 'float' | 'int'
  default: number
  min: number
  max: number
  // When set, the value is an enum encoded as a plain number (e.g. memory
  // type: 0=none/1=LSTM/2=GRU) rendered as a labeled dropdown instead of a
  // number input — keeps every hyperparam a `number` end-to-end (save/load,
  // ExperimentConfig, plugin editor, ...) while still giving it a proper
  // picker UI where a raw numeric code wouldn't mean anything to a user.
  options?: HyperparamOption[]
  visibleWhen?: HyperparamCondition[]
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
    // Set when the user picked a hand-designed architecture on the Network
    // Builder page (/network-builder) instead of the algorithm's default
    // net — resolved server-side (rl_core/netbuilder_store.py) at run time.
    network_spec_id?: string | null
  }
  training: {
    total_timesteps?: number
    num_iterations?: number
    seed?: number
    use_gpu?: boolean
  }
}

// ------------------------------------------------- Network Architecture Builder

export type NetworkFamily = 'actor_critic' | 'q_network' | 'dueling_q' | 'alphazero'

export type NetworkLayerType = 'linear' | 'conv2d' | 'maxpool2d' | 'flatten' | 'activation' | 'dropout' | 'batchnorm'
export type ActivationFn = 'relu' | 'tanh' | 'sigmoid' | 'gelu' | 'leaky_relu'

export interface NetworkLayer {
  type: NetworkLayerType
  out_features?: number | null // linear
  out_channels?: number | null // conv2d
  kernel_size?: number | null // conv2d / maxpool2d
  stride?: number | null // conv2d / maxpool2d
  padding?: number | null // conv2d
  fn?: ActivationFn // activation
  p?: number // dropout
}

export interface NetworkHead {
  name: string
  layers: NetworkLayer[]
}

export interface NetworkSpec {
  trunk: NetworkLayer[]
  heads: NetworkHead[]
}

export interface NetworkDoc {
  name: string
  description: string
  family: NetworkFamily
  spec: NetworkSpec
}

export interface NetworkMeta {
  id: string
  slug: string
  name: string
  description: string
  family: NetworkFamily
  broken?: boolean
  error?: string
}

export interface NetworkFamilyInfo {
  id: NetworkFamily
  required_heads: string[]
}

export interface NetworkPreviewRequest {
  family: NetworkFamily
  spec: NetworkSpec
  environment_id?: string | null
  wrappers?: WrapperNode[]
  game_id?: string | null
}

export interface NetworkHeadPreview {
  layer_shapes: number[][]
  out_shape: number[]
  error: string | null
  error_index: number | null
}

export interface NetworkPreviewResult {
  ok: boolean
  error: string | null
  input_shape?: number[]
  trunk: { shape: number[] }[]
  trunk_error: string | null
  trunk_error_index: number | null
  heads: Record<string, NetworkHeadPreview>
  total_params: number | null
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
  episode_extrinsic_reward_mean?: number | null
  episode_intrinsic_reward_mean?: number | null
  episode_length_mean?: number | null
  exploration_epsilon?: number
  rnd_bonus_mean?: number
  rnd_predictor_loss?: number
  fps?: number
  elapsed_seconds?: number
  // A full episode played end-to-end and packed into a single-play GIF
  // (rendered every `render_every_steps` — see rl_core/algorithms/
  // metrics_callback.py::render_episode) rather than a single freeze-frame,
  // so the live preview shows one coherent playthrough instead of jumping
  // to an arbitrary unrelated moment each time it refreshes.
  episode_gif_base64?: string
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
  layers?: string[]
  device?: string
  total_params?: number
  seed?: number | null
  observation_space?: SpaceInfo
  action_space?: SpaceInfo
  wrappers?: { type: string; params: Record<string, number | string> }[]
  network?: NetworkInfo
  // Move-by-move board history of one self-play game from this iteration
  // (post-move snapshots, first entry is the empty starting board) — lets
  // the Training Monitor replay the whole game instead of showing only the
  // final position.
  board_history?: number[][][] | null
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
  torch_version: string | null
  // torch.version.cuda — null means a CPU-only wheel; a value like "12.1"
  // means the CUDA wheel is installed but torch_cuda_available may still be
  // false (driver too old/missing), which is a different problem than "the
  // wrong wheel got installed" and needs a different fix from the user.
  torch_cuda_build: string | null
  gpus: { id: number; name: string; memory_used_mb: number; memory_total_mb: number; utilization: number | null }[]
}

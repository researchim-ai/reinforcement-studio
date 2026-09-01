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
  // Suggested `total_timesteps` for this env, pre-filled by the Designer
  // instead of the flat 50k default — set for pixel-input/hardcore envs
  // (CarRacing, Atari, BipedalWalkerHardcore, Humanoid) where 50k steps
  // only ever shows the "hasn't learned anything yet" stage.
  recommended_total_timesteps?: number | null
  // Suggested `training.num_envs` for this env — set for envs heavy enough
  // that batching several parallel copies meaningfully speeds up training
  // (Atari, Box2D); `null`/absent means "no particular recommendation, 1
  // is fine".
  recommended_num_envs?: number | null
  // Wrapper-graph nodes to pre-fill when this env is selected — see
  // `handleEnvChange` in ExperimentDesigner.tsx. `null`/absent leaves the
  // wrapper graph empty like before.
  recommended_wrappers?: WrapperNode[] | null
  scene_agent_count?: number | null
  // Number of distinct `team` values across the scene's agent groups — 1
  // for every non-MARL scene. Drives whether `ippo` shows up in
  // `compatible_algorithms` (see `team_count()` in rl_core/scene_store.py).
  scene_team_count?: number | null
  scene_slug?: string | null
}

export interface SceneMeta {
  id: string
  slug: string
  name: string
  description: string
  agent_count: number
  // See `EnvSpec.scene_team_count` above — same value, just under the name
  // `rl_core/scene_store.py::meta()` actually returns.
  team_count: number
  action_kind: ActionKind
  broken?: boolean
  error?: string
}

/** Visual mesh shape — collisions still use radius/AABB in the Python sim. */
export type SceneShape =
  | 'box'
  | 'sphere'
  | 'cylinder'
  | 'cone'
  | 'capsule'
  | 'pyramid'
  | 'crystal'

/** Procedural material pattern (canvas-generated, no external image files). */
export type SceneMaterialPattern = 'solid' | 'checker' | 'stripes' | 'noise' | 'brick' | 'dots'

export interface SceneMaterial {
  pattern?: SceneMaterialPattern
  color?: string
  color2?: string
  roughness?: number
  metalness?: number
  emissive?: boolean
}

export interface SceneSpec {
  name: string
  description?: string
  world: { width: number; depth: number; wall_height?: number }
  objects: SceneObject[]
  items: SceneItem[]
  agents: SceneAgentGroup[]
  episode?: { max_steps?: number }
  // MARL reward rules (predator/prey tagging, cooperative team-shared
  // reward) — see `_tag_rule`/`_team_shared_reward` in
  // rl_core/envs/scene_env.py. Empty/absent for every single-team scene.
  rules?: SceneRules
}

export interface SceneTagRule {
  enabled?: boolean
  predator_role?: string
  prey_role?: string
  predator_reward?: number
  prey_reward?: number
  prey_terminates?: boolean
  prey_respawns?: boolean
}

export interface SceneRules {
  tag?: SceneTagRule
  // Pools every team's per-step rewards into one shared total, credited
  // to every agent on that team — cooperative credit assignment for tasks
  // where the "right" behavior is a team effort, not something any single
  // agent could learn to do from its own reward alone.
  team_shared_reward?: boolean
}

export interface SceneObject {
  id: string
  /** wall = solid collider; prop = visual-only decoration (no collision in MVP). */
  type: 'wall' | 'ground' | 'prop'
  position: [number, number, number]
  size: [number, number, number]
  shape?: SceneShape
  material?: SceneMaterial
}

export interface SceneItem {
  id: string
  type: 'reward' | 'hazard'
  position: [number, number, number]
  radius: number
  reward: number
  respawn?: boolean
  cooldown_steps?: number
  terminate?: boolean
  shape?: SceneShape
  material?: SceneMaterial
  // Only pays out / despawns for agents whose group `team` matches this —
  // other teams still see/collide with it but never collect it. Unset =
  // open to every team (the pre-MARL default).
  restrict_team?: string
}

export interface SceneAgentGroup {
  id: string
  count: number
  team?: string
  // Free-form tag consumed by `rules.tag` (predator/prey) — has no effect
  // on its own beyond distinguishing agents in the sensor readout (see
  // `_entities_for_sensors` in rl_core/envs/scene_env.py); defaults to
  // `"agent"` when unset.
  role?: string
  spawn: { center: [number, number, number]; radius: number }
  body_radius: number
  movement: { type: 'discrete4' | 'discrete8' | 'continuous'; speed: number }
  sensors: { type: 'nearest_k'; k: number; range: number }
  shape?: SceneShape
  material?: SceneMaterial
}

export type SceneSelection =
  | { kind: 'object'; id: string }
  | { kind: 'item'; id: string }
  | { kind: 'agents' }
  | null

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
  // Set only on the four World Model algorithms (dreamer/mbpo/pets/
  // world_models_ha, see backend/routes/environments.py's ALGORITHM_CATALOG)
  // — which World Model type (WorldModelType below) this algorithm expects,
  // so the Designer's Algorithm node can offer only matching saved World
  // Models in its `world_model_id` dropdown, mirroring `NetworkFamily`'s
  // role for `network_spec_id`.
  world_model_type?: WorldModelType | null
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
    // Set instead of `network_spec_id` by the Algorithm node's inline
    // "quick layer editor" (src/components/designer/QuickNetworkEditor.tsx)
    // — a spec built on the fly from just a list of trunk hidden-layer
    // sizes, never saved to disk under a name. Mutually exclusive with
    // `network_spec_id` in practice (the Designer only ever sets one).
    network_spec?: NetworkSpec | null
    // Same "saved catalog entry vs. inline" pair as `network_spec_id`/
    // `network_spec` above, but for a World Model (rl_core/world_models/) —
    // only meaningful for the four algorithms with `world_model_type` set
    // (see AlgorithmSpec above). `world_model_id` picks up a saved World
    // Model's trained checkpoint too, if any; an inline `world_model_spec`
    // (the Designer's quick type+config picker, never saved to disk) never
    // has one. Resolved server-side by `rl_core/world_models/store.py::
    // resolve_world_model_spec`.
    world_model_id?: string | null
    world_model_spec?: { type: WorldModelType; config: WorldModelConfig } | null
  }
  training: {
    total_timesteps?: number
    num_iterations?: number
    seed?: number
    use_gpu?: boolean
    // Number of parallel env copies collected from every step
    // (`gymnasium.vector.AsyncVectorEnv` — one subprocess worker per lane) —
    // 1 (default) is a plain single env, exactly like before this field existed.
    num_envs?: number
    // Fine-tune/continue-training from a previous run's or Model Zoo
    // checkpoint's weights instead of a fresh network (see
    // rl_core/algorithms/resume.py) — the architecture can't change once
    // weights are loaded, so the Designer locks hyperparams/network while
    // this is set (see ExperimentDesigner.tsx's `resumeFrom` state).
    resume_from?: ResumeFrom | null
  }
}

export interface ResumeFrom {
  source: 'run' | 'checkpoint'
  id: string
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

// `network.json`, written next to `config.json` at the start of every run
// (see `rl_core/netbuilder_store.py::write_network_snapshot`) — the exact
// architecture that run actually trained with, independent of whatever
// happens afterwards to a saved `network_spec_id` catalog entry. Promoted
// Model Zoo checkpoints carry a copy of the same file alongside the
// weights.
export interface NetworkSnapshot {
  family: NetworkFamily | null
  spec: NetworkSpec | null
  source: 'inline' | 'catalog' | 'default' | 'unknown'
  network_spec_id?: string | null
  algorithm_id?: string | null
  environment_id?: string | null
  resolved_at?: string
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

// ------------------------------------------------- World Models

// The three implemented families (rl_core/world_models/spec.py::
// WORLD_MODEL_TYPES) — Dreamer-style RSSM, MBPO/PETS-style probabilistic
// ensemble dynamics, and the classic Ha & Schmidhuber VAE+MDN-RNN. Every
// one of the four World Model algorithms below expects exactly one of
// these (see `AlgorithmSpec.world_model_type`).
export type WorldModelType = 'rssm' | 'ensemble' | 'vae_mdnrnn'

// A type's config is a flat bag of architecture-size numbers (deterministic
// state size, ensemble members, latent size, ...) — never a layer graph
// like `NetworkSpec`, since none of the three families expose an
// arbitrary trunk/head structure to hand-design.
export type WorldModelConfig = Record<string, number>

export interface WorldModelConfigField {
  key: string
  label: string
  min: number
  max: number
}

// Static per-type metadata (`GET /world-models/types`) the World Model
// Builder's config form renders from directly, same rationale as
// `NetworkFamilyInfo`/`HyperparamSpec`.
export interface WorldModelTypeInfo {
  id: WorldModelType
  default_config: WorldModelConfig
  fields: WorldModelConfigField[]
}

export interface WorldModelDoc {
  name: string
  description: string
  type: WorldModelType
  config: WorldModelConfig
  environment_id?: string | null
}

export interface WorldModelMeta {
  id: string
  slug: string
  name: string
  description: string
  type: WorldModelType
  config: WorldModelConfig
  environment_id?: string | null
  // Whether `attach_checkpoint` has ever copied trained weights in — an
  // untrained spec still resolves fine (any algorithm/standalone run just
  // builds a fresh model of this type/config instead), it just starts from
  // scratch instead of picking up previous training.
  trained: boolean
  trained_at?: string | null
  source_run_id?: string | null
  broken?: boolean
  error?: string
}

// `world_model.json`, written next to `config.json` at the start of any run
// that resolved a World Model (standalone `kind: "world_model"` runs, or
// any of the four algorithms with `world_model_id`/`world_model_spec` set)
// — mirrors `NetworkSnapshot`'s role for Network Builder specs.
export interface WorldModelSnapshot {
  type: WorldModelType | null
  config: WorldModelConfig | null
  world_model_id?: string | null
  resolved_at?: string
}

// The exact `StartRunRequest` shape (backend/routes/training.py) a
// standalone World Model "Train" run sends — same endpoint
// (`POST /training/start`) and run-tracking pipeline as any other run
// (see `rl_core/world_models/trainer.py`'s module docstring), just with
// `kind: "world_model"` and `algorithm.id` being a `WorldModelType` rather
// than a real algorithm id.
export interface WorldModelStartRequest {
  kind: 'world_model'
  name?: string
  environment: { id: string; wrappers: WrapperNode[] }
  algorithm: {
    id: WorldModelType
    hyperparams: WorldModelConfig
    world_model_id?: string | null
  }
  training: {
    total_timesteps?: number
    seed?: number
    use_gpu?: boolean
    batch_size?: number
    seq_len?: number
    collect_steps_per_iter?: number
    train_steps_per_iter?: number
    learning_rate?: number
    // Parallel env lanes for data collection (see
    // `rl_core/world_models/trainer.py`'s module docstring) — same
    // `training.num_envs` lever every other algorithm's run already has.
    num_envs?: number
  }
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
  // Widened beyond `EnvKind` — a standalone World Model run (see
  // `rl_core/world_models/trainer.py`) writes `kind: "world_model"`,
  // never "gym"/"alphazero".
  kind: EnvKind | 'world_model'
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
  // Per-algorithm training losses — which of these are present depends on
  // `algo`/`kind` (see AlgorithmDiagram-adjacent chart logic in
  // TrainingMonitor.tsx): PPO/A2C report policy_loss/value_loss/entropy,
  // DQN/Rainbow DQN report td_loss, SAC/DDPG/TD3 report actor_loss/critic_loss,
  // AlphaZero reports policy_loss/value_loss (as well as the combined
  // `loss` below, kept for backwards compatibility with older runs). ES
  // never computes a loss at all — it reports population fitness instead
  // (see rl_core/algorithms/native/es.py), reused by the "Loss" chart as
  // its closest analogue.
  policy_loss?: number | null
  value_loss?: number | null
  entropy?: number | null
  td_loss?: number | null
  actor_loss?: number | null
  critic_loss?: number | null
  es_mean_fitness?: number | null
  es_best_fitness?: number | null
  es_sigma?: number | null
  // World Model losses (rl_core/world_models/losses.py) — which subset is
  // present depends on which of the three types this run trains:
  // RSSM (standalone `kind:"world_model"` id="rssm", or `dreamer`) reports
  // recon_loss/kl_loss/reward_loss/continue_loss; Ensemble (id="ensemble",
  // or `mbpo`/`pets`) reports dynamics_loss/reward_loss; VAE+MDN-RNN
  // (id="vae_mdnrnn", or `world_models_ha`) reports vae_recon_loss/
  // vae_kl_loss/mdn_loss/reward_loss/continue_loss. `dreamer`/`mbpo` also
  // report actor_loss/critic_loss (their imagination/model-augmented
  // actor-critic) alongside the world model's own losses above.
  recon_loss?: number | null
  kl_loss?: number | null
  reward_loss?: number | null
  continue_loss?: number | null
  dynamics_loss?: number | null
  vae_recon_loss?: number | null
  vae_kl_loss?: number | null
  mdn_loss?: number | null
  // Dreamer's own imagined-rollout return estimate — not a loss, but
  // charted alongside its actor/critic losses as a quick "is imagination
  // actually predicting anything useful" signal.
  imagined_return_mean?: number | null
  // MBPO's imagined-transition buffer size, and every standalone World
  // Model / `world_models_ha`'s mean reward over its most recent
  // random-policy collection batch — informational, not chart-worthy on
  // their own, but harmless to carry through.
  model_buffer_size?: number | null
  collect_mean_reward?: number | null
  // Best-effort metrics pulled from SB3's own logger for custom plugins
  // that subclass a stable-baselines3 algorithm (see metrics_callback.py).
  entropy_loss?: number | null
  approx_kl?: number | null
  clip_fraction?: number | null
  fps?: number
  elapsed_seconds?: number
  // A full episode played end-to-end and packed into a single-play GIF
  // (rendered every `render_every_steps` — see rl_core/algorithms/
  // metrics_callback.py::render_episode) rather than a single freeze-frame,
  // so the live preview shows one coherent playthrough instead of jumping
  // to an arbitrary unrelated moment each time it refreshes.
  // Only present on the exact snapshot written right after a fresh
  // render (see `runner_utils.py`/`metrics_callback.py`) — every snapshot
  // in between re-embedding the same (up to a few MB) blob for nothing
  // would bloat metrics.json/the metrics WebSocket for no benefit, so the
  // Training Monitor falls back to fetching `episode_gif_file` by URL
  // (GET /training/runs/{run_id}/preview.gif) whenever this is absent.
  episode_gif_base64?: string
  // Relative path inside the run folder — latest preview GIF on disk (see
  // `persist_episode_gif` in metrics_callback.py).
  episode_gif_file?: string
  // The `step` at which `episode_gif_file` was last (re)written — stable
  // across every snapshot until the next render, so it's a cache-busting
  // key that only changes when the file on disk actually does.
  episode_gif_step?: number
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
  // Set when this run was started via "Дообучить" (fine-tune/continue) from
  // a previous run or Model Zoo checkpoint — see `training.resume_from`.
  resumed_from?: ResumeFrom
}

export interface RunSummary {
  run_id: string
  name: string
  kind: EnvKind | 'world_model'
  environment_id: string
  algorithm_id: string
  status: string
  running: boolean
  metrics: MetricsSnapshot | Record<string, never>
  has_model: boolean
  created_at?: string
  // The run's folder as seen by the *backend* process — correct for
  // native/dev, but a container-internal path under Docker. The Training
  // Monitor prefers resolving/opening the real host path itself via
  // `window.electronAPI.runs` when available, and only falls back to
  // showing this verbatim (with a copy button) in the browser/web build.
  run_dir?: string
  // Present when this run is one member of a hyperparameter sweep (see
  // `backend/sweep_manager.py`) — `null`/absent for a plain standalone run.
  sweep?: SweepMembership | null
}

export interface SweepMembership {
  sweep_id: string
  name: string | null
  index: number
  total: number
  params: Record<string, number>
  seed: number
}

export interface SweepSummary {
  sweep_id: string
  name: string | null
  total: number
  run_ids: string[]
  created_at?: string
  environment_id?: string
  algorithm_id?: string
}

export interface SweepDetail {
  sweep_id: string
  runs: RunSummary[]
}

export interface EvaluateResult {
  episodes: number
  rewards: number[]
  lengths: number[]
  reward_mean: number | null
  reward_std: number | null
  reward_min: number | null
  reward_max: number | null
  length_mean: number | null
  episode_gif_base64?: string
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

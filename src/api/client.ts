import type {
  AlgorithmSpec,
  ArenaState,
  ArenaOpponent,
  EnvKind,
  EnvSpec,
  EvaluateResult,
  ExperimentConfig,
  GameInfo,
  InspectResult,
  ModelInfo,
  NetworkDoc,
  NetworkFamilyInfo,
  NetworkMeta,
  NetworkPreviewRequest,
  NetworkPreviewResult,
  NetworkSnapshot,
  NetworkSpec,
  PluginKind,
  PluginScriptMeta,
  PluginTemplate,
  RunSummary,
  SceneMeta,
  SceneSpec,
  SweepDetail,
  SweepSummary,
  SystemInfo,
  ValidateResult,
  WrapperNode,
  WrapperSpec,
} from './types'

let cachedPort: number | null = null

async function resolvePort(): Promise<number> {
  if (cachedPort != null) return cachedPort
  if (window.electronAPI) {
    cachedPort = await window.electronAPI.backend.port()
    return cachedPort
  }
  cachedPort = 8000
  return cachedPort
}

async function getBaseUrl(): Promise<string> {
  if (window.electronAPI) {
    const port = await resolvePort()
    return `http://127.0.0.1:${port}/api`
  }
  return '/api'
}

const basePromise = getBaseUrl()

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const BASE_URL = await basePromise
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })

  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error')
    throw new ApiError(res.status, text)
  }

  return res.json()
}

export const api = {
  health: () => request<{ status: string }>('/system/health'),
  systemInfo: () => request<SystemInfo>('/system/info'),

  listEnvironments: () => request<{ environments: EnvSpec[] }>('/environments/list'),
  listWrappers: () => request<{ wrappers: WrapperSpec[] }>('/environments/wrappers'),
  listAlgorithms: () => request<{ algorithms: AlgorithmSpec[] }>('/environments/algorithms'),
  inspectDesign: (payload: {
    kind: EnvKind
    environment: { id: string; wrappers?: { type: string; params: Record<string, number | string> }[] }
    algorithm: {
      id: string
      hyperparams: Record<string, number>
      network_spec_id?: string | null
      network_spec?: NetworkSpec | null
    }
  }) => request<InspectResult>('/environments/inspect', { method: 'POST', body: JSON.stringify(payload) }),
  resolveUrl: async (path: string) => {
    const base = await basePromise
    return `${base}${path.startsWith('/') ? path : `/${path}`}`
  },

  startRun: (config: ExperimentConfig) =>
    request<{ run_id: string }>('/training/start', { method: 'POST', body: JSON.stringify(config) }),
  stopRun: (runId: string) => request<{ success: boolean }>(`/training/stop/${runId}`, { method: 'POST' }),
  listRuns: () => request<{ runs: RunSummary[] }>('/training/runs'),
  getRun: (runId: string) => request<RunSummary>(`/training/runs/${runId}`),
  deleteRun: (runId: string) => request<{ success: boolean }>(`/training/runs/${runId}`, { method: 'DELETE' }),
  runLogs: (runId: string, lines = 300) =>
    request<{ run_id: string; stdout: string; stderr: string; error: string }>(
      `/training/runs/${runId}/logs?lines=${lines}`,
    ),
  getMetricsHistory: (runId: string) =>
    request<{ history: import('./types').MetricsSnapshot[] }>(`/training/runs/${runId}/metrics_history`),
  getRunNetwork: (runId: string) => request<NetworkSnapshot>(`/training/runs/${runId}/network`),
  getRunConfig: (runId: string) => request<ExperimentConfig>(`/training/runs/${runId}/config`),
  listSelfPlayIterations: (runId: string) =>
    request<{ iterations: string[] }>(`/training/runs/${runId}/games`),
  getSelfPlayGames: (runId: string, iteration: string) =>
    request<{ iteration: string; games: Record<string, unknown>[] }>(
      `/training/runs/${runId}/games/${iteration}`,
    ),

  listModels: () => request<{ models: ModelInfo[] }>('/models/list'),
  promoteRun: (runId: string, name: string) =>
    request<{ success: boolean; id: string }>(`/models/promote/${runId}`, {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),
  deleteCheckpoint: (name: string) =>
    request<{ success: boolean }>(`/models/checkpoints/${name}`, { method: 'DELETE' }),
  getCheckpointNetwork: (name: string) => request<NetworkSnapshot>(`/models/checkpoints/${name}/network`),
  getCheckpointConfig: (name: string) => request<ExperimentConfig>(`/models/checkpoints/${name}/config`),
  evaluateModel: (payload: { source: 'run' | 'checkpoint'; id: string; episodes?: number; record_gif?: boolean; seed?: number | null }) =>
    request<EvaluateResult>('/models/evaluate', { method: 'POST', body: JSON.stringify(payload) }),

  startSweep: (payload: {
    kind: EnvKind
    name?: string | null
    environment: { id: string; wrappers?: WrapperNode[] }
    algorithm: { id: string; hyperparams: Record<string, number> }
    training?: Record<string, unknown>
    grid: Record<string, number[]>
    seeds: number[]
  }) => request<{ sweep_id: string; run_ids: string[]; total: number }>('/sweeps/start', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  listSweeps: () => request<{ sweeps: SweepSummary[] }>('/sweeps'),
  getSweep: (sweepId: string) => request<SweepDetail>(`/sweeps/${sweepId}`),
  deleteSweep: (sweepId: string) =>
    request<{ success: boolean; deleted_runs: number }>(`/sweeps/${sweepId}`, { method: 'DELETE' }),

  listGames: () => request<{ games: GameInfo[] }>('/alphazero/games'),
  listOpponents: (gameId: string) => request<{ opponents: ArenaOpponent[] }>(`/alphazero/opponents?game_id=${gameId}`),
  newArenaSession: (data: {
    game_id: string
    opponent_id: string
    opponent_source: string
    human_first: boolean
    num_simulations?: number
  }) => request<ArenaState>('/alphazero/session', { method: 'POST', body: JSON.stringify(data) }),
  arenaMove: (sessionId: string, action: number) =>
    request<ArenaState>(`/alphazero/session/${sessionId}/move`, {
      method: 'POST',
      body: JSON.stringify({ action }),
    }),
  deleteArenaSession: (sessionId: string) =>
    request<{ success: boolean }>(`/alphazero/session/${sessionId}`, { method: 'DELETE' }),

  listPluginTemplates: () => request<{ templates: PluginTemplate[] }>('/plugins/templates'),
  listPluginScripts: (kind: PluginKind) => request<{ scripts: PluginScriptMeta[] }>(`/plugins/${kind}`),
  getPluginScript: (kind: PluginKind, slug: string) =>
    request<{ slug: string; code: string }>(`/plugins/${kind}/${slug}`),
  savePluginScript: (kind: PluginKind, slug: string, code: string) =>
    request<{ success: boolean }>(`/plugins/${kind}/${slug}`, { method: 'PUT', body: JSON.stringify({ code }) }),
  deletePluginScript: (kind: PluginKind, slug: string) =>
    request<{ success: boolean }>(`/plugins/${kind}/${slug}`, { method: 'DELETE' }),
  validatePluginScript: (kind: PluginKind, slug: string, code: string) =>
    request<ValidateResult>(`/plugins/${kind}/${slug}/validate`, { method: 'POST', body: JSON.stringify({ code }) }),

  listNetworkFamilies: () => request<{ families: NetworkFamilyInfo[] }>('/networks/families'),
  listNetworks: () => request<{ networks: NetworkMeta[] }>('/networks'),
  getNetwork: (slug: string) => request<{ slug: string } & NetworkDoc>(`/networks/${slug}`),
  saveNetwork: (slug: string, doc: NetworkDoc) =>
    request<{ success: boolean }>(`/networks/${slug}`, { method: 'PUT', body: JSON.stringify(doc) }),
  deleteNetwork: (slug: string) => request<{ success: boolean }>(`/networks/${slug}`, { method: 'DELETE' }),
  previewNetwork: (payload: NetworkPreviewRequest) =>
    request<NetworkPreviewResult>('/networks/preview', { method: 'POST', body: JSON.stringify(payload) }),

  listScenes: () => request<{ scenes: SceneMeta[] }>('/scenes'),
  getDefaultScene: () => request<SceneSpec>('/scenes/default'),
  getScene: (slug: string) => request<{ slug: string } & SceneSpec>(`/scenes/${slug}`),
  saveScene: (slug: string, doc: SceneSpec) =>
    request<{ success: boolean; id: string }>(`/scenes/${slug}`, { method: 'PUT', body: JSON.stringify(doc) }),
  deleteScene: (slug: string) => request<{ success: boolean }>(`/scenes/${slug}`, { method: 'DELETE' }),
}

export function createMetricsWebSocket(
  runId: string,
  onMessage: (data: import('./types').MetricsSnapshot) => void,
  onError?: (err: Event) => void,
): Promise<WebSocket> {
  return resolvePort().then((port) => {
    const wsUrl = window.electronAPI
      ? `ws://127.0.0.1:${port}/ws/metrics/${runId}`
      : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/metrics/${runId}`
    const ws = new WebSocket(wsUrl)

    ws.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data))
      } catch { /* ignore malformed */ }
    }
    ws.onerror = (err) => onError?.(err)
    return ws
  })
}

export { ApiError }

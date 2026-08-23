import type {
  AlgorithmSpec,
  ArenaState,
  ArenaOpponent,
  EnvSpec,
  ExperimentConfig,
  GameInfo,
  ModelInfo,
  RunSummary,
  SystemInfo,
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

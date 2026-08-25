import { app } from 'electron'
import fs from 'fs'
import path from 'path'

export type BackendMode = 'native' | 'docker' | 'auto'

export interface AppConfig {
  backendMode: BackendMode
  dockerAutoBuild: boolean
  dockerGpu: boolean
  // Native mode counterpart to dockerGpu: makes the app's private venv
  // install rl_core/requirements-gpu.txt (CUDA torch) instead of the default
  // CPU-only requirements.txt. Requires a backend restart to take effect —
  // see startBackendNative(), which picks the requirements file from this.
  nativeGpu: boolean
  // Gates the CPU/GPU picker shown on first launch (see main.ts's
  // app.whenReady()) — false means "haven't asked yet", so we show the
  // picker instead of silently starting an install. Flips to true as soon
  // as the user picks either option; from then on this app behaves exactly
  // like before (auto-starts on launch, Settings has the same two toggles
  // for changing your mind later).
  setupComplete: boolean
}

const DEFAULT_CONFIG: AppConfig = {
  backendMode: 'auto',
  dockerAutoBuild: true,
  dockerGpu: false,
  nativeGpu: false,
  setupComplete: false,
}

function configPath(): string {
  return path.join(app.getPath('userData'), 'config.json')
}

export function loadConfig(): AppConfig {
  try {
    const raw = fs.readFileSync(configPath(), 'utf-8')
    return { ...DEFAULT_CONFIG, ...JSON.parse(raw) }
  } catch {
    return { ...DEFAULT_CONFIG }
  }
}

export function saveConfig(patch: Partial<AppConfig>): AppConfig {
  const current = loadConfig()
  const next = { ...current, ...patch }
  try {
    fs.mkdirSync(path.dirname(configPath()), { recursive: true })
    fs.writeFileSync(configPath(), JSON.stringify(next, null, 2))
  } catch (err) {
    console.error('[config] save error:', err)
  }
  return next
}

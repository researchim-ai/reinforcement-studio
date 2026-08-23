import { app } from 'electron'
import fs from 'fs'
import path from 'path'

export type BackendMode = 'native' | 'docker' | 'auto'

export interface AppConfig {
  backendMode: BackendMode
  dockerAutoBuild: boolean
  dockerGpu: boolean
}

const DEFAULT_CONFIG: AppConfig = {
  backendMode: 'auto',
  dockerAutoBuild: true,
  dockerGpu: false,
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

export type BackendMode = 'native' | 'docker' | 'auto'

export interface AppConfigShape {
  backendMode: BackendMode
  dockerAutoBuild: boolean
  dockerGpu: boolean
  nativeGpu: boolean
  setupComplete: boolean
}

export interface DetectedGpu {
  index: number
  name: string
  memoryTotalMb: number
}

export type BootPhase =
  | { phase: 'starting' }
  | { phase: 'awaiting-setup'; gpus: DetectedGpu[] }
  | { phase: 'checking-docker' }
  | { phase: 'docker-unavailable'; detail?: string }
  | { phase: 'docker-no-image' }
  | { phase: 'building-image'; line?: string }
  | { phase: 'starting-container' }
  | { phase: 'waiting-container-health'; attempt: number }
  | { phase: 'starting-python' }
  | { phase: 'creating-venv'; line?: string }
  | { phase: 'installing-dependencies'; line?: string }
  | { phase: 'python-starting'; line?: string }
  | { phase: 'ready'; mode: 'native' | 'docker'; port: number }
  | { phase: 'failed'; error: string }

export interface ElectronAPI {
  backend: {
    status: () => Promise<{
      running: boolean
      port: number
      error?: string | null
      mode?: 'native' | 'docker' | 'offline'
      phase?: BootPhase
    }>
    restart: () => Promise<{ success: boolean; error?: string; port: number; mode?: string }>
    port: () => Promise<number>
    bootPhase: () => Promise<BootPhase>
    openLogs: () => Promise<void>
    onBootPhase: (cb: (phase: BootPhase) => void) => () => void
  }
  config: {
    get: () => Promise<AppConfigShape>
    set: (patch: Partial<AppConfigShape>) => Promise<AppConfigShape>
    setBackendMode: (mode: BackendMode) => Promise<AppConfigShape>
    setNativeGpu: (gpu: boolean) => Promise<AppConfigShape>
  }
  setup: {
    detectGpus: () => Promise<DetectedGpu[]>
    choose: (device: 'cpu' | 'gpu') => Promise<AppConfigShape>
  }
  docker: {
    status: () => Promise<{ running: boolean; containerId?: string; hostPort?: number; error?: string }>
    available: () => Promise<boolean>
    imageExists: () => Promise<boolean>
    start: () => Promise<{ success: boolean; error?: string; hostPort?: number }>
    stop: () => Promise<{ success: boolean; error?: string }>
    build: () => Promise<{ success: boolean; error?: string }>
    logs: (tail?: number) => Promise<string>
    onBuildProgress: (cb: (line: string) => void) => () => void
  }
  app: {
    version: () => Promise<string>
    platform: () => Promise<string>
    paths: () => Promise<{ userData: string; logs: string; temp: string }>
  }
  shell: {
    openExternal: (url: string) => Promise<void>
    showInFolder: (p: string) => Promise<void>
  }
  dialog: {
    pickDirectory: () => Promise<{ canceled: boolean; filePaths: string[] }>
  }
}

declare global {
  interface Window {
    electronAPI?: ElectronAPI
  }
}

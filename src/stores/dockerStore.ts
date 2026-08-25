import { create } from 'zustand'
import type { BackendMode, BootPhase } from '@/types/electron'

export type SetupDevice = 'cpu' | 'gpu'

interface DockerState {
  status: 'unknown' | 'running' | 'stopped' | 'building' | 'error'
  backendOnline: boolean
  backendPort: number
  backendError: string | null
  backendMode: 'native' | 'docker' | 'offline'
  bootPhase: BootPhase
  containerId: string | null
  error: string | null
  config: {
    backendMode: BackendMode
    dockerAutoBuild: boolean
    dockerGpu: boolean
    nativeGpu: boolean
    setupComplete: boolean
  } | null
  choosingSetup: boolean

  setBootPhase: (phase: BootPhase) => void
  checkBackend: () => Promise<void>
  checkDocker: () => Promise<void>
  restartBackend: () => Promise<void>
  loadConfig: () => Promise<void>
  setBackendMode: (mode: BackendMode) => Promise<void>
  setDockerGpu: (gpu: boolean) => Promise<void>
  setNativeGpu: (gpu: boolean) => Promise<void>
  chooseSetupDevice: (device: SetupDevice) => Promise<void>
  subscribeToBootPhase: () => () => void
}

export const useDockerStore = create<DockerState>((set) => ({
  status: 'unknown',
  backendOnline: false,
  backendPort: 8000,
  backendError: null,
  backendMode: 'offline',
  bootPhase: { phase: 'starting' },
  containerId: null,
  error: null,
  config: null,
  choosingSetup: false,

  setBootPhase: (bootPhase) => set({ bootPhase }),

  checkBackend: async () => {
    if (window.electronAPI) {
      try {
        const result = await window.electronAPI.backend.status()
        set({
          backendOnline: result.running,
          backendPort: result.port ?? 8000,
          backendError: result.error ?? null,
          backendMode: result.mode ?? 'offline',
          bootPhase: result.phase ?? { phase: 'starting' },
        })
      } catch {
        set({ backendOnline: false })
      }
    } else {
      try {
        const res = await fetch('/api/system/health')
        set({ backendOnline: res.ok })
      } catch {
        set({ backendOnline: false })
      }
    }
  },

  checkDocker: async () => {
    if (window.electronAPI) {
      const result = await window.electronAPI.docker.status()
      set({
        status: result.running ? 'running' : 'stopped',
        containerId: result.containerId ?? null,
        error: result.error ?? null,
      })
    }
  },

  restartBackend: async () => {
    if (window.electronAPI) {
      set({ backendOnline: false, backendError: null, bootPhase: { phase: 'starting' } })
      const result = await window.electronAPI.backend.restart()
      set({
        backendOnline: result.success,
        backendPort: result.port ?? 8000,
        backendError: result.error ?? null,
      })
    }
  },

  loadConfig: async () => {
    if (window.electronAPI) {
      const cfg = await window.electronAPI.config.get()
      set({ config: cfg })
    }
  },

  setBackendMode: async (mode) => {
    if (!window.electronAPI) return
    set({ backendOnline: false, bootPhase: { phase: 'starting' } })
    const cfg = await window.electronAPI.config.setBackendMode(mode)
    set({ config: cfg })
  },

  setDockerGpu: async (gpu) => {
    if (!window.electronAPI) return
    const cfg = await window.electronAPI.config.set({ dockerGpu: gpu })
    set({ config: cfg })
  },

  setNativeGpu: async (gpu) => {
    if (!window.electronAPI) return
    // Unlike setDockerGpu, this restarts the backend immediately (main
    // process side) — it's what makes ensurePythonEnv() notice
    // requirements-gpu.txt vs requirements.txt changed and actually
    // reinstall torch, same as switching backend mode.
    set({ backendOnline: false, bootPhase: { phase: 'starting' } })
    const cfg = await window.electronAPI.config.setNativeGpu(gpu)
    set({ config: cfg })
  },

  chooseSetupDevice: async (device) => {
    if (!window.electronAPI) return
    set({ choosingSetup: true, bootPhase: { phase: 'starting' } })
    const cfg = await window.electronAPI.setup.choose(device)
    set({ config: cfg, choosingSetup: false })
  },

  subscribeToBootPhase: () => {
    if (!window.electronAPI) return () => {}
    return window.electronAPI.backend.onBootPhase((phase) => {
      set({ bootPhase: phase })
      if (phase.phase === 'ready') {
        set({ backendOnline: true, backendPort: phase.port, backendMode: phase.mode, backendError: null })
      } else if (phase.phase === 'failed') {
        set({ backendOnline: false, backendError: phase.error })
      }
    })
  },
}))

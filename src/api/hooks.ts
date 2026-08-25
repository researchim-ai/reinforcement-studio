import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api } from './client'
import type { EnvKind, NetworkFamily, NetworkSpec, PluginKind, WrapperNode } from './types'

export function useEnvironments() {
  return useQuery({ queryKey: ['environments'], queryFn: api.listEnvironments })
}

export function useWrappers() {
  return useQuery({ queryKey: ['wrappers'], queryFn: api.listWrappers })
}

export function useAlgorithms() {
  return useQuery({ queryKey: ['algorithms'], queryFn: api.listAlgorithms })
}

export function useRuns(pollMs = 3000) {
  return useQuery({ queryKey: ['runs'], queryFn: api.listRuns, refetchInterval: pollMs })
}

export function useRun(runId: string | undefined, pollMs = 2000) {
  return useQuery({
    queryKey: ['run', runId],
    queryFn: () => api.getRun(runId as string),
    enabled: !!runId,
    refetchInterval: pollMs,
  })
}

export function useModels() {
  return useQuery({ queryKey: ['models'], queryFn: api.listModels })
}

export function useGames() {
  return useQuery({ queryKey: ['games'], queryFn: api.listGames })
}

export function useOpponents(gameId: string | undefined) {
  return useQuery({
    queryKey: ['opponents', gameId],
    queryFn: () => api.listOpponents(gameId as string),
    enabled: !!gameId,
  })
}

export function useSystemInfo() {
  return useQuery({ queryKey: ['system-info'], queryFn: api.systemInfo, refetchInterval: 5000 })
}

export function usePluginTemplates() {
  return useQuery({ queryKey: ['plugin-templates'], queryFn: api.listPluginTemplates })
}

export function usePluginScripts(kind: PluginKind) {
  return useQuery({ queryKey: ['plugin-scripts', kind], queryFn: () => api.listPluginScripts(kind) })
}

export function useNetworkFamilies() {
  return useQuery({ queryKey: ['network-families'], queryFn: api.listNetworkFamilies })
}

export function useNetworks() {
  return useQuery({ queryKey: ['networks'], queryFn: api.listNetworks })
}

/** Delays propagating `value` until it's stayed stable for `delayMs` — used
 * to avoid firing a backend request on every keystroke while the user is
 * still tweaking a hyperparameter in the Designer. */
function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs)
    return () => clearTimeout(id)
  }, [value, delayMs])
  return debounced
}

/** Live env/network preview for the Experiment Designer — builds the
 * (wrapped) env and the algorithm's network on the backend, without running
 * any training, so input/output dims + layer summary + param count can be
 * tracked while the user is still designing. */
export function useInspectDesign(payload: {
  kind: EnvKind
  environment: { id: string; wrappers?: { type: string; params: Record<string, number | string> }[] }
  algorithm: { id: string; hyperparams: Record<string, number> }
} | null) {
  const debounced = useDebouncedValue(payload, 400)
  const key = debounced ? JSON.stringify(debounced) : null
  return useQuery({
    queryKey: ['inspect-design', key],
    queryFn: () => api.inspectDesign(debounced as NonNullable<typeof debounced>),
    enabled: !!key,
    placeholderData: (prev) => prev,
  })
}

/** Live shape-inference preview for the Network Builder — re-checks the
 * spec against a real env/game's observation & action space on every
 * change (debounced), so per-layer shapes/param count/errors update as the
 * user edits the trunk or a head. */
export function useNetworkPreview(payload: {
  family: NetworkFamily
  spec: NetworkSpec
  environmentId?: string | null
  wrappers?: WrapperNode[]
  gameId?: string | null
} | null) {
  const debounced = useDebouncedValue(payload, 350)
  const key = debounced ? JSON.stringify(debounced) : null
  return useQuery({
    queryKey: ['network-preview', key],
    queryFn: () =>
      api.previewNetwork({
        family: debounced!.family,
        spec: debounced!.spec,
        environment_id: debounced!.environmentId,
        wrappers: debounced!.wrappers,
        game_id: debounced!.gameId,
      }),
    enabled: !!key && (!!debounced?.environmentId || !!debounced?.gameId),
    placeholderData: (prev) => prev,
  })
}

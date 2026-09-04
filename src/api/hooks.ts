import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api } from './client'
import type { AnyNetworkSpec, EnvKind, NetworkFamily, PluginKind, WrapperNode } from './types'

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

/** The exact architecture a run (or a promoted checkpoint) actually
 * trained with (`network.json`) — used by the "Сохранить архитектуру"
 * action on the Training Monitor / Model Zoo to feed `api.saveNetwork`
 * without re-deriving the spec from `config.json`. */
export function useRunNetwork(runId: string | undefined) {
  return useQuery({
    queryKey: ['run-network', runId],
    queryFn: () => api.getRunNetwork(runId as string),
    enabled: !!runId,
  })
}

/** Full compound torch module tree captured once from the live algorithm. */
export function useRunArchitecture(runId: string | undefined) {
  return useQuery({
    queryKey: ['run-architecture', runId],
    queryFn: () => api.getRunArchitecture(runId as string),
    enabled: !!runId,
  })
}

export function useCheckpointNetwork(name: string | undefined) {
  return useQuery({
    queryKey: ['checkpoint-network', name],
    queryFn: () => api.getCheckpointNetwork(name as string),
    enabled: !!name,
  })
}

/** The full `config.json` a run started with — used by the "Дообучить"
 * (resume) flow to preselect/lock the exact environment/wrappers/
 * hyperparams/network a source run trained with. */
export function useRunConfig(runId: string | undefined) {
  return useQuery({
    queryKey: ['run-config', runId],
    queryFn: () => api.getRunConfig(runId as string),
    enabled: !!runId,
  })
}

export function useCheckpointConfig(name: string | undefined) {
  return useQuery({
    queryKey: ['checkpoint-config', name],
    queryFn: () => api.getCheckpointConfig(name as string),
    enabled: !!name,
  })
}

/** Full metrics history for several runs at once — powers the Compare Runs
 * overlay chart (Training Monitor) and the Sweeps results view, both of
 * which need every selected run's whole curve, not just the latest
 * snapshot. */
export function useMultiRunHistory(runIds: string[]) {
  return useQuery({
    queryKey: ['multi-run-history', [...runIds].sort()],
    queryFn: async () => {
      const entries = await Promise.all(
        runIds.map(async (id) => [id, (await api.getMetricsHistory(id)).history] as const),
      )
      return Object.fromEntries(entries) as Record<string, import('./types').MetricsSnapshot[]>
    },
    enabled: runIds.length > 0,
  })
}

export function useSweeps(pollMs = 4000) {
  return useQuery({ queryKey: ['sweeps'], queryFn: api.listSweeps, refetchInterval: pollMs })
}

export function useSweep(sweepId: string | undefined, pollMs = 3000) {
  return useQuery({
    queryKey: ['sweep', sweepId],
    queryFn: () => api.getSweep(sweepId as string),
    enabled: !!sweepId,
    refetchInterval: pollMs,
  })
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

export function useWorldModelTypes() {
  return useQuery({ queryKey: ['world-model-types'], queryFn: api.listWorldModelTypes })
}

export function useWorldModels() {
  return useQuery({ queryKey: ['world-models'], queryFn: api.listWorldModels })
}

/** The exact World Model spec (`world_model.json`) a run actually
 * resolved/used — counterpart to `useRunNetwork` for the four World Model
 * algorithms and standalone `kind: "world_model"` runs. */
export function useRunWorldModel(runId: string | undefined) {
  return useQuery({
    queryKey: ['run-world-model', runId],
    queryFn: () => api.getRunWorldModel(runId as string),
    enabled: !!runId,
  })
}

export function useScenes() {
  return useQuery({ queryKey: ['scenes'], queryFn: api.listScenes })
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
  algorithm: {
    id: string
    hyperparams: Record<string, number>
    network_spec_id?: string | null
    network_spec?: AnyNetworkSpec | null
  }
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
  spec: AnyNetworkSpec
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

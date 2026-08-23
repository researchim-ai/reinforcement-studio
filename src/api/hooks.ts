import { useQuery } from '@tanstack/react-query'
import { api } from './client'

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

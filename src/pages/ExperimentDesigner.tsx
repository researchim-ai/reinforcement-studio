import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ReactFlow, Background, Controls, type Node, type Edge } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { toast } from 'sonner'
import { Loader2, Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useAlgorithms, useEnvironments, useInspectDesign, useWrappers } from '@/api/hooks'
import { api } from '@/api/client'
import { EnvNode, type EnvNodeData } from '@/components/designer/EnvNode'
import { WrapperNode, type WrapperNodeData } from '@/components/designer/WrapperNode'
import { AlgorithmNode, type AlgorithmNodeData } from '@/components/designer/AlgorithmNode'
import { TrainingNode, type TrainingNodeData } from '@/components/designer/TrainingNode'
import { InspectPanel } from '@/components/designer/InspectPanel'
import type { EnvKind, WrapperNode as WrapperNodeSpec } from '@/api/types'

const nodeTypes = {
  env: EnvNode,
  wrapper: WrapperNode,
  algorithm: AlgorithmNode,
  training: TrainingNode,
}

const COL_X = [40, 340, 660, 1010]
const ROW_Y = 60
const WRAPPER_GAP = 200

export function ExperimentDesigner() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const requestedEnvId = searchParams.get('env')
  const { data: envData, isLoading: envsLoading } = useEnvironments()
  const { data: wrapperData } = useWrappers()
  const { data: algoData } = useAlgorithms()

  const environments = envData?.environments ?? []
  const wrapperCatalog = wrapperData?.wrappers ?? []
  const allAlgorithms = algoData?.algorithms ?? []

  const [environmentId, setEnvironmentId] = useState<string>('')
  const [wrappers, setWrappers] = useState<WrapperNodeSpec[]>([])
  const [algorithmId, setAlgorithmId] = useState<string>('')
  const [hyperparams, setHyperparams] = useState<Record<string, number>>({})
  const [name, setName] = useState('Мой эксперимент')
  const [totalTimesteps, setTotalTimesteps] = useState(50_000)
  const [numIterations, setNumIterations] = useState(20)
  const [seed, setSeed] = useState(42)
  const [useGpu, setUseGpu] = useState(false)
  const [starting, setStarting] = useState(false)

  const selectedEnv = environments.find((e) => e.id === environmentId)
  const kind: EnvKind = selectedEnv?.kind ?? 'gym'
  const compatibleAlgorithms = allAlgorithms.filter((a) => {
    if (a.kind !== kind) return false
    if (!a.is_custom) return !selectedEnv || selectedEnv.compatible_algorithms.includes(a.id)
    // Custom algorithms don't appear in any built-in env's compatible_algorithms
    // list — for the Gym track, match on action kind instead (declared by the
    // plugin's SUPPORTED_ACTION_KINDS); AlphaZero board games have no action
    // kind concept, so custom AlphaZero trainers are always considered compatible.
    if (a.kind === 'gym') {
      return !selectedEnv || !a.supported_action_kinds || a.supported_action_kinds.includes(selectedEnv.action_kind)
    }
    return true
  })

  // Auto-pick a default env once the catalog loads — prefer whatever was
  // requested from the Environments gallery ("Использовать в дизайнере"),
  // falling back to the first available one otherwise.
  useEffect(() => {
    if (!environmentId && environments.length > 0) {
      const requested = requestedEnvId ? environments.find((e) => e.id === requestedEnvId) : undefined
      const first = requested ?? environments.find((e) => e.available) ?? environments[0]
      setEnvironmentId(first.id)
      if (requestedEnvId) {
        setSearchParams((prev) => {
          prev.delete('env')
          return prev
        }, { replace: true })
      }
    }
  }, [environments, environmentId, requestedEnvId, setSearchParams])

  useEffect(() => {
    if (compatibleAlgorithms.length > 0 && !compatibleAlgorithms.some((a) => a.id === algorithmId)) {
      setAlgorithmId(compatibleAlgorithms[0].id)
      const defaults: Record<string, number> = {}
      for (const hp of compatibleAlgorithms[0].hyperparams) defaults[hp.key] = hp.default
      setHyperparams(defaults)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compatibleAlgorithms.map((a) => a.id).join(','), algorithmId])

  const handleEnvChange = useCallback((id: string) => {
    setEnvironmentId(id)
    setAlgorithmId('')
    setWrappers([])
  }, [])

  const handleAlgoChange = useCallback(
    (id: string) => {
      setAlgorithmId(id)
      const spec = allAlgorithms.find((a) => a.id === id)
      const defaults: Record<string, number> = {}
      for (const hp of spec?.hyperparams ?? []) defaults[hp.key] = hp.default
      setHyperparams(defaults)
    },
    [allAlgorithms],
  )

  const addWrapper = useCallback(() => {
    if (wrapperCatalog.length === 0) return
    setWrappers((prev) => [...prev, { type: wrapperCatalog[0].type, params: wrapperCatalog[0].params }])
  }, [wrapperCatalog])

  const removeWrapper = useCallback((index: number) => {
    setWrappers((prev) => prev.filter((_, i) => i !== index))
  }, [])

  const changeWrapperType = useCallback(
    (index: number, type: string) => {
      const spec = wrapperCatalog.find((w) => w.type === type)
      setWrappers((prev) => prev.map((w, i) => (i === index ? { type, params: spec?.params ?? {} } : w)))
    },
    [wrapperCatalog],
  )

  const inspectPayload = useMemo(() => {
    if (!environmentId || !algorithmId) return null
    return {
      kind,
      environment: { id: environmentId, wrappers: kind === 'gym' ? wrappers : undefined },
      algorithm: { id: algorithmId, hyperparams },
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [environmentId, algorithmId, kind, JSON.stringify(wrappers), JSON.stringify(hyperparams)])
  const { data: inspectData, isFetching: inspectFetching } = useInspectDesign(inspectPayload)

  const canRun = !!environmentId && !!algorithmId && !starting

  const handleRun = useCallback(async () => {
    if (!canRun) return
    setStarting(true)
    try {
      const config = {
        kind,
        name,
        environment: { id: environmentId, wrappers: kind === 'gym' ? wrappers : [] },
        algorithm: { id: algorithmId, hyperparams },
        training: {
          total_timesteps: totalTimesteps,
          num_iterations: numIterations,
          seed,
          use_gpu: useGpu,
        },
      }
      const { run_id } = await api.startRun(config)
      toast.success(`Запуск создан: ${run_id}`)
      navigate(`/monitor?run=${run_id}`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось запустить обучение')
    } finally {
      setStarting(false)
    }
  }, [canRun, kind, name, environmentId, wrappers, algorithmId, hyperparams, totalTimesteps, numIterations, seed, useGpu, navigate])

  const algoColumnIndex = 2
  const trainingColumnIndex = 3
  const wrapperColumnBase = COL_X[1]

  const nodes: Node[] = useMemo(() => {
    const list: Node[] = []

    list.push({
      id: 'env',
      type: 'env',
      position: { x: COL_X[0], y: ROW_Y },
      data: {
        environments,
        selectedId: environmentId,
        onChange: handleEnvChange,
      } satisfies EnvNodeData,
    })

    wrappers.forEach((w, i) => {
      list.push({
        id: `wrapper-${i}`,
        type: 'wrapper',
        position: { x: wrapperColumnBase + i * WRAPPER_GAP, y: ROW_Y },
        data: {
          catalog: wrapperCatalog,
          type: w.type,
          onChange: (type: string) => changeWrapperType(i, type),
          onRemove: () => removeWrapper(i),
        } satisfies WrapperNodeData,
      })
    })

    const algoX = kind === 'gym' ? wrapperColumnBase + wrappers.length * WRAPPER_GAP + (wrappers.length ? 40 : 0) : COL_X[algoColumnIndex]

    list.push({
      id: 'algorithm',
      type: 'algorithm',
      position: { x: algoX, y: ROW_Y },
      data: {
        algorithms: compatibleAlgorithms,
        selectedId: algorithmId,
        hyperparams,
        onChangeAlgo: handleAlgoChange,
        onChangeHyperparam: (key: string, value: number) =>
          setHyperparams((prev) => ({ ...prev, [key]: value })),
      } satisfies AlgorithmNodeData,
    })

    list.push({
      id: 'training',
      type: 'training',
      position: { x: algoX + 340, y: ROW_Y },
      data: {
        kind,
        name,
        totalTimesteps,
        numIterations,
        seed,
        useGpu,
        starting,
        disabled: !canRun,
        onChangeName: setName,
        onChangeTotalTimesteps: setTotalTimesteps,
        onChangeNumIterations: setNumIterations,
        onChangeSeed: setSeed,
        onChangeUseGpu: setUseGpu,
        onRun: handleRun,
      } satisfies TrainingNodeData,
    })

    return list
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    environments, environmentId, wrappers, wrapperCatalog, compatibleAlgorithms, algorithmId,
    hyperparams, kind, name, totalTimesteps, numIterations, seed, useGpu, starting, canRun,
  ])

  const edges: Edge[] = useMemo(() => {
    const list: Edge[] = []
    const chain = ['env', ...wrappers.map((_, i) => `wrapper-${i}`), 'algorithm', 'training']
    for (let i = 0; i < chain.length - 1; i++) {
      list.push({ id: `${chain[i]}-${chain[i + 1]}`, source: chain[i], target: chain[i + 1], animated: true })
    }
    return list
  }, [wrappers])

  if (envsLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="relative h-full w-full">
      <div className="absolute left-4 top-4 z-10 flex gap-2">
        {kind === 'gym' && (
          <Button size="sm" variant="outline" onClick={addWrapper} disabled={wrapperCatalog.length === 0}>
            <Plus className="h-3.5 w-3.5" />
            Добавить wrapper
          </Button>
        )}
      </div>
      <InspectPanel data={inspectData} isFetching={inspectFetching} />
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        proOptions={{ hideAttribution: true }}
        colorMode="dark"
        fitView
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  )
}

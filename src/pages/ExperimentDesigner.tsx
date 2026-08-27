import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ReactFlow, Background, Controls, type Node, type Edge } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { toast } from 'sonner'
import { Loader2, Plus, Sparkles, X, FlaskConical } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useAlgorithms, useCheckpointConfig, useEnvironments, useInspectDesign, useNetworks, useRunConfig, useWrappers } from '@/api/hooks'
import { api } from '@/api/client'
import { EnvNode, type EnvNodeData } from '@/components/designer/EnvNode'
import { WrapperNode, type WrapperNodeData } from '@/components/designer/WrapperNode'
import { AlgorithmNode, type AlgorithmNodeData } from '@/components/designer/AlgorithmNode'
import { TrainingNode, type TrainingNodeData } from '@/components/designer/TrainingNode'
import { InspectPanel } from '@/components/designer/InspectPanel'
import { AddNode, type AddNodeData } from '@/components/flow/AddNode'
import { SweepDialog } from '@/components/designer/SweepDialog'
import { useNodePositions } from '@/lib/useNodePositions'
import { requiredFamilyFor, quickSpecForFamily } from '@/lib/networkBuilder'
import type { AlgorithmSpec, EnvKind, EnvSpec, NetworkSpec, ResumeFrom, WrapperNode as WrapperNodeSpec } from '@/api/types'

/** Algorithm hyperparam defaults, with the selected env's overrides (if any)
 * layered on top — e.g. Gomoku ships a much bigger MCTS/network budget than
 * Tic-Tac-Toe so AlphaZero actually has a chance to learn within a
 * reasonable number of iterations instead of sitting at a near-uniform
 * policy the whole run. */
function buildDefaultHyperparams(spec: AlgorithmSpec | undefined, env: EnvSpec | undefined): Record<string, number> {
  const defaults: Record<string, number> = {}
  for (const hp of spec?.hyperparams ?? []) defaults[hp.key] = hp.default
  if (env?.default_hyperparams) Object.assign(defaults, env.default_hyperparams)
  return defaults
}

const nodeTypes = {
  env: EnvNode,
  wrapper: WrapperNode,
  algorithm: AlgorithmNode,
  training: TrainingNode,
  add: AddNode,
}

const COL_X = [40, 340, 660, 1010]
const ROW_Y = 60
const WRAPPER_GAP = 200

export function ExperimentDesigner() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const requestedEnvId = searchParams.get('env')
  const resumeRunId = searchParams.get('resumeRun')
  const resumeCheckpointName = searchParams.get('resumeCheckpoint')
  const { data: envData, isLoading: envsLoading } = useEnvironments()
  const { data: wrapperData } = useWrappers()
  const { data: algoData } = useAlgorithms()
  const { data: networkData } = useNetworks()

  const environments = envData?.environments ?? []
  const wrapperCatalog = wrapperData?.wrappers ?? []
  const allAlgorithms = algoData?.algorithms ?? []
  const networks = networkData?.networks ?? []

  const [environmentId, setEnvironmentId] = useState<string>('')
  const [wrappers, setWrappers] = useState<WrapperNodeSpec[]>([])
  const [algorithmId, setAlgorithmId] = useState<string>('')
  const [hyperparams, setHyperparams] = useState<Record<string, number>>({})
  const [networkSpecId, setNetworkSpecId] = useState<string | null>(null)
  const [quickHiddenLayers, setQuickHiddenLayers] = useState<number[] | null>(null)
  const [name, setName] = useState('Мой эксперимент')
  const [totalTimesteps, setTotalTimesteps] = useState(50_000)
  const [numEnvs, setNumEnvs] = useState(1)
  const [numIterations, setNumIterations] = useState(20)
  const [seed, setSeed] = useState(42)
  const [useGpu, setUseGpu] = useState(false)
  const [starting, setStarting] = useState(false)
  const [sweepOpen, setSweepOpen] = useState(false)

  // "Дообучить" (resume/fine-tune) — set once the source run's/checkpoint's
  // config has loaded and been applied below. While set, the architecture
  // and hyperparams are locked (AlgorithmNode/EnvNode/WrapperNode go
  // read-only) since loading saved weights into a differently-shaped
  // network fails outright — only `total_timesteps`/`num_envs`/`seed`
  // remain editable. `resumeNetworkSpec` carries the exact inline
  // `network_spec` (if any) the source used, bypassing the quick-editor/
  // catalog machinery entirely since it must match exactly.
  const [resumeFrom, setResumeFrom] = useState<ResumeFrom | null>(null)
  const [resumeApplied, setResumeApplied] = useState(false)
  const [resumeNetworkSpec, setResumeNetworkSpec] = useState<NetworkSpec | null>(null)
  const resumePending = (!!resumeRunId || !!resumeCheckpointName) && !resumeApplied
  const { data: resumeRunConfig } = useRunConfig(resumeRunId ?? undefined)
  const { data: resumeCheckpointConfig } = useCheckpointConfig(resumeCheckpointName ?? undefined)

  useEffect(() => {
    const sourceConfig = resumeRunId ? resumeRunConfig : resumeCheckpointConfig
    if (!sourceConfig || resumeApplied) return
    setEnvironmentId(sourceConfig.environment.id)
    setWrappers(sourceConfig.environment.wrappers ?? [])
    setAlgorithmId(sourceConfig.algorithm.id)
    setHyperparams(sourceConfig.algorithm.hyperparams ?? {})
    setNetworkSpecId(sourceConfig.algorithm.network_spec_id ?? null)
    setQuickHiddenLayers(null)
    setResumeNetworkSpec(sourceConfig.algorithm.network_spec ?? null)
    if (sourceConfig.training?.num_envs) setNumEnvs(sourceConfig.training.num_envs)
    setResumeFrom({ source: resumeRunId ? 'run' : 'checkpoint', id: (resumeRunId ?? resumeCheckpointName) as string })
    setResumeApplied(true)
    setSearchParams((prev) => {
      prev.delete('resumeRun')
      prev.delete('resumeCheckpoint')
      return prev
    }, { replace: true })
  }, [resumeRunId, resumeCheckpointName, resumeRunConfig, resumeCheckpointConfig, resumeApplied, setSearchParams])

  const selectedEnv = environments.find((e) => e.id === environmentId)
  const kind: EnvKind = selectedEnv?.kind ?? 'gym'
  // Memoized — this ran fresh on *every* render before (including every
  // pointer-move frame while dragging a node), which handed the `nodes`
  // useMemo below a brand-new array reference each time and forced it to
  // rebuild the entire node list (all callbacks, all data objects) 60+
  // times/sec during a drag instead of just moving one node, which is what
  // actually made everything look janky/flickery while dragging.
  const compatibleAlgorithms = useMemo(
    () =>
      allAlgorithms.filter((a) => {
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
      }),
    [allAlgorithms, kind, selectedEnv],
  )

  // Auto-pick a default env once the catalog loads — prefer whatever was
  // requested from the Environments gallery ("Использовать в дизайнере"),
  // falling back to the first available one otherwise.
  useEffect(() => {
    if (resumePending) return
    if (!environmentId && environments.length > 0) {
      const requested = requestedEnvId ? environments.find((e) => e.id === requestedEnvId) : undefined
      const first = requested ?? environments.find((e) => e.available) ?? environments[0]
      setEnvironmentId(first.id)
      if (first.recommended_total_timesteps) setTotalTimesteps(first.recommended_total_timesteps)
      if (first.recommended_num_envs) setNumEnvs(first.recommended_num_envs)
      if (requestedEnvId) {
        setSearchParams((prev) => {
          prev.delete('env')
          return prev
        }, { replace: true })
      }
    }
  }, [environments, environmentId, requestedEnvId, resumePending, setSearchParams])

  useEffect(() => {
    if (resumePending) return
    if (compatibleAlgorithms.length > 0 && !compatibleAlgorithms.some((a) => a.id === algorithmId)) {
      setAlgorithmId(compatibleAlgorithms[0].id)
      setHyperparams(buildDefaultHyperparams(compatibleAlgorithms[0], selectedEnv))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compatibleAlgorithms.map((a) => a.id).join(','), algorithmId, selectedEnv, resumePending])

  const handleEnvChange = useCallback((id: string) => {
    setEnvironmentId(id)
    setAlgorithmId('')
    setNetworkSpecId(null)
    setQuickHiddenLayers(null)
    // Pre-fill a per-env recommended step budget (e.g. CarRacing/Atari need
    // millions, not the flat 50k default) — only when the field is set,
    // otherwise leave whatever the user already had.
    const nextEnv = environments.find((e) => e.id === id)
    if (nextEnv?.recommended_total_timesteps) setTotalTimesteps(nextEnv.recommended_total_timesteps)
    if (nextEnv?.recommended_num_envs) setNumEnvs(nextEnv.recommended_num_envs)
    // Same idea for the wrapper graph — e.g. CarRacing wants a specific
    // Frame Skip -> Resize -> Grayscale -> Frame Stack -> Normalize Reward
    // pipeline (see `recommended_wrappers` in rl_core/envs/registry.py);
    // envs without one just get the previous flat "clear on env change".
    setWrappers(nextEnv?.recommended_wrappers ?? [])
  }, [environments])

  const handleAlgoChange = useCallback(
    (id: string) => {
      setAlgorithmId(id)
      const spec = allAlgorithms.find((a) => a.id === id)
      setHyperparams(buildDefaultHyperparams(spec, selectedEnv))
      setNetworkSpecId(null)
      setQuickHiddenLayers(null)
    },
    [allAlgorithms, selectedEnv],
  )

  const addWrapper = useCallback(
    (index: number) => {
      if (wrapperCatalog.length === 0) return
      setWrappers((prev) => {
        const next = [...prev]
        next.splice(index, 0, { type: wrapperCatalog[0].type, params: wrapperCatalog[0].params })
        return next
      })
    },
    [wrapperCatalog],
  )

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

  // The Algorithm node's inline "quick layer editor" only understands a
  // plain MLP trunk, so it's scoped to the same families as a hand-designed
  // `network_spec` (ppo/a2c/dqn/rainbow_dqn) — see `requiredFamilyFor`.
  const selectedAlgorithm = allAlgorithms.find((a) => a.id === algorithmId)
  const quickFamily = selectedAlgorithm ? requiredFamilyFor(selectedAlgorithm.id, selectedAlgorithm.kind) : null
  const quickNetworkSpec = useMemo(
    () => (quickFamily && quickHiddenLayers ? quickSpecForFamily(quickFamily, quickHiddenLayers) : null),
    [quickFamily, quickHiddenLayers],
  )
  // While resuming, the network is whatever the source run/checkpoint
  // actually used (inline or catalog) — never re-derived from the quick
  // editor/`networkSpecId`, both of which are locked in the UI anyway.
  const effectiveNetworkSpecId = resumeFrom ? null : networkSpecId
  const effectiveNetworkSpec = resumeFrom ? resumeNetworkSpec : quickNetworkSpec

  const inspectPayload = useMemo(() => {
    if (!environmentId || !algorithmId) return null
    return {
      kind,
      environment: { id: environmentId, wrappers: kind === 'gym' ? wrappers : undefined },
      algorithm: { id: algorithmId, hyperparams, network_spec_id: effectiveNetworkSpecId, network_spec: effectiveNetworkSpec },
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [environmentId, algorithmId, kind, JSON.stringify(wrappers), JSON.stringify(hyperparams), effectiveNetworkSpecId, effectiveNetworkSpec])
  const { data: inspectData, isFetching: inspectFetching } = useInspectDesign(inspectPayload)

  const canRun = !!environmentId && !!algorithmId && !starting && !resumePending

  const handleRun = useCallback(async () => {
    if (!canRun) return
    setStarting(true)
    try {
      const config = {
        kind,
        name,
        environment: { id: environmentId, wrappers: kind === 'gym' ? wrappers : [] },
        algorithm: { id: algorithmId, hyperparams, network_spec_id: effectiveNetworkSpecId, network_spec: effectiveNetworkSpec },
        training: {
          total_timesteps: totalTimesteps,
          num_envs: numEnvs,
          num_iterations: numIterations,
          seed,
          use_gpu: useGpu,
          resume_from: resumeFrom,
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
  }, [canRun, kind, name, environmentId, wrappers, algorithmId, hyperparams, effectiveNetworkSpecId, effectiveNetworkSpec, totalTimesteps, numEnvs, numIterations, seed, useGpu, resumeFrom, navigate])

  const algoColumnIndex = 2
  const trainingColumnIndex = 3
  const wrapperColumnBase = COL_X[1]
  const { applyPositions, onNodesChange, positions } = useNodePositions()

  const algoX = useMemo(
    () => (kind === 'gym' ? wrapperColumnBase + wrappers.length * WRAPPER_GAP + (wrappers.length ? 40 : 0) : COL_X[algoColumnIndex]),
    [kind, wrapperColumnBase, wrappers.length],
  )

  const rawNodes: Node[] = useMemo(() => {
    const list: Node[] = []

    list.push({
      id: 'env',
      type: 'env',
      position: { x: COL_X[0], y: ROW_Y },
      data: {
        environments,
        selectedId: environmentId,
        observationSpace: inspectData?.environment?.observation_space,
        actionSpace: inspectData?.environment?.action_space,
        onChange: handleEnvChange,
        disabled: !!resumeFrom,
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
          disabled: !!resumeFrom,
        } satisfies WrapperNodeData,
      })
    })

    list.push({
      id: 'algorithm',
      type: 'algorithm',
      position: { x: algoX, y: ROW_Y },
      data: {
        algorithms: compatibleAlgorithms,
        selectedId: algorithmId,
        hyperparams,
        networks,
        networkSpecId,
        quickHiddenLayers,
        locked: !!resumeFrom,
        onChangeAlgo: handleAlgoChange,
        onChangeHyperparam: (key: string, value: number) =>
          setHyperparams((prev) => ({ ...prev, [key]: value })),
        onChangeNetworkSpecId: (id: string | null) => {
          setNetworkSpecId(id)
          // Mutually exclusive with the quick layer editor — picking a
          // saved architecture replaces the whole spec (heads included),
          // which the quick editor can't represent.
          if (id) setQuickHiddenLayers(null)
          // NoisyNet replaces concrete Linear layers in the built-in Q
          // heads, so it cannot be applied to an arbitrary NetworkSpec.
          // Fall back to ε-greedy immediately instead of letting an
          // incompatible configuration reach the backend and fail at run.
          if (id) {
            setHyperparams((prev) => (
              prev.action_exploration === 1 ? { ...prev, action_exploration: 0 } : prev
            ))
          }
        },
        onChangeQuickLayers: (layers: number[] | null) => {
          setQuickHiddenLayers(layers)
          // Same NoisyNet incompatibility as above, once the trunk is
          // actually hand-edited (`layers` becomes non-null).
          if (layers) {
            setHyperparams((prev) => (
              prev.action_exploration === 1 ? { ...prev, action_exploration: 0 } : prev
            ))
          }
        },
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
        recommendedTotalTimesteps: selectedEnv?.recommended_total_timesteps,
        numEnvs,
        recommendedNumEnvs: selectedEnv?.recommended_num_envs,
        numIterations,
        seed,
        useGpu,
        starting,
        disabled: !canRun,
        onChangeName: setName,
        onChangeTotalTimesteps: setTotalTimesteps,
        onChangeNumEnvs: setNumEnvs,
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
    hyperparams, networks, networkSpecId, quickHiddenLayers, kind, name, totalTimesteps, numEnvs, numIterations, seed, useGpu, starting, canRun, algoX, resumeFrom, inspectData,
  ])

  // The "+" insertion buttons must track wherever the env/wrapper/algorithm
  // nodes actually are *right now* — including mid-drag — not just their
  // auto-layout column, otherwise dragging any node leaves the button
  // floating behind at its old spot. Kept as its own (cheap) memo, separate
  // from `rawNodes` above, specifically so it can depend on live drag
  // `positions` without forcing the whole node list — env/wrapper/algorithm/
  // training, each with a bunch of callbacks — to rebuild every drag frame.
  const addNodes: Node[] = useMemo(() => {
    if (kind !== 'gym' || resumeFrom) return []
    const at = (id: string, fallback: { x: number; y: number }) => positions[id] ?? fallback
    const chain = [
      at('env', { x: COL_X[0], y: ROW_Y }),
      ...wrappers.map((_, i) => at(`wrapper-${i}`, { x: wrapperColumnBase + i * WRAPPER_GAP, y: ROW_Y })),
    ]
    const algoPos = at('algorithm', { x: algoX, y: ROW_Y })
    const list: Node[] = []
    for (let k = 0; k <= wrappers.length; k++) {
      const left = chain[k]
      const right = k < wrappers.length ? chain[k + 1] : algoPos
      list.push({
        id: `add-wrapper-${k}`,
        type: 'add',
        position: { x: (left.x + right.x) / 2, y: (left.y + right.y) / 2 + 140 },
        data: { onAdd: () => addWrapper(k) } satisfies AddNodeData,
      })
    }
    return list
  }, [kind, resumeFrom, wrappers, wrapperColumnBase, algoX, addWrapper, positions])

  const nodes = applyPositions([...rawNodes, ...addNodes])

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
        {kind === 'gym' && !resumeFrom && (
          <Button size="sm" variant="outline" onClick={() => addWrapper(wrappers.length)} disabled={wrapperCatalog.length === 0}>
            <Plus className="h-3.5 w-3.5" />
            Добавить wrapper
          </Button>
        )}
        {kind === 'gym' && !resumeFrom && selectedAlgorithm && (
          <Button size="sm" variant="outline" onClick={() => setSweepOpen(true)} disabled={!environmentId || !algorithmId}>
            <FlaskConical className="h-3.5 w-3.5" />
            Запустить как sweep
          </Button>
        )}
        {resumeFrom && (
          <div className="flex items-center gap-2 rounded-lg border border-primary/40 bg-primary/10 px-3 py-1.5 text-xs text-primary">
            <Sparkles className="h-3.5 w-3.5" />
            Дообучение от {resumeFrom.source === 'run' ? 'запуска' : 'модели'} «{resumeFrom.id}»
            <button
              type="button"
              onClick={() => {
                setResumeFrom(null)
                setResumeNetworkSpec(null)
              }}
              className="ml-1 rounded p-0.5 hover:bg-primary/20"
              title="Отменить дообучение (значения останутся, но снова можно редактировать)"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        )}
      </div>
      <InspectPanel
        data={inspectData}
        isFetching={inspectFetching}
        sceneAgentCount={environmentId.startsWith('scene:') ? (selectedEnv?.scene_agent_count ?? null) : null}
      />
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        proOptions={{ hideAttribution: true }}
        colorMode="dark"
        fitView
      >
        <Background />
        <Controls />
      </ReactFlow>
      {selectedAlgorithm && (
        <SweepDialog
          open={sweepOpen}
          onClose={() => setSweepOpen(false)}
          kind={kind}
          environmentId={environmentId}
          wrappers={kind === 'gym' ? wrappers : []}
          algorithm={selectedAlgorithm}
          baseHyperparams={hyperparams}
          training={{ total_timesteps: totalTimesteps, num_envs: numEnvs, use_gpu: useGpu, seed }}
          onCreated={(sweepId) => navigate(`/sweeps?id=${sweepId}`)}
        />
      )}
    </div>
  )
}

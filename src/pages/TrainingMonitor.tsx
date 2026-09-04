import { useCallback, useEffect, useMemo, useRef, useState, memo } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { toast } from 'sonner'
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis,
} from 'recharts'
import { Loader2, Square, Trash2, Save, FileText, Network, GitCompare, Sparkles, PlayCircle, X, Brain } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Dialog } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useAlgorithms, useRuns, useRunArchitecture, useRunNetwork, useRunWorldModel } from '@/api/hooks'
import { api, createMetricsWebSocket } from '@/api/client'
import { AlgorithmDiagram, type AlgorithmDiagramNetwork } from '@/components/AlgorithmDiagram'
import { SaveArchitectureDialog } from '@/components/networkbuilder/SaveArchitectureDialog'
import { CompareRunsPanel } from '@/components/monitor/CompareRunsPanel'
import { RunFolderLink } from '@/components/monitor/RunFolderLink'
import { EvaluateDialog } from '@/components/models/EvaluateDialog'
import { Tooltip as HpTooltip } from '@/components/ui/tooltip'
import type { HyperparamSpec, MetricsSnapshot, RunArchitecture, SpaceInfo } from '@/api/types'
import { cn, formatDuration } from '@/lib/utils'
import { spaceSize } from '@/lib/spaceInfo'
import { WORLD_MODEL_TYPE_LABELS } from '@/lib/worldModels'
import { useSettingsStore } from '@/stores/settingsStore'

const STATUS_VARIANT: Record<string, 'success' | 'secondary' | 'destructive' | 'outline'> = {
  running: 'success',
  completed: 'secondary',
  stopped: 'outline',
  failed: 'destructive',
  interrupted: 'destructive',
}

interface LossField {
  key: keyof MetricsSnapshot
  label: string
  color: string
}

// Which loss-like fields make sense to chart depends on the algorithm — a
// DQN run never has a "policy_loss" and a PPO run never has a "td_loss".
// See rl_core/algorithms/native/{ppo,a2c,dqn,rainbow_dqn,sac}.py and
// rl_core/alphazero/base.py for where each of these gets computed.
const LOSS_FIELDS_BY_ALGO: Record<string, LossField[]> = {
  ppo: [
    { key: 'policy_loss', label: 'Policy loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'value_loss', label: 'Value loss', color: 'oklch(0.72 0.16 145)' },
    { key: 'entropy', label: 'Entropy', color: 'oklch(0.72 0.16 55)' },
  ],
  a2c: [
    { key: 'policy_loss', label: 'Policy loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'value_loss', label: 'Value loss', color: 'oklch(0.72 0.16 145)' },
    { key: 'entropy', label: 'Entropy', color: 'oklch(0.72 0.16 55)' },
  ],
  dqn: [
    { key: 'td_loss', label: 'TD loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'rnd_predictor_loss', label: 'RND predictor loss', color: 'oklch(0.72 0.16 30)' },
  ],
  rainbow_dqn: [
    { key: 'td_loss', label: 'TD loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'rnd_predictor_loss', label: 'RND predictor loss', color: 'oklch(0.72 0.16 30)' },
  ],
  sac: [
    { key: 'actor_loss', label: 'Actor loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'critic_loss', label: 'Critic loss', color: 'oklch(0.72 0.16 145)' },
  ],
  ddpg: [
    { key: 'actor_loss', label: 'Actor loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'critic_loss', label: 'Critic loss', color: 'oklch(0.72 0.16 145)' },
  ],
  td3: [
    { key: 'actor_loss', label: 'Actor loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'critic_loss', label: 'Critic loss', color: 'oklch(0.72 0.16 145)' },
  ],
  // ES never computes a gradient-based loss at all — this "Loss" card
  // instead doubles up as its fitness-progress chart, which is the closest
  // analogue this algorithm has (see rl_core/algorithms/native/es.py).
  es: [
    { key: 'es_mean_fitness', label: 'Mean fitness (популяция)', color: 'oklch(0.7 0.15 260)' },
    { key: 'es_best_fitness', label: 'Best fitness (популяция)', color: 'oklch(0.72 0.16 145)' },
  ],
  alphazero: [
    { key: 'policy_loss', label: 'Policy loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'value_loss', label: 'Value loss', color: 'oklch(0.72 0.16 145)' },
  ],
  // World Models (rl_core/world_models/) — both the three standalone
  // `kind: "world_model"` types (algorithm_id is the type itself, e.g.
  // "rssm") and the four algorithms built on top of them.
  rssm: [
    { key: 'recon_loss', label: 'Reconstruction loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'kl_loss', label: 'KL loss', color: 'oklch(0.72 0.16 145)' },
    { key: 'reward_loss', label: 'Reward loss', color: 'oklch(0.72 0.16 55)' },
    { key: 'continue_loss', label: 'Continue loss', color: 'oklch(0.72 0.16 30)' },
  ],
  dreamer: [
    { key: 'recon_loss', label: 'Reconstruction loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'kl_loss', label: 'KL loss', color: 'oklch(0.72 0.16 145)' },
    { key: 'reward_loss', label: 'Reward loss', color: 'oklch(0.72 0.16 55)' },
    { key: 'continue_loss', label: 'Continue loss', color: 'oklch(0.72 0.16 30)' },
    { key: 'actor_loss', label: 'Actor loss (в воображении)', color: 'oklch(0.65 0.2 300)' },
    { key: 'critic_loss', label: 'Critic loss (в воображении)', color: 'oklch(0.65 0.2 20)' },
  ],
  ensemble: [
    { key: 'dynamics_loss', label: 'Dynamics loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'reward_loss', label: 'Reward loss', color: 'oklch(0.72 0.16 55)' },
  ],
  mbpo: [
    { key: 'dynamics_loss', label: 'Dynamics loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'reward_loss', label: 'Reward loss', color: 'oklch(0.72 0.16 55)' },
    { key: 'actor_loss', label: 'Actor loss (SAC)', color: 'oklch(0.65 0.2 300)' },
    { key: 'critic_loss', label: 'Critic loss (SAC)', color: 'oklch(0.65 0.2 20)' },
  ],
  pets: [
    { key: 'dynamics_loss', label: 'Dynamics loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'reward_loss', label: 'Reward loss', color: 'oklch(0.72 0.16 55)' },
  ],
  vae_mdnrnn: [
    { key: 'vae_recon_loss', label: 'VAE reconstruction loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'vae_kl_loss', label: 'VAE KL loss', color: 'oklch(0.72 0.16 145)' },
    { key: 'mdn_loss', label: 'MDN-RNN loss', color: 'oklch(0.72 0.16 55)' },
    { key: 'reward_loss', label: 'Reward loss', color: 'oklch(0.72 0.16 30)' },
    { key: 'continue_loss', label: 'Continue loss', color: 'oklch(0.65 0.2 20)' },
  ],
  world_models_ha: [
    { key: 'vae_recon_loss', label: 'VAE reconstruction loss', color: 'oklch(0.7 0.15 260)' },
    { key: 'vae_kl_loss', label: 'VAE KL loss', color: 'oklch(0.72 0.16 145)' },
    { key: 'mdn_loss', label: 'MDN-RNN loss', color: 'oklch(0.72 0.16 55)' },
    { key: 'es_mean_fitness', label: 'ES: mean fitness (контроллер)', color: 'oklch(0.65 0.2 300)' },
    { key: 'es_best_fitness', label: 'ES: best fitness (контроллер)', color: 'oklch(0.65 0.2 20)' },
  ],
}

// Fallback for custom/plugin algorithms the app doesn't recognize by id —
// try every loss-like field we know about and only chart the ones that
// actually show up at least once (see the `.filter` below where this is used).
const ALL_LOSS_FIELDS: LossField[] = [
  { key: 'policy_loss', label: 'Policy loss', color: 'oklch(0.7 0.15 260)' },
  { key: 'value_loss', label: 'Value loss', color: 'oklch(0.72 0.16 145)' },
  { key: 'entropy', label: 'Entropy', color: 'oklch(0.72 0.16 55)' },
  { key: 'td_loss', label: 'TD loss', color: 'oklch(0.7 0.15 260)' },
  { key: 'actor_loss', label: 'Actor loss', color: 'oklch(0.7 0.15 260)' },
  { key: 'critic_loss', label: 'Critic loss', color: 'oklch(0.72 0.16 145)' },
  { key: 'rnd_predictor_loss', label: 'RND predictor loss', color: 'oklch(0.72 0.16 30)' },
  { key: 'loss', label: 'Loss', color: 'oklch(0.7 0.15 260)' },
  { key: 'entropy_loss', label: 'Entropy loss', color: 'oklch(0.72 0.16 55)' },
  { key: 'recon_loss', label: 'Reconstruction loss', color: 'oklch(0.7 0.15 260)' },
  { key: 'kl_loss', label: 'KL loss', color: 'oklch(0.72 0.16 145)' },
  { key: 'continue_loss', label: 'Continue loss', color: 'oklch(0.72 0.16 30)' },
  { key: 'dynamics_loss', label: 'Dynamics loss', color: 'oklch(0.7 0.15 260)' },
  { key: 'vae_recon_loss', label: 'VAE reconstruction loss', color: 'oklch(0.7 0.15 260)' },
  { key: 'vae_kl_loss', label: 'VAE KL loss', color: 'oklch(0.72 0.16 145)' },
  { key: 'mdn_loss', label: 'MDN-RNN loss', color: 'oklch(0.72 0.16 55)' },
]

export function TrainingMonitor() {
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const selectedRunId = params.get('run') ?? undefined
  const { data: runsData, refetch } = useRuns()
  const runs = runsData?.runs ?? []

  const [history, setHistory] = useState<MetricsSnapshot[]>([])
  const [promoteOpen, setPromoteOpen] = useState(false)
  const [promoteName, setPromoteName] = useState('')
  const [saveArchOpen, setSaveArchOpen] = useState(false)
  const [evaluateOpen, setEvaluateOpen] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  // "Сравнение" mode: pick 2+ runs from the sidebar list and overlay their
  // reward curves instead of drilling into any one of them — see
  // CompareRunsPanel. Independent of `selectedRunId`/the websocket
  // subscription below, which keep working normally underneath.
  const [compareMode, setCompareMode] = useState(false)
  const [compareIds, setCompareIds] = useState<Set<string>>(new Set())
  const [showCompare, setShowCompare] = useState(false)
  const toggleCompareId = useCallback((runId: string) => {
    setCompareIds((prev) => {
      const next = new Set(prev)
      if (next.has(runId)) next.delete(runId)
      else next.add(runId)
      return next
    })
  }, [])
  const compareRuns = useMemo(() => runs.filter((r) => compareIds.has(r.run_id)), [runs, compareIds])

  const selectedRun = runs.find((r) => r.run_id === selectedRunId)
  const { data: runNetwork } = useRunNetwork(selectedRunId)
  const { data: runArchitecture } = useRunArchitecture(selectedRunId)
  const canSaveArchitecture = !!runNetwork?.family && !!runNetwork?.spec
  const { data: runWorldModel } = useRunWorldModel(selectedRunId)
  const { data: algosData } = useAlgorithms()
  const language = useSettingsStore((s) => s.language)
  // key -> HyperparamSpec for the currently selected run's algorithm, so
  // the "Гиперпараметры алгоритма" card can show a proper label + a
  // hover-tooltip description instead of the raw hyperparam key.
  const hyperparamSpecs = useMemo(() => {
    const algo = algosData?.algorithms.find((a) => a.id === selectedRun?.algorithm_id)
    const map = new Map<string, HyperparamSpec>()
    for (const hp of algo?.hyperparams ?? []) map.set(hp.key, hp)
    return map
  }, [algosData, selectedRun?.algorithm_id])

  useEffect(() => {
    if (!selectedRunId) return
    setHistory([])
    let cancelled = false

    // `metrics.json` only ever holds the *latest* snapshot — the full
    // reward/loss curve for anything before the current websocket
    // connection lives in `metrics_history.jsonl` (see
    // rl_core/metrics_history.py). Fetch that first so the chart doesn't
    // start empty and rebuild from scratch on every visit, but — unlike
    // before — connect the websocket unconditionally, not chained after
    // this REST call succeeding. A run with no history file yet, or a
    // transient network hiccup on this one request, must never leave the
    // chart stuck empty forever just because the initial fetch failed;
    // live snapshots still need to be able to arrive and start drawing.
    api
      .getMetricsHistory(selectedRunId)
      .then((res) => {
        if (!cancelled) setHistory(res.history)
      })
      .catch(() => {})

    const mergeSnapshot = (snapshot: MetricsSnapshot) => {
      if (cancelled) return
      setHistory((prev) => {
        const last = prev[prev.length - 1]
        // The very first live message after connecting is almost always
        // the same step already present at the end of the just-loaded
        // history — replace instead of duplicating it.
        if (last && last.step === snapshot.step) return [...prev.slice(0, -1), snapshot]
        return [...prev.slice(-1999), snapshot]
      })
    }

    let ws: WebSocket | null = null
    createMetricsWebSocket(selectedRunId, mergeSnapshot).then((socket) => {
      if (cancelled) {
        socket.close()
        return
      }
      ws = socket
      wsRef.current = socket
    })

    return () => {
      cancelled = true
      ws?.close()
      wsRef.current = null
    }
  }, [selectedRunId])

  const selectRun = useCallback(
    (runId: string) => setParams({ run: runId }),
    [setParams],
  )

  const handleStop = useCallback(async () => {
    if (!selectedRunId) return
    await api.stopRun(selectedRunId)
    toast.success('Остановка запрошена')
    refetch()
  }, [selectedRunId, refetch])

  const handleDelete = useCallback(async () => {
    if (!selectedRunId) return
    try {
      await api.deleteRun(selectedRunId)
      setParams({})
      refetch()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось удалить запуск')
    }
  }, [selectedRunId, refetch, setParams])

  const handlePromote = useCallback(async () => {
    if (!selectedRunId || !promoteName.trim()) return
    try {
      await api.promoteRun(selectedRunId, promoteName.trim())
      toast.success('Модель сохранена в Model Zoo')
      setPromoteOpen(false)
      setPromoteName('')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось сохранить модель')
    }
  }, [selectedRunId, promoteName])

  // Defensive against duplicate/out-of-order `step` values sneaking into
  // `history` (e.g. a run finishing writes one final snapshot at the same
  // step as its last periodic one) — sorted ascending and deduped so the
  // charts below always draw left-to-right without ever doubling back on
  // themselves. Kept separate from `history` itself since other things
  // here (GIF/board carry-forward) want the raw arrival order.
  const chartData = useMemo(() => {
    const byStep = new Map<number, MetricsSnapshot>()
    for (const snap of history) byStep.set(snap.step, snap)
    return [...byStep.values()].sort((a, b) => a.step - b.step)
  }, [history])

  const latest = chartData[chartData.length - 1] ?? (selectedRun?.metrics as MetricsSnapshot | undefined)
  const isAlphaZero = latest?.kind === 'alphazero' || selectedRun?.kind === 'alphazero'
  // A standalone World Model run (rl_core/world_models/trainer.py) —
  // "collects experience, fits a model", never an agent at all, so
  // evaluate/resume (both meant for a real policy) don't apply the way
  // they do for `dreamer`/`mbpo`/`pets`/`world_models_ha` (plain `kind: "gym"`
  // runs, unaffected here).
  const isWorldModel = latest?.kind === 'world_model' || selectedRun?.kind === 'world_model'

  // The backend only records a fresh episode/game every couple thousand
  // steps (rendering + encoding is expensive) — most metrics snapshots
  // simply omit `episode_gif_base64`/`board_history`. Reading those straight
  // off `latest` made the "Живой просмотр среды" card mount/unmount on every
  // snapshot that lacked them, i.e. exactly the "то исчезает то появляется"
  // flicker. Instead, carry the last non-empty one forward until a newer
  // one shows up — the GIF/board-replay component below then plays that
  // whole recorded episode/game on a loop until it's replaced.
  const { lastGif, lastGifFile, lastGifStep, lastBoardHistory, lastBoardWinner } = useMemo(() => {
    let gif = selectedRun?.metrics?.episode_gif_base64
    let gifFile = selectedRun?.metrics?.episode_gif_file
    let gifStep = selectedRun?.metrics?.episode_gif_step
    let boardHistory = selectedRun?.metrics?.board_history
    let boardWinner = selectedRun?.metrics?.board_winner
    for (const snap of history) {
      if (snap.episode_gif_base64) gif = snap.episode_gif_base64
      if (snap.episode_gif_file) gifFile = snap.episode_gif_file
      if (snap.episode_gif_step != null) gifStep = snap.episode_gif_step
      if (snap.board_history) boardHistory = snap.board_history
      if (snap.board_winner != null) boardWinner = snap.board_winner
    }
    return { lastGif: gif, lastGifFile: gifFile, lastGifStep: gifStep, lastBoardHistory: boardHistory, lastBoardWinner: boardWinner }
  }, [history, selectedRun])

  // Most periodic snapshots deliberately omit `episode_gif_base64` (see
  // `runner_utils.py`/`metrics_callback.py` — only the exact write right
  // after a fresh render carries the full, up-to-a-few-MB blob) — this
  // resolves the backend URL to fall back to whenever that's the case but
  // a preview file still exists (`episode_gif_file`). `lastGifStep` is a
  // cache-busting key: it only changes when the file on disk actually does,
  // so this doesn't refetch on every metrics tick, just every new episode.
  const [lastGifUrl, setLastGifUrl] = useState<string | null>(null)
  useEffect(() => {
    if (lastGif || !lastGifFile || !selectedRunId) {
      setLastGifUrl(null)
      return
    }
    let cancelled = false
    api.resolveUrl(`/training/runs/${selectedRunId}/preview.gif?t=${lastGifStep ?? 0}`).then((url) => {
      if (!cancelled) setLastGifUrl(url)
    })
    return () => { cancelled = true }
  }, [lastGif, lastGifFile, lastGifStep, selectedRunId])

  // Latent-space scatter (rl_core/world_models/viz.py::render_latent_scatter)
  // — only produced for the two World Model types with an actual single
  // latent vector (RSSM/VAE+MDN-RNN; the Ensemble type has no shared latent
  // to plot). Written alongside `episode_preview.gif` on the same cadence
  // (`_PREVIEW_EVERY_STEPS` in trainer.py), so `lastGifStep` doubles as its
  // cache-busting key too — no separate metric field needed.
  const hasLatentSpace = runWorldModel?.type === 'rssm' || runWorldModel?.type === 'vae_mdnrnn'
  const [latentUrl, setLatentUrl] = useState<string | null>(null)
  const [latentMissing, setLatentMissing] = useState(false)
  useEffect(() => {
    if (!hasLatentSpace || !selectedRunId) {
      setLatentUrl(null)
      return
    }
    let cancelled = false
    setLatentMissing(false)
    api.resolveUrl(`/training/runs/${selectedRunId}/latent_space.png?t=${lastGifStep ?? 0}`).then((url) => {
      if (!cancelled) setLatentUrl(url)
    })
    return () => { cancelled = true }
  }, [hasLatentSpace, selectedRunId, lastGifStep])

  // Only chart a loss field if it actually shows up somewhere in this run's
  // history — e.g. `rnd_predictor_loss` only exists when RND intrinsic
  // exploration is enabled, and a NoisyNet-only DQN run never reports it.
  const lossFields = useMemo(() => {
    const candidates = LOSS_FIELDS_BY_ALGO[selectedRun?.algorithm_id ?? ''] ?? ALL_LOSS_FIELDS
    return candidates.filter((field) => chartData.some((snap) => snap[field.key] != null))
  }, [selectedRun?.algorithm_id, chartData])

  // Most runs never enable RND, so `episode_extrinsic_reward_mean`/
  // `episode_intrinsic_reward_mean` are `null` in *every* snapshot. Only
  // render those two extra lines once they actually have something to show.
  const hasExtrinsicSplit = chartData.some(
    (snap) => snap.episode_extrinsic_reward_mean != null || snap.episode_intrinsic_reward_mean != null,
  )

  // This used to be `{isAlphaZero ? <Line .../> : (<><Line/><Line/><Line/></>)}`
  // written straight in the JSX below. Turns out Recharts' <LineChart> reads
  // its series off `props.children` and does *not* look inside a Fragment
  // that a ternary branch evaluates to — so every `<Line>` nested in that
  // `<>...</>` (including the main "Суммарная" reward line, which has
  // nothing to do with the RND split) silently rendered *nothing*: 0 series,
  // 0 paths, empty chart. Confirmed with a repro against Recharts 2.15.4
  // rendering the exact same JSX shape. Building a flat array of elements
  // ahead of time and passing that in as a single child (`{rewardLines}`)
  // is a plain array, which `<LineChart>` traverses just fine — same
  // pattern the Loss chart below already used via `.map()`, which is why
  // that one never had this problem.
  // `type="linear"` (plain straight segments) rather than `"monotone"`
  // (cubic spline) — monotone interpolation overshoots between points that
  // zig-zag up/down, which is normal for noisy per-step metrics, and that
  // overshoot is exactly the "wriggling snake" look. Linear segments track
  // the actual points with no artificial curvature.
  const rewardLines = isAlphaZero
    ? [
        <Line key="win_rate" type="linear" dataKey="win_rate_vs_prev" stroke="oklch(0.7 0.15 260)" dot={false} strokeWidth={2} />,
      ]
    : [
        <Line key="episode_reward_mean" name="Суммарная" type="linear" dataKey="episode_reward_mean" stroke="oklch(0.7 0.15 260)" dot={false} strokeWidth={2} connectNulls />,
        ...(hasExtrinsicSplit
          ? [
              <Line key="episode_extrinsic_reward_mean" name="Extrinsic" type="linear" dataKey="episode_extrinsic_reward_mean" stroke="oklch(0.72 0.16 145)" dot={false} strokeWidth={1.5} connectNulls />,
              <Line key="episode_intrinsic_reward_mean" name="Intrinsic" type="linear" dataKey="episode_intrinsic_reward_mean" stroke="oklch(0.72 0.16 55)" dot={false} strokeWidth={1.5} connectNulls />,
            ]
          : []),
      ]

  return (
    <div className="flex h-full">
      <div className="flex w-72 shrink-0 flex-col border-r border-border">
        <div className="flex items-center justify-between gap-2 border-b border-border p-3">
          <span className="text-xs font-medium text-muted-foreground">Запуски</span>
          <Button
            variant={compareMode ? 'secondary' : 'ghost'}
            size="sm"
            className="h-7 text-xs"
            onClick={() => {
              setCompareMode((v) => !v)
              if (compareMode) {
                setCompareIds(new Set())
                setShowCompare(false)
              }
            }}
          >
            <GitCompare className="h-3.5 w-3.5" /> Сравнить
          </Button>
        </div>
        <ScrollArea className="h-full flex-1 p-3">
          <div className="space-y-2">
            {runs.length === 0 && (
              <p className="p-4 text-center text-sm text-muted-foreground">Пока нет запусков</p>
            )}
            {runs.map((run) => (
              <button
                key={run.run_id}
                onClick={() => (compareMode ? toggleCompareId(run.run_id) : selectRun(run.run_id))}
                className={cn(
                  'w-full rounded-lg border border-border p-3 text-left transition-colors hover:border-primary/50',
                  !compareMode && selectedRunId === run.run_id && 'border-primary bg-primary/5',
                  compareMode && compareIds.has(run.run_id) && 'border-primary bg-primary/5',
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-2">
                    {compareMode && (
                      <input
                        type="checkbox"
                        checked={compareIds.has(run.run_id)}
                        onChange={() => toggleCompareId(run.run_id)}
                        onClick={(e) => e.stopPropagation()}
                        className="accent-primary"
                      />
                    )}
                    <span className="truncate text-sm font-medium">{run.name}</span>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <RunFolderLink runId={run.run_id} runDir={run.run_dir} compact />
                    <Badge variant={STATUS_VARIANT[run.status] ?? 'outline'} className="text-[10px]">
                      {run.status}
                    </Badge>
                  </div>
                </div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {run.environment_id} · {run.algorithm_id}
                </div>
              </button>
            ))}
          </div>
        </ScrollArea>
        {compareMode && compareIds.size >= 2 && (
          <div className="border-t border-border p-3">
            <Button className="w-full" size="sm" onClick={() => setShowCompare(true)}>
              <GitCompare className="h-3.5 w-3.5" /> Сравнить ({compareIds.size})
            </Button>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-auto p-6">
        {showCompare ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Сравнение запусков</h2>
              <Button variant="outline" size="sm" onClick={() => setShowCompare(false)}>
                <X className="h-3.5 w-3.5" /> Закрыть
              </Button>
            </div>
            <CompareRunsPanel runs={compareRuns} />
          </div>
        ) : !selectedRun ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            Выбери запуск слева, чтобы увидеть метрики
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="min-w-0">
                <h2 className="text-lg font-semibold">{selectedRun.name}</h2>
                <p className="text-sm text-muted-foreground">
                  {selectedRun.environment_id} · {selectedRun.algorithm_id}
                </p>
                {selectedRun && (
                  <RunFolderLink runId={selectedRun.run_id} runDir={selectedRun.run_dir} className="mt-1" />
                )}
                {latest?.resumed_from && (
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    Дообучение от {latest.resumed_from.source === 'run' ? 'запуска' : 'модели'}{' '}
                    <span className="font-mono">{latest.resumed_from.id}</span>
                  </p>
                )}
                {selectedRun.sweep && (
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    Sweep {selectedRun.sweep.name ?? selectedRun.sweep.sweep_id} · #{selectedRun.sweep.index + 1}/{selectedRun.sweep.total}
                  </p>
                )}
              </div>
              <div className="flex flex-wrap justify-end gap-2">
                {selectedRun.running && (
                  <Button variant="outline" size="sm" onClick={handleStop}>
                    <Square className="h-3.5 w-3.5" /> Остановить
                  </Button>
                )}
                {selectedRun.has_model && !isAlphaZero && !isWorldModel && (
                  <Button variant="outline" size="sm" onClick={() => setEvaluateOpen(true)}>
                    <PlayCircle className="h-3.5 w-3.5" /> Оценить
                  </Button>
                )}
                {selectedRun.has_model && selectedRun.kind === 'gym' && (
                  <Button variant="outline" size="sm" onClick={() => navigate(`/designer?resumeRun=${selectedRun.run_id}`)}>
                    <Sparkles className="h-3.5 w-3.5" /> Дообучить
                  </Button>
                )}
                {selectedRun.has_model && !isWorldModel && (
                  <Button variant="outline" size="sm" onClick={() => setPromoteOpen(true)}>
                    <Save className="h-3.5 w-3.5" /> Сохранить в Model Zoo
                  </Button>
                )}
                {!selectedRun.running && (
                  <Button variant="destructive" size="sm" onClick={handleDelete}>
                    <Trash2 className="h-3.5 w-3.5" /> Удалить
                  </Button>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <StatCard label="Шаг / итерация" value={`${latest?.step ?? 0} / ${latest?.total_timesteps ?? '—'}`} />
              <StatCard label="Параллельных сред" value={String(latest?.num_envs ?? 1)} />
              <StatCard label="Траекторий завершено" value={latest?.episodes_completed != null ? String(latest.episodes_completed) : '—'} />
              {isAlphaZero ? (
                <>
                  <StatCard label="Win-rate vs prev" value={latest?.win_rate_vs_prev != null ? `${(latest.win_rate_vs_prev * 100).toFixed(0)}%` : '—'} />
                  <StatCard label="Loss" value={latest?.loss?.toFixed(3) ?? '—'} />
                  <StatCard label="Buffer" value={String(latest?.buffer_size ?? '—')} />
                </>
              ) : isWorldModel ? (
                <>
                  <StatCard label="Награда сбора данных" value={latest?.collect_mean_reward?.toFixed(1) ?? '—'} />
                  <StatCard label="FPS" value={String(latest?.fps ?? '—')} />
                </>
              ) : (
                <>
                  <StatCard label="Средняя награда" value={latest?.episode_reward_mean?.toFixed(1) ?? '—'} />
                  <StatCard label="Длина эпизода" value={latest?.episode_length_mean?.toFixed(0) ?? '—'} />
                  <StatCard label="FPS" value={String(latest?.fps ?? '—')} />
                </>
              )}
              <StatCard label="Время" value={latest?.elapsed_seconds ? formatDuration(latest.elapsed_seconds) : '—'} />
            </div>

            {!isAlphaZero && (
              latest?.exploration_epsilon != null
              || latest?.rnd_bonus_mean != null
              || latest?.rnd_predictor_loss != null
            ) && (
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                {latest?.exploration_epsilon != null && Number(latest.hyperparams?.action_exploration ?? 0) === 0 && (
                  <StatCard label="Текущий ε" value={latest.exploration_epsilon.toFixed(3)} />
                )}
                {latest?.episode_extrinsic_reward_mean != null && (
                  <StatCard label="Extrinsic reward" value={latest.episode_extrinsic_reward_mean.toFixed(2)} />
                )}
                {latest?.episode_intrinsic_reward_mean != null && (
                  <StatCard label="Intrinsic reward" value={latest.episode_intrinsic_reward_mean.toFixed(2)} />
                )}
                {latest?.rnd_predictor_loss != null && latest.rnd_predictor_loss > 0 && (
                  <StatCard label="RND predictor loss" value={latest.rnd_predictor_loss.toFixed(4)} />
                )}
              </div>
            )}

            {((!isAlphaZero && (lastGif || lastGifUrl)) || (isAlphaZero && lastBoardHistory && lastBoardHistory.length > 0)) && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Живой просмотр среды</CardTitle>
                  <p className="text-xs text-muted-foreground">
                    {isAlphaZero
                      ? 'Самая свежая self-play партия: проигрывается один раз и остаётся на финальной позиции до следующей итерации.'
                      : isWorldModel
                        ? 'Слева — настоящая траектория из среды, справа — та же самая последовательность действий, "воображённая" (предсказанная) моделью мира.'
                        : 'Самый свежий записанный эпизод: проигрывается один раз и остаётся на финальном кадре до следующей записи.'}
                    {!isAlphaZero && !isWorldModel && (
                      <> Также сохраняется в <span className="font-mono">episode_preview.gif</span> и{' '}
                      <span className="font-mono">previews/</span> в папке запуска.</>
                    )}
                  </p>
                </CardHeader>
                <CardContent className="flex flex-col items-center gap-2">
                  {!isAlphaZero && (lastGif || lastGifUrl) && (
                    <img
                      key={lastGif ? lastGif.slice(0, 32) : lastGifUrl}
                      src={lastGif ? `data:image/gif;base64,${lastGif}` : lastGifUrl!}
                      alt="env episode replay"
                      className="max-h-64 rounded-lg border border-border"
                    />
                  )}
                  {isAlphaZero && lastBoardHistory && lastBoardHistory.length > 0 && (
                    <BoardReplay history={lastBoardHistory} winner={lastBoardWinner} />
                  )}
                </CardContent>
              </Card>
            )}

            {runWorldModel?.type && (
              <Card>
                <CardHeader className="flex-row items-center justify-between space-y-0">
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <Brain className="h-4 w-4 text-primary" /> World Model: {WORLD_MODEL_TYPE_LABELS[runWorldModel.type]}
                  </CardTitle>
                  {runWorldModel.world_model_id && (
                    <Button variant="outline" size="sm" onClick={() => navigate('/world-models')}>
                      <Network className="h-3.5 w-3.5" /> {runWorldModel.world_model_id}
                    </Button>
                  )}
                </CardHeader>
                <CardContent className="space-y-3">
                  {runWorldModel.config && Object.keys(runWorldModel.config).length > 0 && (
                    <KeyValueGrid entries={Object.entries(runWorldModel.config)} />
                  )}
                  {hasLatentSpace && (
                    <div className="space-y-1">
                      <p className="text-xs text-muted-foreground">
                        Латентное пространство модели (PCA-проекция, цвет = шаг во времени внутри эпизода):
                      </p>
                      {latentUrl && !latentMissing ? (
                        <img
                          key={latentUrl}
                          src={latentUrl}
                          alt="latent space scatter"
                          className="max-h-64 rounded-lg border border-border"
                          onError={() => setLatentMissing(true)}
                        />
                      ) : (
                        <p className="text-xs text-muted-foreground">Пока нет ни одного снимка латентного пространства.</p>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {!isWorldModel && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Схема алгоритма</CardTitle>
                </CardHeader>
                <CardContent>
                  <AlgorithmDiagram
                    algorithmId={selectedRun.algorithm_id}
                    kind={isAlphaZero ? 'alphazero' : 'gym'}
                    hyperparams={latest?.hyperparams}
                    network={buildDiagramNetwork(latest, runArchitecture)}
                  />
                </CardContent>
              </Card>
            )}

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {latest?.hyperparams && Object.keys(latest.hyperparams).length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Гиперпараметры алгоритма</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <HyperparamGrid
                      entries={Object.entries(latest.hyperparams)}
                      specs={hyperparamSpecs}
                      language={language}
                    />
                  </CardContent>
                </Card>
              )}

              {(latest?.total_params != null || latest?.network) && (
                <Card>
                  <CardHeader className="flex-row items-center justify-between space-y-0">
                    <CardTitle className="text-sm">Конфигурация модели</CardTitle>
                    {canSaveArchitecture && (
                      <Button variant="outline" size="sm" onClick={() => setSaveArchOpen(true)}>
                        <Network className="h-3.5 w-3.5" /> Сохранить архитектуру
                      </Button>
                    )}
                  </CardHeader>
                  <CardContent>
                    <KeyValueGrid entries={buildNetworkEntries(latest)} />
                  </CardContent>
                </Card>
              )}

              {!isAlphaZero && latest?.wrappers && latest.wrappers.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Wrappers среды</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-1.5">
                      {latest.wrappers.map((w, i) => (
                        <Badge key={i} variant="secondary" className="text-[10px]">{w.type}</Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">
                  {isAlphaZero ? 'Win-rate по итерациям' : 'Награда по шагам (реальная и intrinsic отдельно)'}
                </CardTitle>
              </CardHeader>
              <CardContent className="h-64">
                {chartData.length < 2 ? (
                  <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Ждём метрики...
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.3 0 0)" />
                      {/* `type="number"` (rather than the default "category") positions
                          each point by its actual step *value* instead of its position in
                          the array — a duplicate/out-of-order entry can then never make the
                          line double back on itself, it can only overlap or skip a point. */}
                      <XAxis dataKey="step" type="number" domain={['dataMin', 'dataMax']} stroke="oklch(0.65 0 0)" fontSize={11} />
                      <YAxis stroke="oklch(0.65 0 0)" fontSize={11} />
                      <RTooltip
                        contentStyle={{ background: 'oklch(0.17 0 0)', border: '1px solid oklch(0.3 0 0)', fontSize: 12 }}
                      />
                      {rewardLines}
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>

            {lossFields.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Loss</CardTitle>
                  <p className="text-xs text-muted-foreground">
                    Метрики обучения конкретного алгоритма — какие именно, зависит от {selectedRun.algorithm_id}.
                  </p>
                </CardHeader>
                <CardContent className="h-64">
                  {chartData.length < 2 ? (
                    <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Ждём метрики...
                    </div>
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.3 0 0)" />
                        <XAxis dataKey="step" type="number" domain={['dataMin', 'dataMax']} stroke="oklch(0.65 0 0)" fontSize={11} />
                        <YAxis stroke="oklch(0.65 0 0)" fontSize={11} />
                        <RTooltip
                          contentStyle={{ background: 'oklch(0.17 0 0)', border: '1px solid oklch(0.3 0 0)', fontSize: 12 }}
                        />
                        {lossFields.map((field) => (
                          <Line
                            key={field.key}
                            name={field.label}
                            type="linear"
                            dataKey={field.key}
                            stroke={field.color}
                            dot={false}
                            strokeWidth={1.5}
                            connectNulls
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </CardContent>
              </Card>
            )}

            {latest?.error && (
              <Card className="border-destructive/50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-sm text-destructive">
                    <FileText className="h-4 w-4" /> Ошибка
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-xs text-muted-foreground">{latest.error}</CardContent>
              </Card>
            )}
          </div>
        )}
      </div>

      <Dialog open={promoteOpen} onClose={() => setPromoteOpen(false)} title="Сохранить модель">
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            Модель попадёт в Model Zoo под этим именем и переживёт удаление запуска.
          </p>
          <Input
            value={promoteName}
            onChange={(e) => setPromoteName(e.target.value)}
            placeholder="Название модели"
            autoFocus
          />
          <Button className="w-full" onClick={handlePromote} disabled={!promoteName.trim()}>
            Сохранить
          </Button>
        </div>
      </Dialog>

      {canSaveArchitecture && runNetwork && (
        <SaveArchitectureDialog
          open={saveArchOpen}
          onClose={() => setSaveArchOpen(false)}
          family={runNetwork.family!}
          spec={runNetwork.spec!}
          defaultName={selectedRun ? `${selectedRun.name} — сеть` : undefined}
        />
      )}

      {selectedRun && (
        <EvaluateDialog
          open={evaluateOpen}
          onClose={() => setEvaluateOpen(false)}
          source="run"
          id={selectedRun.run_id}
          label={selectedRun.name}
        />
      )}
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="p-3">
        <div className="text-lg font-semibold">{value}</div>
        <div className="text-xs text-muted-foreground">{label}</div>
      </CardContent>
    </Card>
  )
}

function formatValue(value: unknown): string {
  if (value == null) return '—'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
  if (typeof value === 'boolean') return value ? 'да' : 'нет'
  return String(value)
}

function formatSpace(space?: SpaceInfo): string {
  if (!space) return '—'
  // `Discrete` spaces have an empty `shape` ([]) and carry their size in
  // `n` instead — `space.shape ? ... : ...` treated `[]` as truthy (empty
  // arrays are truthy in JS) and never fell through to the `n=` branch,
  // showing a blank "Discrete []" instead of "Discrete n=2".
  const shape = space.shape && space.shape.length > 0
    ? `[${space.shape.join(', ')}]`
    : space.n != null ? `n=${space.n}` : ''
  return `${space.type} ${shape}`.trim()
}

function spaceToShape(space?: SpaceInfo): number[] | undefined {
  if (!space) return undefined
  if (space.shape && space.shape.length > 0) return space.shape
  if (space.n != null) return [space.n]
  return undefined
}

/** Adapts whatever a live run's `MetricsSnapshot` happens to carry (AlphaZero's
 * `network{}` dims vs. gym runs' flat `layers`/spaces) into the one shape
 * `AlgorithmDiagram` understands — mirrors the InspectPanel's Designer-side
 * adapter for the exact same `InspectNetwork` data, just sourced from a live
 * run instead of a pre-run inspect call. */
function buildDiagramNetwork(
  latest: MetricsSnapshot | undefined,
  architecture?: RunArchitecture,
): AlgorithmDiagramNetwork | null {
  if (!latest) return null
  if (
    latest.total_params == null
    && !latest.network
    && !latest.layers?.length
    && !latest.architecture_components?.length
    && !architecture?.components.length
  ) return null
  return {
    policy: latest.policy,
    layers: latest.layers,
    components: architecture?.components ?? latest.architecture_components,
    totalParams: architecture?.total_params ?? latest.total_params,
    trainableParams: architecture?.trainable_params ?? latest.trainable_params,
    inputShape: spaceToShape(latest.observation_space),
    outputShape: spaceToShape(latest.action_space),
    channels: latest.network?.channels,
    numBlocks: latest.network?.num_blocks,
    actionSize: latest.network?.action_size,
    inputPlanes: latest.network?.input_planes,
    rows: latest.network?.rows,
    cols: latest.network?.cols,
  }
}

function buildNetworkEntries(latest: MetricsSnapshot): [string, string][] {
  const entries: [string, string][] = []
  if (latest.policy) entries.push(['Policy', latest.policy])
  if (latest.network) {
    entries.push(['Каналы / блоки', `${latest.network.channels} / ${latest.network.num_blocks}`])
    entries.push(['Доска', `${latest.network.rows}×${latest.network.cols}`])
    entries.push(['Размер действия', String(latest.network.action_size)])
    entries.push(['Входные плоскости', String(latest.network.input_planes)])
  }
  if (latest.observation_space) {
    entries.push(['Входов (наблюдения)', `${spaceSize(latest.observation_space) ?? '—'} · ${formatSpace(latest.observation_space)}`])
  }
  if (latest.action_space) {
    entries.push(['Выходов (действия)', `${spaceSize(latest.action_space) ?? '—'} · ${formatSpace(latest.action_space)}`])
  }
  if (latest.total_params != null) entries.push(['Параметры сети', latest.total_params.toLocaleString('ru-RU')])
  if (latest.device) entries.push(['Устройство', latest.device.toUpperCase()])
  if (latest.seed != null) entries.push(['Seed', String(latest.seed)])
  return entries
}

function KeyValueGrid({ entries }: { entries: [string, unknown][] }) {
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-3">
      {entries.map(([key, value]) => (
        <div key={key} className="flex flex-col">
          <span className="text-muted-foreground">{key}</span>
          <span className="font-mono font-medium">{formatValue(value)}</span>
        </div>
      ))}
    </div>
  )
}

/** Like `KeyValueGrid`, but specifically for a run's algorithm hyperparams:
 * shows the human-readable `label` from the algorithm's schema (falling
 * back to the raw key for custom-plugin hyperparams, which don't have one)
 * and, whenever a `desc` is available, wraps the label in a hover tooltip
 * explaining what the hyperparameter actually does, in whichever of the
 * two supported UI languages is currently selected. */
function HyperparamGrid({
  entries,
  specs,
  language,
}: {
  entries: [string, unknown][]
  specs: Map<string, HyperparamSpec>
  language: 'ru' | 'en'
}) {
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-3">
      {entries.map(([key, value]) => {
        const spec = specs.get(key)
        const label = spec?.label ?? key
        const desc = spec?.desc?.[language]
        return (
          <div key={key} className="flex flex-col">
            {desc ? (
              <HpTooltip content={desc}>
                <span className="cursor-help text-muted-foreground underline decoration-dotted decoration-muted-foreground/50 underline-offset-2">
                  {label}
                </span>
              </HpTooltip>
            ) : (
              <span className="text-muted-foreground">{label}</span>
            )}
            <span className="font-mono font-medium">{formatValue(value)}</span>
          </div>
        )
      })}
    </div>
  )
}

/** Steps through the latest recorded self-play game once and remains on its
 * final position. A newer game replaces `history` and restarts playback. */
const BoardReplay = memo(function BoardReplay({ history, winner }: { history: number[][][]; winner: number | null | undefined }) {
  const [moveIndex, setMoveIndex] = useState(0)

  useEffect(() => {
    setMoveIndex(0)
    if (history.length <= 1) return
    let cancelled = false
    let timeoutId: ReturnType<typeof setTimeout>
    const scheduleNext = (idx: number) => {
      const isLast = idx === history.length - 1
      if (isLast) return
      timeoutId = setTimeout(() => {
        if (cancelled) return
        const next = idx + 1
        setMoveIndex(next)
        scheduleNext(next)
      }, 550)
    }
    scheduleNext(0)
    return () => {
      cancelled = true
      clearTimeout(timeoutId)
    }
  }, [history])

  const board = history[Math.min(moveIndex, history.length - 1)]
  const isLastMove = moveIndex === history.length - 1

  return (
    <div className="flex flex-col items-center gap-2">
      <BoardPreview board={board} />
      <p className="text-xs text-muted-foreground">
        Ход {moveIndex} / {history.length - 1}
        {isLastMove && winner != null && (
          winner === 0 ? ' · ничья' : ` · победил игрок ${winner === 1 ? '●' : '○'}`
        )}
      </p>
    </div>
  )
})

function BoardPreview({ board }: { board: number[][] }) {
  const cols = board[0]?.length ?? 0
  return (
    <div className="inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
      {board.map((rowVals, row) =>
        rowVals.map((cell, col) => (
          <div
            key={`${row}-${col}`}
            className={cn(
              'flex h-8 w-8 items-center justify-center rounded-md border border-border text-sm font-bold',
              cell === 0 ? 'bg-muted/40' : 'bg-muted',
            )}
          >
            {cell === 1 && <span className="text-primary">●</span>}
            {cell === -1 && <span className="text-warning">○</span>}
          </div>
        )),
      )}
    </div>
  )
}

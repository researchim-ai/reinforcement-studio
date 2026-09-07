import { Fragment, type ReactNode } from 'react'
import {
  ArrowRight, Brain, Crosshair, Database, Dices, Moon, Repeat, RefreshCw, Shuffle, Swords,
  Target, TrendingUp, Eye, Layers, History, Sparkles, Users,
} from 'lucide-react'
import { cn, formatNumber } from '@/lib/utils'
import type { ArchitectureComponent } from '@/api/types'

type HpValue = number | string | boolean
type Hyperparams = Record<string, HpValue> | undefined

function hp(hyperparams: Hyperparams, key: string, prefix = ''): string | undefined {
  const v = hyperparams?.[key]
  return v == null ? undefined : `${prefix}${v}`
}

function joinDetail(...parts: (string | undefined)[]): string | undefined {
  const filtered = parts.filter(Boolean)
  return filtered.length ? filtered.join(' · ') : undefined
}

/** `memory_type` is stored as a plain int (0=none/1=LSTM/2=GRU) — see
 * `rl_core/algorithms/native/networks.py::memory_type_from_hyperparams` —
 * kept in sync here rather than imported since the frontend has no access
 * to the Python module. */
function memoryLabel(hyperparams: Hyperparams): string | null {
  const code = Number(hyperparams?.memory_type ?? 0)
  if (code === 1) return 'LSTM'
  if (code === 2) return 'GRU'
  return null
}

const MEMORY_FAMILIES: Family[] = ['ppo', 'a2c', 'dqn', 'rainbow_dqn']

interface Step {
  icon: typeof Brain
  title: string
  detail?: string
}

type Family =
  | 'ppo' | 'a2c' | 'dqn' | 'rainbow_dqn' | 'sac' | 'ddpg' | 'td3' | 'es' | 'alphazero'
  | 'dreamer' | 'mbpo' | 'pets' | 'world_models_ha' | 'efficientzero' | 'unizero' | 'latentimzero'
  | 'ippo' | 'qmix' | 'generic'

/** Best-effort family detection from the algorithm id — used to pick which
 * canned "how it learns" loop to render. Custom plugins (`custom:*`) fall
 * through to a family based on their base-class-agnostic id text (many
 * people name a PPO tweak "custom:ppo-clip" etc.) or, failing that, a
 * generic agent<->env loop that's honest about not knowing the internals. */
function familyOf(algorithmId: string, kind: 'gym' | 'alphazero'): Family {
  if (kind === 'alphazero') return 'alphazero'
  const id = algorithmId.replace(/^custom:/, '').toLowerCase()
  if (id.includes('rainbow')) return 'rainbow_dqn'
  if (id.includes('td3')) return 'td3'
  if (id.includes('ddpg')) return 'ddpg'
  if (id.includes('sac')) return 'sac'
  if (id.includes('dqn')) return 'dqn'
  if (id.includes('a2c')) return 'a2c'
  if (id.includes('ppo')) return 'ppo'
  if (id === 'es' || id.includes('evolution') || id.includes('cma-es') || id.includes('cmaes')) return 'es'
  if (id.includes('dreamer')) return 'dreamer'
  if (id.includes('mbpo')) return 'mbpo'
  if (id.includes('pets')) return 'pets'
  if (id.includes('world_models_ha') || id.includes('world-models-ha')) return 'world_models_ha'
  if (id.includes('latentimzero')) return 'latentimzero'
  if (id.includes('efficientzero') || id.includes('efficient-zero') || id.includes('muzero')) return 'efficientzero'
  if (id.includes('unizero') || id.includes('researchimzero')) return 'unizero'
  if (id === 'ippo' || id.includes('ippo')) return 'ippo'
  if (id === 'qmix' || id.includes('qmix')) return 'qmix'
  return 'generic'
}

function buildSteps(family: Family, hyperparams: Hyperparams): { steps: Step[]; loopCaption: string } {
  const noisy = Number(hyperparams?.action_exploration ?? 0) === 1
  const rnd = Number(hyperparams?.intrinsic_exploration ?? 0) === 1
  const actionStep: Step = noisy
    ? { icon: Dices, title: 'Действие через NoisyNet', detail: hp(hyperparams, 'noisy_sigma0', 'σ₀=') }
    : {
        icon: Dices,
        title: 'Действие ε-greedy',
        detail: joinDetail(
          hp(hyperparams, 'exploration_initial_eps', 'ε₀='),
          hp(hyperparams, 'exploration_final_eps', 'εmin='),
          hp(hyperparams, 'exploration_fraction', 'decay='),
        ),
      }
  const rndStep: Step = {
    icon: Sparkles,
    title: 'RND: бонус новизны',
    detail: hp(hyperparams, 'rnd_bonus_coef', 'coef='),
  }
  switch (family) {
    case 'ppo':
      return {
        steps: [
          { icon: Shuffle, title: 'Rollout в среде', detail: hp(hyperparams, 'n_steps', 'n_steps=') },
          { icon: TrendingUp, title: 'Advantage (GAE)', detail: hp(hyperparams, 'gamma', 'γ=') },
          {
            icon: Brain,
            title: 'Обновление policy + value',
            detail: joinDetail(hp(hyperparams, 'batch_size', 'batch='), hp(hyperparams, 'ent_coef', 'ent=')),
          },
        ],
        loopCaption: 'Повторяется, пока не наберётся total_timesteps шагов',
      }
    case 'a2c':
      return {
        steps: [
          { icon: Shuffle, title: 'n-step rollout', detail: hp(hyperparams, 'n_steps', 'n_steps=') },
          { icon: TrendingUp, title: 'Advantage', detail: hp(hyperparams, 'gamma', 'γ=') },
          { icon: Brain, title: 'Одно обновление policy + value', detail: hp(hyperparams, 'ent_coef', 'ent=') },
        ],
        loopCaption: 'Повторяется, пока не наберётся total_timesteps шагов',
      }
    case 'dqn':
      return {
        steps: [
          actionStep,
          ...(rnd ? [rndStep] : []),
          { icon: Database, title: 'Replay buffer', detail: hp(hyperparams, 'buffer_size', 'size=') },
          {
            icon: Brain,
            title: 'Обновление Q-сети',
            detail: joinDetail(hp(hyperparams, 'batch_size', 'batch='), hp(hyperparams, 'gamma', 'γ=')),
          },
          { icon: RefreshCw, title: 'Синхронизация target-сети', detail: hp(hyperparams, 'target_update_interval', 'every=') },
        ],
        loopCaption: 'Повторяется на каждом шаге, пока не наберётся total_timesteps',
      }
    case 'rainbow_dqn': {
      const distributional = Number(hyperparams?.distributional ?? 0) === 1
      return {
        steps: [
          actionStep,
          ...(rnd ? [rndStep] : []),
          {
            icon: Database,
            title: 'Приоритетный буфер',
            detail: joinDetail(hp(hyperparams, 'buffer_size', 'size='), hp(hyperparams, 'n_step', 'n-step=')),
          },
          {
            icon: Brain,
            title: distributional ? 'Double + Dueling + QR-DQN' : 'Double + Dueling обновление',
            detail: distributional
              ? joinDetail(hp(hyperparams, 'batch_size', 'batch='), hp(hyperparams, 'num_quantiles', 'квантилей='))
              : joinDetail(hp(hyperparams, 'batch_size', 'batch='), hp(hyperparams, 'gamma', 'γ=')),
          },
          { icon: RefreshCw, title: 'Синхронизация target-сети', detail: hp(hyperparams, 'target_update_interval', 'every=') },
        ],
        loopCaption: distributional
          ? 'Полный Rainbow: Double DQN + Dueling + Prioritized Replay + n-step + Distributional (QR-DQN)'
          : 'Rainbow-lite: Double DQN + Dueling-сеть + Prioritized Replay + n-step возврат',
      }
    }
    case 'sac':
      return {
        steps: [
          { icon: Dices, title: 'Действие из stochastic policy', detail: hp(hyperparams, 'ent_coef', 'α=') },
          { icon: Database, title: 'Replay buffer', detail: hp(hyperparams, 'buffer_size', 'size=') },
          {
            icon: Brain,
            title: 'Обновление 2×Q-критиков + policy',
            detail: joinDetail(hp(hyperparams, 'batch_size', 'batch='), hp(hyperparams, 'gamma', 'γ=')),
          },
          { icon: RefreshCw, title: 'Мягкое обновление target-сетей', detail: hp(hyperparams, 'tau', 'τ=') },
        ],
        loopCaption: 'Off-policy: непрерывные действия, максимизация награды + энтропии',
      }
    case 'ddpg':
      return {
        steps: [
          { icon: Dices, title: 'Действие + гауссов шум', detail: hp(hyperparams, 'exploration_noise', 'σ=') },
          { icon: Database, title: 'Replay buffer', detail: hp(hyperparams, 'buffer_size', 'size=') },
          {
            icon: Brain,
            title: 'Обновление critic + actor (DPG)',
            detail: joinDetail(hp(hyperparams, 'batch_size', 'batch='), hp(hyperparams, 'gamma', 'γ=')),
          },
          { icon: RefreshCw, title: 'Мягкое обновление target-сетей', detail: hp(hyperparams, 'tau', 'τ=') },
        ],
        loopCaption: 'Off-policy: детерминированная политика + один critic, эксплорация через шум действия',
      }
    case 'td3':
      return {
        steps: [
          { icon: Dices, title: 'Действие + гауссов шум', detail: hp(hyperparams, 'exploration_noise', 'σ=') },
          { icon: Database, title: 'Replay buffer', detail: hp(hyperparams, 'buffer_size', 'size=') },
          {
            icon: Brain,
            title: '2×critic (min) + сглаживание target',
            detail: joinDetail(hp(hyperparams, 'batch_size', 'batch='), hp(hyperparams, 'policy_noise', 'smooth σ=')),
          },
          { icon: RefreshCw, title: 'Отложенное обновление actor + target', detail: hp(hyperparams, 'policy_delay', 'delay=') },
        ],
        loopCaption: 'Off-policy: twin critics + delayed policy updates + target policy smoothing',
      }
    case 'es':
      return {
        steps: [
          { icon: Shuffle, title: 'Возмущение популяции', detail: joinDetail(hp(hyperparams, 'population_size', 'N='), hp(hyperparams, 'sigma', 'σ=')) },
          { icon: Eye, title: 'Полный эпизод на кандидата', detail: hp(hyperparams, 'episodes_per_eval', 'эпизодов=') },
          { icon: TrendingUp, title: 'Ранжирование fitness популяции' },
          { icon: Brain, title: 'Adam-шаг по оценке градиента', detail: hp(hyperparams, 'learning_rate', 'lr=') },
        ],
        loopCaption: 'Gradient-free: без backprop через среду — параметры сети обновляются по fitness целых эпизодов',
      }
    case 'dreamer':
      return {
        steps: [
          { icon: Eye, title: 'Реальный шаг', detail: hp(hyperparams, 'collect_steps_per_iter', 'каждые=') },
          { icon: RefreshCw, title: 'Дообучение RSSM', detail: hp(hyperparams, 'seq_len', 'seq_len=') },
          { icon: Moon, title: 'Воображение (prior)', detail: hp(hyperparams, 'imagination_horizon', 'horizon=') },
          { icon: Brain, title: 'Actor-critic на воображении', detail: hp(hyperparams, 'batch_size', 'batch=') },
        ],
        loopCaption: 'Model-based: реальный опыт только дообучает RSSM — actor-critic целиком обучается внутри воображаемых траекторий',
      }
    case 'mbpo':
      return {
        steps: [
          { icon: Eye, title: 'Реальный шаг (SAC)', detail: hp(hyperparams, 'train_freq', 'каждые=') },
          { icon: RefreshCw, title: 'Обучение ансамбля', detail: hp(hyperparams, 'model_train_freq', 'каждые=') },
          { icon: Moon, title: 'Короткие модельные rollouts', detail: hp(hyperparams, 'rollout_length', 'длина=') },
          { icon: Brain, title: 'SAC-обновление', detail: hp(hyperparams, 'real_ratio', 'real_ratio=') },
        ],
        loopCaption: 'Model-based: SAC учится на смеси настоящих и коротких воображаемых переходов, отращенных от реальных состояний',
      }
    case 'pets':
      return {
        steps: [
          { icon: Eye, title: 'Случайный шаг / шаг плана' },
          { icon: RefreshCw, title: 'Обучение ансамбля', detail: hp(hyperparams, 'model_train_freq', 'каждые=') },
          { icon: Crosshair, title: 'CEM-планирование', detail: joinDetail(hp(hyperparams, 'cem_horizon', 'horizon='), hp(hyperparams, 'cem_candidates', 'N=')) },
          { icon: Target, title: 'Первое действие плана' },
        ],
        loopCaption: 'Без обучаемой политики — каждое действие ищется заново CEM-поиском по текущему ансамблю, план целиком выбрасывается после первого шага',
      }
    case 'world_models_ha':
      return {
        steps: [
          { icon: Dices, title: 'Случайный сбор (фаза 1)', detail: hp(hyperparams, 'world_model_phase_steps', 'шагов=') },
          { icon: RefreshCw, title: 'VAE + MDN-RNN', detail: hp(hyperparams, 'seq_len', 'seq_len=') },
          { icon: Shuffle, title: 'ES-популяция (фаза 2)', detail: joinDetail(hp(hyperparams, 'population_size', 'N='), hp(hyperparams, 'sigma', 'σ=')) },
          { icon: Brain, title: 'Контроллер [z,h]→action' },
        ],
        loopCaption: 'Две фазы: сначала обучается world model на случайном опыте, затем она замораживается и крошечный линейный контроллер эволюционируется сверху',
      }
    case 'efficientzero':
      return {
        steps: [
          { icon: Eye, title: 'Представление s = h(obs)' },
          {
            icon: Crosshair,
            title: 'Gumbel/sampling-поиск (дерево)',
            detail: joinDetail(hp(hyperparams, 'num_simulations', 'sim='), hp(hyperparams, 'num_top_actions', 'top=')),
          },
          { icon: Target, title: 'Действие + policy/value target' },
          {
            icon: Brain,
            title: 'Unroll: reward + value + consistency',
            detail: hp(hyperparams, 'unroll_steps', 'unroll='),
          },
        ],
        loopCaption: 'Model-based planning: на каждом шаге заново ищется улучшенная политика через representation/dynamics/prediction сети, поиск даёт и действие, и обучающий target',
      }
    case 'unizero':
      return {
        steps: [
          { icon: History, title: 'Токены: [obs, act, obs, act, ...]', detail: hp(hyperparams, 'context_length', 'context=') },
          {
            icon: Layers,
            title: 'Causal Transformer',
            detail: joinDetail(hp(hyperparams, 'num_layers', 'layers='), hp(hyperparams, 'embed_dim', 'dim=')),
          },
          {
            icon: Crosshair,
            title: 'Gumbel-поиск по представлению',
            detail: hp(hyperparams, 'num_simulations', 'sim='),
          },
          {
            icon: Brain,
            title: 'Reward/value/policy + consistency',
            detail: hp(hyperparams, 'unroll_steps', 'unroll='),
          },
        ],
        loopCaption: 'Как EfficientZero, но вместо рекуррентной (LSTM/MLP) динамики — единый causal Transformer над явной последовательностью токенов [obs₀, act₀, obs₁, act₁, ...]: каждое прошлое наблюдение остаётся напрямую доступным через attention, а не сжимается в один вектор состояния',
      }
    case 'latentimzero':
      return {
        steps: [
          { icon: History, title: 'Posterior stochastic state', detail: hp(hyperparams, 'context_length', 'context=') },
          { icon: Moon, title: 'Prior imagination', detail: hp(hyperparams, 'imagination_horizon', 'horizon=') },
          { icon: Brain, title: 'Continue-aware λ-return actor-critic', detail: hp(hyperparams, 'imagination_lambda', 'λ=') },
          { icon: Crosshair, title: 'Uncertainty-gated Gumbel planner', detail: hp(hyperparams, 'num_simulations', 'sim≤') },
        ],
        loopCaption: 'Stochastic imagination planning: реальный posterior обучает модель мира, длинные prior-rollout дают actor/value сигналы, а uncertainty-gated поиск корректирует и дистиллирует политику',
      }
    case 'ippo':
      return {
        steps: [
          { icon: Users, title: 'N команд, N политик' },
          { icon: Shuffle, title: 'Общий rollout сцены', detail: hp(hyperparams, 'n_steps', 'n_steps=') },
          { icon: TrendingUp, title: 'Advantage (GAE) на команду', detail: hp(hyperparams, 'gamma', 'γ=') },
          { icon: Brain, title: 'PPO-обновление на команду', detail: hp(hyperparams, 'batch_size', 'batch=') },
        ],
        loopCaption: 'Independent learning: каждая команда — полностью отдельный PPO, команды взаимодействуют только через общую динамику среды',
      }
    case 'qmix':
      return {
        steps: [
          { icon: Users, title: 'N команд, общий Q на команду', detail: hp(hyperparams, 'exploration_final_eps', 'ε_min=') },
          { icon: Database, title: 'Буфер командных переходов', detail: hp(hyperparams, 'buffer_size', 'size=') },
          { icon: Layers, title: 'Mixing-сеть → Q_tot', detail: hp(hyperparams, 'mixing_embed_dim', 'embed=') },
          { icon: Target, title: 'TD-обновление на команду', detail: hp(hyperparams, 'target_update_interval', 'target каждые=') },
        ],
        loopCaption: 'Value decomposition: агенты команды делят одну Q-сеть, их Q-значения монотонно смешиваются в Q_tot — команды учатся независимо друг от друга',
      }
    case 'alphazero':
      return {
        steps: [
          {
            icon: Dices,
            title: 'Self-play + MCTS',
            detail: joinDetail(hp(hyperparams, 'games_per_iteration', 'игр='), hp(hyperparams, 'num_simulations', 'sim=')),
          },
          { icon: Database, title: 'Буфер партий', detail: hp(hyperparams, 'buffer_size', 'size=') },
          {
            icon: Brain,
            title: 'Обучение сети',
            detail: joinDetail(hp(hyperparams, 'epochs', 'epochs='), hp(hyperparams, 'batch_size', 'batch=')),
          },
          {
            icon: Swords,
            title: 'Арена vs чемпион',
            detail: joinDetail(hp(hyperparams, 'eval_games', 'игр='), hp(hyperparams, 'win_rate_threshold', 'порог=')),
          },
          { icon: Target, title: 'Принять / откатить', detail: 'чемпион меняется только при победе ≥ порога' },
        ],
        loopCaption: 'Повторяется num_iterations раз; чемпион обновляется только когда новая сеть выигрывает арену',
      }
    default:
      return {
        steps: [
          { icon: Eye, title: 'Наблюдение среды' },
          { icon: Dices, title: 'Действие агента' },
          { icon: Brain, title: 'Награда → обновление параметров' },
        ],
        loopCaption: 'Кастомный алгоритм — точный цикл зависит от плагина',
      }
  }
}

function StepChip({ icon: Icon, title, detail }: Step) {
  return (
    <div className="flex w-28 shrink-0 flex-col items-center gap-1 rounded-lg border border-border/70 bg-background/60 p-2 text-center">
      <div className="flex h-7 w-7 items-center justify-center rounded-full bg-primary/10 text-primary">
        <Icon className="h-3.5 w-3.5" />
      </div>
      <div className="text-[10.5px] font-medium leading-tight">{title}</div>
      {detail && <div className="text-[9px] leading-tight text-muted-foreground">{detail}</div>}
    </div>
  )
}

function StepArrow() {
  return <ArrowRight className="h-3 w-3 shrink-0 text-muted-foreground/40" />
}

function LoopDiagram({ algorithmId, kind, hyperparams }: { algorithmId: string; kind: 'gym' | 'alphazero'; hyperparams: Hyperparams }) {
  const family = familyOf(algorithmId, kind)
  const { steps: baseSteps, loopCaption: baseCaption } = buildSteps(family, hyperparams)
  const memory = MEMORY_FAMILIES.includes(family) ? memoryLabel(hyperparams) : null
  // Inserted right after the first step (observe/rollout/ε-greedy act) —
  // that's conceptually where the recurrent core sits, folding the raw
  // observation into a running hidden state before whatever comes next
  // (advantage estimation, the Q-/policy-update, ...) ever sees it.
  const steps = memory
    ? [baseSteps[0], { icon: History, title: `${memory}-память`, detail: hp(hyperparams, 'memory_hidden_size', 'hidden=') }, ...baseSteps.slice(1)]
    : baseSteps
  const loopCaption = memory ? `${baseCaption} · рекуррентная память (${memory}) переносится между шагами эпизода` : baseCaption
  return (
    <div>
      <div className="flex flex-wrap items-center gap-1.5">
        {steps.map((s, i) => (
          <Fragment key={i}>
            <StepChip {...s} />
            {i < steps.length - 1 && <StepArrow />}
          </Fragment>
        ))}
      </div>
      <div className="mt-2 flex items-center gap-1 text-[10px] text-muted-foreground">
        <Repeat className="h-3 w-3 shrink-0" />
        <span>{loopCaption}</span>
      </div>
    </div>
  )
}

export interface AlgorithmDiagramNetwork {
  policy?: string
  layers?: string[]
  components?: ArchitectureComponent[]
  totalParams?: number
  trainableParams?: number
  inputShape?: number[] | null
  outputShape?: number[] | null
  channels?: number
  numBlocks?: number
  actionSize?: number
  inputPlanes?: number
  rows?: number
  cols?: number
  note?: string | null
}

function ShapeChip({ label, shape }: { label: string; shape?: number[] | null }) {
  return (
    <div className="flex w-28 shrink-0 flex-col items-center gap-1 rounded-lg border border-dashed border-border/70 bg-background/40 p-2 text-center">
      <div className="text-[10.5px] font-medium leading-tight">{label}</div>
      <div className="font-mono text-[9px] leading-tight text-muted-foreground">
        {shape && shape.length > 0 ? `[${shape.join('×')}]` : '—'}
      </div>
    </div>
  )
}

/** Pretty-prints an inspect layer string ("Linear(64→64)") into a compact
 * two-line chip — no per-network-family logic needed since this comes
 * straight from real introspection of whatever module the algorithm (built-
 * in or plugin) actually exposes, so it works for architectures we've never
 * heard of too. */
function LayerChip({ layer }: { layer: string }) {
  const match = layer.match(/^([A-Za-z0-9]+)\((.*)\)$/)
  return (
    <div className="flex w-24 shrink-0 flex-col items-center gap-0.5 rounded-lg border border-border/70 bg-background/60 px-2 py-1.5 text-center">
      <div className="text-[10px] font-medium leading-tight">{match ? match[1] : layer}</div>
      {match && <div className="font-mono text-[9px] leading-tight text-muted-foreground">{match[2]}</div>}
    </div>
  )
}

function componentLabel(name: string): string {
  const labels: Record<string, string> = {
    representation: 'Representation / encoder',
    dynamics: 'Dynamics model',
    prediction: 'Prediction heads',
    tokenizer: 'Observation tokenizer',
    action_embed: 'Action embedding',
    transformer: 'Causal Transformer',
    heads: 'Reward · Value · Policy heads',
    projector: 'SimSiam projector',
    predictor: 'SimSiam predictor',
    target_tokenizer: 'EMA observation tokenizer',
    target_transformer: 'EMA Causal Transformer',
    target_heads: 'EMA target heads',
    direct_parameters: 'Отдельные обучаемые параметры',
  }
  return labels[name] ?? name.replaceAll('_', ' ')
}

function ArchitectureComponentCard({ component }: { component: ArchitectureComponent }) {
  return (
    <div className={cn(
      'min-w-0 rounded-lg border bg-background/50',
      component.role === 'target' && 'border-dashed bg-muted/20',
    )}>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 border-b border-border/70 px-3 py-2">
        <span className="text-xs font-semibold">{componentLabel(component.name)}</span>
        <span className="font-mono text-[9px] text-muted-foreground">{component.type}</span>
        {component.role === 'target' && (
          <span className="rounded bg-muted px-1.5 py-0.5 text-[9px] font-medium text-muted-foreground">TARGET / EMA</span>
        )}
        <span className="ml-auto font-mono text-[9px] text-muted-foreground">
          {formatNumber(component.params)} параметров
          {component.trainable_params !== component.params && ` · ${formatNumber(component.trainable_params)} обучаемых`}
        </span>
      </div>
      {component.layers.length > 0 ? (
        <div className="divide-y divide-border/40">
          {component.layers.map((layer, index) => (
            <div key={`${layer.path}-${index}`} className="grid grid-cols-[minmax(8rem,1.25fr)_minmax(7rem,1fr)_auto] items-center gap-2 px-3 py-1.5 text-[10px]">
              <span className="truncate font-mono text-muted-foreground" title={layer.path}>{layer.path}</span>
              <span className="min-w-0">
                <span className="font-medium">{layer.type}</span>
                {layer.detail && <span className="ml-1.5 font-mono text-[9px] text-muted-foreground">{layer.detail}</span>}
              </span>
              <span className="font-mono text-[9px] text-muted-foreground">{formatNumber(layer.params)}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="px-3 py-2 text-[10px] text-muted-foreground">Нет вложенных слоёв с параметрами</div>
      )}
    </div>
  )
}

function NetworkDiagram({
  network,
  compact,
}: {
  network?: AlgorithmDiagramNetwork | null
  compact?: boolean
}) {
  if (!network) {
    return <p className="text-xs text-muted-foreground">Нет данных о сети</p>
  }

  const isBoardNet = network.channels != null && network.numBlocks != null
  const shownLayers = network.layers ?? []
  const hasComponents = (network.components?.length ?? 0) > 0
  const inputShape = network.inputShape ?? (
    isBoardNet ? [network.inputPlanes ?? 3, network.rows ?? 0, network.cols ?? 0] : undefined
  )
  const outputShape = network.outputShape ?? (isBoardNet && network.actionSize != null ? [network.actionSize] : undefined)

  return (
    <div>
      {compact ? (
        <div className="flex flex-wrap items-center gap-1.5">
          <ShapeChip label="Вход" shape={inputShape} />
          <StepArrow />
          <ShapeChip label="Выход" shape={outputShape} />
        </div>
      ) : hasComponents ? (
        <div className="space-y-2">
          <ShapeChip label="Вход среды" shape={network.inputShape} />
          <div className="grid grid-cols-1 gap-2 xl:grid-cols-2">
            {network.components!.map((component, index) => (
              <ArchitectureComponentCard key={`${component.name}-${index}`} component={component} />
            ))}
          </div>
          <ShapeChip label="Действие среды" shape={network.outputShape} />
        </div>
      ) : <div className="flex flex-wrap items-center gap-1.5">
        {isBoardNet ? (
          <>
            <ShapeChip label="Вход (доска)" shape={[network.inputPlanes ?? 3, network.rows ?? 0, network.cols ?? 0]} />
            <StepArrow />
            <StepChip icon={Layers} title="Conv stem" detail={`${network.channels} ch`} />
            <StepArrow />
            <StepChip icon={Layers} title={`ResBlock ×${network.numBlocks}`} detail={`${network.channels} ch, 3×3`} />
            <StepArrow />
            <div className="flex items-center gap-1.5 rounded-lg border border-dashed border-border/70 p-1.5">
              <StepChip icon={Target} title="Policy head" detail={`→ ${network.actionSize ?? '?'}`} />
              <StepChip icon={TrendingUp} title="Value head" detail="→ 1 (tanh)" />
            </div>
          </>
        ) : (
          <>
            <ShapeChip label="Вход" shape={network.inputShape} />
            {shownLayers.length > 0 && <StepArrow />}
            {shownLayers.map((l, i) => (
              <Fragment key={i}>
                <LayerChip layer={l} />
                {i < shownLayers.length - 1 && <StepArrow />}
              </Fragment>
            ))}
            <StepArrow />
            <ShapeChip label="Выход" shape={network.outputShape} />
          </>
        )}
      </div>}
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
        {network.policy && <span>Policy: <span className="font-mono text-foreground/80">{network.policy}</span></span>}
        {network.totalParams != null && (
          <span>Всего параметров: <span className="font-mono text-foreground/80">{formatNumber(network.totalParams)}</span></span>
        )}
        {network.trainableParams != null && (
          <span>Обучаемых: <span className="font-mono text-foreground/80">{formatNumber(network.trainableParams)}</span></span>
        )}
        {network.note && <span className="italic">{network.note}</span>}
      </div>
    </div>
  )
}

export interface AlgorithmDiagramProps {
  // Optional because callers that only render `show.network` (e.g. the
  // Designer's env/network InspectPanel, which knows the net's shape but
  // not which algorithm produced it in a strongly-typed way at that call
  // site) have nothing meaningful to pass here.
  algorithmId?: string
  kind?: 'gym' | 'alphazero'
  hyperparams?: Hyperparams
  network?: AlgorithmDiagramNetwork | null
  show?: { loop?: boolean; network?: boolean }
  /** Skip the small "Как учится"/"Архитектура сети" section captions —
   * for call sites that already have their own, more specific Card title
   * right above (e.g. Designer's InspectPanel says "Нейросеть" already). */
  hideTitles?: boolean
  /** Designer-side preview: I/O shapes and parameter totals only.
   * Skips the per-layer / per-component weight dump, which overflows the
   * floating inspect card for world-model algorithms. */
  compact?: boolean
  className?: string
}

/** Two-part "what is this algorithm actually doing" visualization, reused
 * for every built-in algorithm (PPO/A2C/DQN/AlphaZero) *and* custom plugins:
 * - the training-loop diagram only needs the algorithm id/kind/hyperparams,
 *   so it renders instantly in the Designer before anything is inspected;
 * - the network diagram is driven entirely by real introspection data
 *   (`InspectNetwork` in the Designer, `MetricsSnapshot` fields in the
 *   Monitor) so it's honest about whatever architecture a plugin actually
 *   builds, without needing per-plugin special-casing. */
export function AlgorithmDiagram({ algorithmId, kind, hyperparams, network, show, hideTitles, compact, className }: AlgorithmDiagramProps) {
  const showLoop = (show?.loop ?? true) && algorithmId != null && kind != null
  const showNetwork = show?.network ?? true
  return (
    <div className={cn('space-y-3', className)}>
      {showLoop && (
        <Section title="Как учится" hide={hideTitles}>
          <LoopDiagram algorithmId={algorithmId!} kind={kind!} hyperparams={hyperparams} />
        </Section>
      )}
      {showNetwork && (
        <Section title="Архитектура сети" hide={hideTitles}>
          <NetworkDiagram network={network} compact={compact} />
        </Section>
      )}
    </div>
  )
}

function Section({ title, hide, children }: { title: string; hide?: boolean; children: ReactNode }) {
  return (
    <div>
      {!hide && (
        <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">{title}</div>
      )}
      {children}
    </div>
  )
}

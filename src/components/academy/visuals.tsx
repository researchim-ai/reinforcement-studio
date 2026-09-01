import { Fragment, type ReactNode } from 'react'
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis,
} from 'recharts'
import { ArrowRight, type LucideIcon, ExternalLink, Info, Lightbulb, TriangleAlert, CircleCheck } from 'lucide-react'
import { cn } from '@/lib/utils'

/** Inline citation link — "(Author et al., Year)" that actually opens the
 * paper. Used right next to the in-text mention of a paper throughout the
 * course instead of leaving citations as dead text. Prefer a direct arXiv
 * abstract page when one reliably exists for a paper; for pre-arXiv-era
 * classics (Tiger, Hallway, RockSample, ...) a Google Scholar search on the
 * exact title is used instead — always resolves to the right paper without
 * risking a stale/wrong direct link to a paywalled PDF. */
export function PaperLink({ children, url }: { children: ReactNode; url: string }) {
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-0.5 whitespace-nowrap text-primary/90 underline decoration-primary/30 underline-offset-2 hover:text-primary hover:decoration-primary"
    >
      {children}
      <ExternalLink className="h-2.5 w-2.5 shrink-0" />
    </a>
  )
}

// ---------------------------------------------------------------------------
// Generic building blocks reused across every lesson — kept intentionally
// close to the compact chip/arrow style already used in
// AlgorithmDiagram.tsx (Designer/Monitor) so an Academy diagram never feels
// like a different visual language from the rest of the app, just bigger
// and with more room for explanatory text.
// ---------------------------------------------------------------------------

export interface FlowItem {
  icon: LucideIcon
  title: string
  detail?: string
}

export function FlowChip({ icon: Icon, title, detail }: FlowItem) {
  return (
    <div className="flex w-32 shrink-0 flex-col items-center gap-1 rounded-lg border border-border/70 bg-background/60 p-2.5 text-center">
      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-primary">
        <Icon className="h-4 w-4" />
      </div>
      <div className="text-[11px] font-medium leading-tight">{title}</div>
      {detail && <div className="text-[10px] leading-tight text-muted-foreground">{detail}</div>}
    </div>
  )
}

export function FlowArrow({ label }: { label?: string }) {
  return (
    <div className="flex shrink-0 flex-col items-center gap-0.5 text-muted-foreground/50">
      <ArrowRight className="h-4 w-4" />
      {label && <span className="text-[9px] leading-tight text-muted-foreground/70">{label}</span>}
    </div>
  )
}

/** A horizontal row of `FlowChip`s connected by arrows — the workhorse
 * diagram for "these boxes feed into that box" explanations (replay buffer →
 * update, dueling heads recombining, actor/critic split, ...). Wraps onto
 * multiple lines on narrow screens since flex-wrap is set. */
export function FlowRow({ items, className }: { items: FlowItem[]; className?: string }) {
  return (
    <div className={cn('flex flex-wrap items-center gap-2', className)}>
      {items.map((item, i) => (
        <Fragment key={i}>
          <FlowChip {...item} />
          {i < items.length - 1 && <FlowArrow />}
        </Fragment>
      ))}
    </div>
  )
}

export function Formula({ children, caption }: { children: ReactNode; caption?: string }) {
  return (
    <div className="rounded-lg border border-dashed border-border/70 bg-muted/40 px-4 py-3">
      <div className="overflow-x-auto whitespace-pre font-mono text-[13px] leading-relaxed text-foreground/90">{children}</div>
      {caption && <div className="mt-1.5 text-[11px] text-muted-foreground">{caption}</div>}
    </div>
  )
}

const CALLOUT_STYLE: Record<'info' | 'tip' | 'warning' | 'good', { icon: LucideIcon; cls: string }> = {
  info: { icon: Info, cls: 'border-sky-500/30 bg-sky-500/5 text-sky-400' },
  tip: { icon: Lightbulb, cls: 'border-amber-500/30 bg-amber-500/5 text-amber-400' },
  warning: { icon: TriangleAlert, cls: 'border-destructive/30 bg-destructive/5 text-destructive' },
  good: { icon: CircleCheck, cls: 'border-emerald-500/30 bg-emerald-500/5 text-emerald-400' },
}

export function Callout({ tone = 'info', title, children }: { tone?: 'info' | 'tip' | 'warning' | 'good'; title: string; children: ReactNode }) {
  const { icon: Icon, cls } = CALLOUT_STYLE[tone]
  return (
    <div className={cn('rounded-lg border px-3.5 py-3', cls)}>
      <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold">
        <Icon className="h-3.5 w-3.5" />
        {title}
      </div>
      <div className="text-[13px] leading-relaxed text-foreground/85">{children}</div>
    </div>
  )
}

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-2.5">
      <h3 className="text-sm font-semibold text-foreground">{title}</h3>
      <div className="space-y-2.5 text-[13px] leading-relaxed text-muted-foreground">{children}</div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Charts (recharts — same lib/colors already used in TrainingMonitor.tsx)
// ---------------------------------------------------------------------------

const CHART_GRID = 'oklch(0.3 0 0)'
const CHART_AXIS = 'oklch(0.65 0 0)'
const CHART_LINE_1 = 'oklch(0.7 0.15 260)'
const CHART_LINE_2 = 'oklch(0.7 0.18 25)'
const CHART_TOOLTIP = { background: 'oklch(0.17 0 0)', border: '1px solid oklch(0.3 0 0)', fontSize: 12 }

/** ε-greedy exploration schedule — linear decay from 1.0 down to
 * `finalEps` over `fraction` of training, then flat. Purely illustrative
 * (uses the same defaults DQN/Rainbow DQN ship with), not wired to a live
 * run — the point is to make "exploration_fraction" mean something visual
 * instead of just a number in a form. */
export function EpsilonDecayChart({ fraction = 0.2, finalEps = 0.05 }: { fraction?: number; finalEps?: number }) {
  const data = Array.from({ length: 41 }, (_, i) => {
    const x = i / 40
    const eps = x <= fraction ? 1 - (1 - finalEps) * (x / fraction) : finalEps
    return { x: Math.round(x * 100), eps: Number(eps.toFixed(3)) }
  })
  return (
    <div className="h-40 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 4, right: 12, left: -18, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
          <XAxis dataKey="x" stroke={CHART_AXIS} fontSize={10} unit="%" tickLine={false} />
          <YAxis stroke={CHART_AXIS} fontSize={10} domain={[0, 1]} tickLine={false} />
          <RTooltip contentStyle={CHART_TOOLTIP} formatter={(v: number) => v.toFixed(2)} labelFormatter={(l) => `${l}% обучения`} />
          <Line type="monotone" dataKey="eps" name="ε" stroke={CHART_LINE_1} dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

/** The PPO clipped surrogate objective, L(r)/A as a function of the
 * probability ratio r = π_new(a|s) / π_old(a|s), for a good action (A>0,
 * blue) and a bad one (A<0, red) — the actual textbook figure from the PPO
 * paper (Schulman et al., 2017), computed directly from the clip formula
 * rather than approximated. */
export function PPOClipChart({ eps = 0.2 }: { eps?: number }) {
  const data = Array.from({ length: 41 }, (_, i) => {
    const r = i * 0.05
    return {
      r: Number(r.toFixed(2)),
      posA: Number(Math.min(r, 1 + eps).toFixed(3)),
      negA: Number((-Math.max(r, 1 - eps)).toFixed(3)),
    }
  })
  return (
    <div className="h-44 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 4, right: 12, left: -8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
          <XAxis dataKey="r" stroke={CHART_AXIS} fontSize={10} tickLine={false} />
          <YAxis stroke={CHART_AXIS} fontSize={10} tickLine={false} />
          <ReferenceLine x={1} stroke={CHART_AXIS} strokeDasharray="2 2" />
          <ReferenceLine x={1 - eps} stroke={CHART_AXIS} strokeOpacity={0.4} strokeDasharray="2 2" />
          <ReferenceLine x={1 + eps} stroke={CHART_AXIS} strokeOpacity={0.4} strokeDasharray="2 2" />
          <RTooltip contentStyle={CHART_TOOLTIP} labelFormatter={(l) => `r = ${l}`} />
          <Line type="monotone" dataKey="posA" name="L/A, A>0 (хорошее действие)" stroke={CHART_LINE_1} dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="negA" name="L/A, A<0 (плохое действие)" stroke={CHART_LINE_2} dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

/** How much a reward t steps in the future is worth *today* under discount
 * factor γ — γ close to 1 barely discounts anything (agent is "patient"),
 * γ close to 0 only cares about the immediate step. */
export function DiscountChart() {
  const data = Array.from({ length: 13 }, (_, t) => ({
    t,
    'γ=0.99': Number((0.99 ** t).toFixed(3)),
    'γ=0.9': Number((0.9 ** t).toFixed(3)),
  }))
  return (
    <div className="h-40 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 4, right: 12, left: -18, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
          <XAxis dataKey="t" stroke={CHART_AXIS} fontSize={10} tickLine={false} label={{ value: 'шагов вперёд', position: 'insideBottom', offset: -2, fontSize: 10, fill: CHART_AXIS }} />
          <YAxis stroke={CHART_AXIS} fontSize={10} domain={[0, 1]} tickLine={false} />
          <RTooltip contentStyle={CHART_TOOLTIP} />
          <Line type="monotone" dataKey="γ=0.99" stroke={CHART_LINE_1} dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="γ=0.9" stroke={CHART_LINE_2} dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Hand-built conceptual diagrams (no chart data involved)
// ---------------------------------------------------------------------------

/** The canonical agent/environment RL loop — deliberately more spelled-out
 * than the compact per-algorithm loop chips elsewhere in the app, since
 * this is the very first diagram a newcomer sees in the whole course. */
export function AgentEnvLoopDiagram() {
  return (
    <div className="flex items-center justify-center gap-6 py-2">
      <div className="flex h-24 w-32 flex-col items-center justify-center gap-1 rounded-xl border-2 border-primary/40 bg-primary/5 text-center">
        <span className="text-sm font-semibold">Агент</span>
        <span className="text-[10px] text-muted-foreground">политика π(a|s)</span>
      </div>
      <div className="flex flex-col items-center gap-3">
        <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <span>действие aₜ</span>
          <ArrowRight className="h-3.5 w-3.5" />
        </div>
        <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <ArrowRight className="h-3.5 w-3.5 rotate-180" />
          <span>состояние sₜ₊₁, награда rₜ₊₁</span>
        </div>
      </div>
      <div className="flex h-24 w-32 flex-col items-center justify-center gap-1 rounded-xl border-2 border-border bg-muted/40 text-center">
        <span className="text-sm font-semibold">Среда</span>
        <span className="text-[10px] text-muted-foreground">динамика + reward</span>
      </div>
    </div>
  )
}

/** A tiny illustrative MCTS tree — one expansion + backup, drawn as static
 * SVG. Not driven by real numbers (this is the concept lesson, not a live
 * replay of an actual search), just enough structure to make
 * "N(s,a), Q(s,a), UCB" concrete instead of abstract. */
export function MCTSTreeDiagram() {
  const children: { x: number; n: number; q: string; best?: boolean }[] = [
    { x: 60, n: 12, q: '+0.41' },
    { x: 160, n: 31, q: '+0.72', best: true },
    { x: 260, n: 4, q: '-0.10' },
  ]
  return (
    <svg viewBox="0 0 320 150" className="h-40 w-full max-w-md">
      {children.map((c) => (
        <line key={c.x} x1={160} y1={30} x2={c.x} y2={95} stroke="oklch(0.4 0 0)" strokeWidth={1.5} />
      ))}
      <circle cx={160} cy={20} r={16} className="fill-primary/20" stroke="oklch(0.7 0.15 260)" strokeWidth={1.5} />
      <text x={160} y={24} textAnchor="middle" className="fill-foreground text-[9px] font-medium">root</text>
      {children.map((c) => (
        <g key={c.x}>
          <circle
            cx={c.x}
            cy={105}
            r={18}
            className={c.best ? 'fill-emerald-500/20' : 'fill-background'}
            stroke={c.best ? 'oklch(0.75 0.18 150)' : 'oklch(0.45 0 0)'}
            strokeWidth={c.best ? 2 : 1.5}
          />
          <text x={c.x} y={101} textAnchor="middle" className="fill-foreground text-[8px] font-semibold">N={c.n}</text>
          <text x={c.x} y={112} textAnchor="middle" className="fill-muted-foreground text-[8px]">Q={c.q}</text>
        </g>
      ))}
      <text x={160} y={140} textAnchor="middle" className="fill-muted-foreground text-[9px]">
        выбирается ход с максимальным UCB = Q + c·P(s,a)·√N(s)/(1+N(s,a))
      </text>
    </svg>
  )
}

/** Hidden state carried between timesteps — the one picture that makes
 * "recurrent" concrete: without it, every box below would be independent
 * and blind to everything before it. */
export function MemoryTimelineDiagram() {
  const steps = ['t=0 (подсказка)', 't=1', 't=2', '…', 't=T (решение)']
  return (
    <div>
      <div className="flex items-center gap-0 overflow-x-auto py-2">
        {steps.map((label, i) => (
          <Fragment key={i}>
            <div className="flex flex-col items-center gap-1.5">
              <div className="flex h-14 w-20 shrink-0 flex-col items-center justify-center gap-0.5 rounded-lg border border-border/70 bg-background/60 text-center">
                <span className="text-[9px] text-muted-foreground">oₜ</span>
                <span className="text-[10px] font-medium">LSTM/GRU</span>
              </div>
              <span className="whitespace-nowrap text-[9px] text-muted-foreground">{label}</span>
            </div>
            {i < steps.length - 1 && (
              <div className="flex shrink-0 flex-col items-center px-1 pb-4">
                <span className="text-[8px] text-primary/80">hₜ</span>
                <ArrowRight className="h-3.5 w-3.5 text-primary/60" />
              </div>
            )}
          </Fragment>
        ))}
      </div>
      <p className="text-[10px] text-muted-foreground">
        Скрытое состояние hₜ несёт информацию из всех прошлых шагов — обрывается только на границе эпизода.
      </p>
    </div>
  )
}

/** The single picture that makes "world model" concrete: one real step
 * (expensive — an actual `env.step()`) branches into a short chain of
 * *imagined* steps (cheap — pure forward passes through the learned
 * dynamics model, dashed to mark "not real"). This is the shared payoff
 * every one of Dreamer/MBPO/PETS/World Models exploits in its own way —
 * imagination-based actor-critic training, model-augmented replay, or
 * direct planning — so this diagram is reused across all four lessons
 * rather than redrawn per-algorithm. */
export function ImaginationDiagram() {
  const imagined = ['ẑ₂ / ŝ₂', 'ẑ₃ / ŝ₃', 'ẑ₄ / ŝ₄', '…']
  return (
    <div className="space-y-2.5 py-2">
      <div className="flex items-center gap-0 overflow-x-auto">
        <div className="flex h-14 w-28 shrink-0 flex-col items-center justify-center gap-0.5 rounded-lg border-2 border-primary/50 bg-primary/10 text-center">
          <span className="text-[10px] font-medium">s₀ (реальность)</span>
        </div>
        <div className="flex shrink-0 flex-col items-center px-1.5">
          <span className="whitespace-nowrap text-[8px] text-muted-foreground">1× env.step()</span>
          <ArrowRight className="h-3.5 w-3.5 text-primary/60" />
        </div>
        <div className="flex h-14 w-28 shrink-0 flex-col items-center justify-center gap-0.5 rounded-lg border-2 border-primary/50 bg-primary/10 text-center">
          <span className="text-[10px] font-medium">s₁ (реальность)</span>
        </div>
      </div>
      <div className="ml-[7.5rem] flex items-center gap-0 overflow-x-auto border-l-2 border-dashed border-amber-500/40 pl-3">
        {imagined.map((label, i) => (
          <Fragment key={label}>
            <div className="flex h-11 w-16 shrink-0 flex-col items-center justify-center rounded-lg border border-dashed border-amber-500/50 bg-amber-500/5 text-center">
              <span className="text-[10px] text-amber-500">{label}</span>
            </div>
            {i < imagined.length - 1 && <ArrowRight className="h-3 w-3 shrink-0 text-amber-500/40" />}
          </Fragment>
        ))}
        <span className="ml-2 whitespace-nowrap text-[9px] text-muted-foreground">
          «воображение» — только forward-проходы через модель, ни одного настоящего env.step()
        </span>
      </div>
    </div>
  )
}

/** Full (Markov) state vs. what the agent actually observes in a POMDP —
 * the crossed-out rows are exactly the information that's missing, which
 * is the whole reason a feed-forward policy is capped and a recurrent one
 * is not. */
export function PomdpCompareDiagram({
  full,
  observed,
}: {
  full: string[]
  observed: string[]
}) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      <div className="rounded-lg border border-border/70 bg-background/40 p-3">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">Полное (Markov) состояние</div>
        <ul className="space-y-1 text-[12px]">
          {full.map((f) => (
            <li key={f} className={cn('rounded px-1.5 py-0.5', !observed.includes(f) && 'text-muted-foreground/50 line-through')}>
              {f}
            </li>
          ))}
        </ul>
      </div>
      <div className="rounded-lg border border-primary/30 bg-primary/5 p-3">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">Наблюдение агента</div>
        <ul className="space-y-1 text-[12px]">
          {observed.map((o) => (
            <li key={o} className="rounded bg-primary/10 px-1.5 py-0.5">{o}</li>
          ))}
        </ul>
      </div>
    </div>
  )
}

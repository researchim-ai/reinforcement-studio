import { Fragment, type ReactNode } from 'react'
import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis,
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

const SUBSCRIPT_DIGITS = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']

// ---------------------------------------------------------------------------
// MuZero-family (EfficientZero/UniZero) diagrams.
// ---------------------------------------------------------------------------

/** The MuZero/EfficientZero training-unroll picture: only `s₀` is ever
 * built from a *real* observation (via `h()`) — every later `sₖ` in the
 * unroll comes purely from `dynamics g(s, a)`, chained forward through
 * known real actions, never re-grounded in a real `obsₖ`. The real
 * `obs₁, obs₂, ...` of this same window still exist (dashed) but only feed
 * the *consistency loss target*, never the dynamics' input — the single
 * fact `TokenAttentionDiagram` below (UniZero) is drawn to contrast with:
 * UniZero's Transformer instead re-reads every real token directly. */
export function LatentChainDiagram() {
  const n = 4
  const cols = Array.from({ length: 2 * n - 1 }, (_, i) => (i % 2 === 0 ? '4.25rem' : '2.75rem')).join(' ')
  return (
    <div className="overflow-x-auto py-2">
      <div className="grid items-center gap-y-1.5" style={{ gridTemplateColumns: cols, width: 'max-content' }}>
        {Array.from({ length: n }, (_, i) => (
          <div
            key={`s-${i}`}
            className={cn(
              'flex h-12 flex-col items-center justify-center rounded-lg border-2 text-center',
              i === 0 ? 'border-primary/60 bg-primary/10' : 'border-border/70 bg-background/60',
            )}
            style={{ gridRow: 1, gridColumn: 2 * i + 1 }}
          >
            <span className="text-[11px] font-semibold">s{SUBSCRIPT_DIGITS[i]}</span>
            <span className="text-[8px] text-muted-foreground">policy/value</span>
          </div>
        ))}
        {Array.from({ length: n - 1 }, (_, i) => (
          <div key={`arr-${i}`} className="flex flex-col items-center" style={{ gridRow: 1, gridColumn: 2 * i + 2 }}>
            <span className="whitespace-nowrap text-[7px] text-muted-foreground">g(s,a{SUBSCRIPT_DIGITS[i]})</span>
            <ArrowRight className="h-3.5 w-3.5 text-muted-foreground/60" />
          </div>
        ))}
        {Array.from({ length: n }, (_, i) => (
          <div key={`o-${i}`} className="flex flex-col items-center gap-0.5" style={{ gridRow: 2, gridColumn: 2 * i + 1 }}>
            {i === 0 ? (
              <>
                <ArrowRight className="h-3 w-3 rotate-90 text-primary/70" />
                <span className="whitespace-nowrap text-[8px] text-primary/80">obs₀ → h()</span>
              </>
            ) : (
              <>
                <div className="h-2.5 w-px border-l border-dashed border-amber-500/50" />
                <span className="whitespace-nowrap text-[7.5px] text-amber-500/80">obs{SUBSCRIPT_DIGITS[i]} — только цель consistency</span>
              </>
            )}
          </div>
        ))}
      </div>
      <p className="mt-2.5 text-[10px] text-muted-foreground">
        Только s₀ строится из настоящего наблюдения (h(obs₀)) — s₁, s₂, s₃ получаются чисто из dynamics g(s,a) по
        известным реальным действиям, а настоящие obs₁, obs₂, obs₃ этого же окна участвуют только как <i>цель</i>
        consistency loss, никогда как вход динамики. Вся история до sₖ доступна dynamics-сети только через то, что
        успело «уложиться» в один вектор sₖ₋₁.
      </p>
    </div>
  )
}

/** The single picture UniZero's whole pitch rests on: every past
 * observation/action stays its own token, and the Transformer's causal
 * attention lets the *last* token look directly at every earlier one in
 * the window — nothing is squeezed into one carried-over vector the way
 * `LatentChainDiagram` (previous lesson) draws it. Also marks exactly
 * which token position each head reads its hidden state off of. */
export function TokenAttentionDiagram() {
  const tokens: { label: string; kind: 'obs' | 'act' | 'cur' }[] = [
    { label: 'obs₀', kind: 'obs' },
    { label: 'act₀', kind: 'act' },
    { label: 'obs₁', kind: 'obs' },
    { label: 'act₁', kind: 'act' },
    { label: 'obsₜ', kind: 'cur' },
  ]
  const xs = [16, 78, 140, 202, 264]
  const w = 44
  const h = 28
  const y = 96
  const centers = xs.map((x) => x + w / 2)
  const lastCx = centers[centers.length - 1]
  const arcHeights = [54, 44, 32, 20]
  const fill: Record<string, string> = { obs: 'fill-background', act: 'fill-amber-500/10', cur: 'fill-primary/20' }
  const stroke: Record<string, string> = {
    obs: 'oklch(0.45 0 0)', act: 'oklch(0.75 0.15 70)', cur: 'oklch(0.7 0.15 260)',
  }

  return (
    <svg viewBox="0 0 336 172" className="h-44 w-full max-w-xl">
      <text x={168} y={13} textAnchor="middle" className="fill-muted-foreground text-[9px]">
        причинное (causal) внимание: последний токен «видит» каждый прошлый obs/act-токен окна памяти напрямую
      </text>

      {centers.slice(0, -1).map((cx, i) => (
        <path
          key={`arc-${i}`}
          d={`M ${lastCx} ${y - 2} Q ${(lastCx + cx) / 2} ${y - 2 - arcHeights[i]} ${cx} ${y - 2}`}
          fill="none"
          stroke="oklch(0.7 0.15 260)"
          strokeOpacity={0.5}
          strokeWidth={1.3}
        />
      ))}

      {tokens.map((t, i) => (
        <g key={t.label}>
          <rect
            x={xs[i]} y={y} width={w} height={h} rx={5}
            className={fill[t.kind]} stroke={stroke[t.kind]} strokeWidth={t.kind === 'cur' ? 2 : 1.3}
          />
          <text x={centers[i]} y={y + h / 2 + 3} textAnchor="middle" className="fill-foreground text-[9px] font-medium">
            {t.label}
          </text>
        </g>
      ))}

      <line x1={centers[3]} y1={y + h + 3} x2={centers[3]} y2={y + h + 9} stroke="oklch(0.75 0.15 70)" strokeWidth={1} />
      <text x={centers[3]} y={y + h + 20} textAnchor="middle" className="fill-amber-500 text-[8px]">reward здесь</text>
      <line x1={centers[4]} y1={y + h + 3} x2={centers[4]} y2={y + h + 21} stroke="oklch(0.7 0.15 260)" strokeWidth={1} />
      <text x={centers[4]} y={y + h + 32} textAnchor="middle" className="fill-primary text-[8px]">policy/value здесь</text>
    </svg>
  )
}

/** The Gumbel-search "funnel": round 1 samples m candidates via
 * Gumbel-Top-k (logits + Gumbel noise, no Dirichlet noise needed), then
 * Sequential Halving repeatedly throws away the worse half by
 * `completed Q` until one candidate — the real `env_action` — survives.
 * Static/illustrative numbers, same spirit as `MCTSTreeDiagram`
 * (AlphaZero lesson) but for the halving tournament instead of PUCT. */
export function GumbelHalvingDiagram() {
  const round1 = [
    { x: 8, label: 'a₁', score: '0.62', alive: true },
    { x: 88, label: 'a₂', score: '0.88', alive: true },
    { x: 168, label: 'a₃', score: '0.35', alive: false },
    { x: 248, label: 'a₄', score: '0.51', alive: false },
  ]
  const round2 = [{ x: 8, label: 'a₁', n: 6 }, { x: 88, label: 'a₂', n: 10 }]
  const winnerX = 88

  return (
    <svg viewBox="0 0 336 208" className="h-52 w-full max-w-xl">
      <text x={168} y={12} textAnchor="middle" className="fill-muted-foreground text-[9px]">
        раунд 1: Gumbel-Top-k из prior — m кандидатов проходят в дерево
      </text>
      {round1.map((c) => (
        <g key={c.label} opacity={c.alive ? 1 : 0.35}>
          <rect
            x={c.x} y={22} width={56} height={30} rx={5}
            className={c.alive ? 'fill-primary/10' : 'fill-background'}
            stroke={c.alive ? 'oklch(0.7 0.15 260)' : 'oklch(0.45 0 0)'} strokeWidth={1.3}
          />
          <text x={c.x + 28} y={36} textAnchor="middle" className="fill-foreground text-[9px] font-medium">{c.label}</text>
          <text x={c.x + 28} y={47} textAnchor="middle" className="fill-muted-foreground text-[7.5px]">g+prior={c.score}</text>
          {!c.alive && <line x1={c.x + 6} y1={26} x2={c.x + 50} y2={48} stroke="oklch(0.55 0.2 25)" strokeWidth={1.5} />}
        </g>
      ))}

      <text x={168} y={72} textAnchor="middle" className="fill-muted-foreground text-[9px]">
        раунд 2 (Sequential Halving): половина отброшена по completed Q
      </text>
      {round2.map((c) => (
        <g key={c.label}>
          <line x1={c.x + 28} y1={52} x2={c.x + 28} y2={84} stroke="oklch(0.4 0 0)" strokeWidth={1.2} />
          <rect x={c.x} y={84} width={56} height={30} rx={5} className="fill-primary/10" stroke="oklch(0.7 0.15 260)" strokeWidth={1.3} />
          <text x={c.x + 28} y={98} textAnchor="middle" className="fill-foreground text-[9px] font-medium">{c.label}</text>
          <text x={c.x + 28} y={109} textAnchor="middle" className="fill-muted-foreground text-[7.5px]">N={c.n}</text>
        </g>
      ))}

      <text x={168} y={138} textAnchor="middle" className="fill-muted-foreground text-[9px]">
        финал: победитель — max visit-count среди выживших
      </text>
      <line x1={winnerX + 28} y1={114} x2={winnerX + 28} y2={150} stroke="oklch(0.4 0 0)" strokeWidth={1.2} />
      <rect x={winnerX - 20} y={150} width={96} height={38} rx={6} className="fill-emerald-500/15" stroke="oklch(0.75 0.18 150)" strokeWidth={2} />
      <text x={winnerX + 28} y={166} textAnchor="middle" className="fill-foreground text-[9px] font-semibold">env_action = a₂</text>
      <text x={winnerX + 28} y={178} textAnchor="middle" className="fill-muted-foreground text-[8px]">N=32 (num_simulations)</text>
    </svg>
  )
}

/** Two-hot categorical target — MuZero Appendix F's answer to "one big
 * reward spike shouldn't blow up a scalar-MSE loss": the true value is
 * never rounded to its nearest bin, its probability mass is split between
 * the two bins straddling it (here `target ≈ 2.35` between bins `2` and
 * `3`), and the network learns a full softmax over every bin instead of
 * regressing one scalar. */
export function TwoHotBinsChart() {
  const target = 2.35
  const lower = Math.floor(target)
  const frac = target - lower
  const bins = Array.from({ length: 11 }, (_, i) => i - 5)
  const data = bins.map((b) => ({
    bin: b,
    weight: b === lower ? Number((1 - frac).toFixed(2)) : b === lower + 1 ? Number(frac.toFixed(2)) : 0,
  }))
  return (
    <div className="h-40 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 4, right: 12, left: -18, bottom: 14 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
          <XAxis
            dataKey="bin" stroke={CHART_AXIS} fontSize={10} tickLine={false}
            label={{ value: `корзина h(x), target = h(${target})`, position: 'insideBottom', offset: -6, fontSize: 10, fill: CHART_AXIS }}
          />
          <YAxis stroke={CHART_AXIS} fontSize={10} domain={[0, 1]} tickLine={false} />
          <RTooltip contentStyle={CHART_TOOLTIP} formatter={(v: number) => v.toFixed(2)} />
          <Bar dataKey="weight" name="two-hot вес корзины" fill={CHART_LINE_1} radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/** Teacher-forced unroll vs closed-loop overshooting — the extra training
 * signal ResearchImZero adds on top of UniZero. Solid tokens are real
 * observations from replay; dashed ẑ tokens are the model's own latent
 * predictions, fed back in so the search-time imagination path is trained
 * rather than only the teacher-forced one. */
export function ClosedLoopDiagram() {
  const real = ['obs₀', 'act₀', 'obs₁', 'act₁', 'obs₂']
  const imagined = ['obs₀', 'act₀', 'ẑ₁', 'act₁', 'ẑ₂']
  return (
    <div className="space-y-3 py-1">
      <div>
        <div className="mb-1 text-[10px] font-medium text-muted-foreground">Teacher forcing — настоящие токены из replay</div>
        <div className="flex flex-wrap items-center gap-1.5">
          {real.map((label, i) => (
            <Fragment key={`tf-${label}-${i}`}>
              {i > 0 && <ArrowRight className="h-3 w-3 text-muted-foreground/50" />}
              <div className={cn(
                'rounded-md border px-2 py-1 font-mono text-[10px]',
                label.startsWith('act') ? 'border-amber-500/40 bg-amber-500/10' : 'border-primary/40 bg-primary/10',
              )}>
                {label}
              </div>
            </Fragment>
          ))}
        </div>
      </div>
      <div>
        <div className="mb-1 text-[10px] font-medium text-muted-foreground">Closed-loop — после act₀ дальше идут свои ẑ, как при поиске</div>
        <div className="flex flex-wrap items-center gap-1.5">
          {imagined.map((label, i) => (
            <Fragment key={`cl-${label}-${i}`}>
              {i > 0 && <ArrowRight className="h-3 w-3 text-muted-foreground/50" />}
              <div className={cn(
                'rounded-md border px-2 py-1 font-mono text-[10px]',
                label.startsWith('ẑ')
                  ? 'border-dashed border-emerald-500/50 bg-emerald-500/10'
                  : label.startsWith('act')
                    ? 'border-amber-500/40 bg-amber-500/10'
                    : 'border-primary/40 bg-primary/10',
              )}>
                {label}
              </div>
            </Fragment>
          ))}
        </div>
      </div>
      <p className="text-[10px] text-muted-foreground">
        Consistency на teacher-forced пути учит «угадать следующий obs, когда он уже есть во входе».
        Closed-loop заставляет ту же голову угадывать его, когда следующего настоящего токена нет —
        ровно то, что делает Gumbel-поиск, раскрывая узел вглубь.
      </p>
    </div>
  )
}

/** LatentImZero's one-tree budget checkpoints. Sequential Halving is
 * sampled once for the max budget (32); 0/4/16 are snapshots of that same
 * tree, not four independent searches. VoC decides CONTINUE vs STOP. */
export function VocBudgetDiagram() {
  const stages: { b: string; title: string; detail: string; x: number }[] = [
    { b: '0', title: 'актор', detail: 'π_θ, без поиска', x: 8 },
    { b: '4', title: 'small', detail: 'мало expansion', x: 90 },
    { b: '16', title: 'medium', detail: 'половина дерева', x: 172 },
    { b: '32', title: 'max', detail: 'полный бюджет', x: 254 },
  ]
  return (
    <svg viewBox="0 0 336 132" className="h-36 w-full max-w-xl">
      <text x={168} y={12} textAnchor="middle" className="fill-muted-foreground text-[9px]">
        один Gumbel-планировщик, четыре чекпоинта того же дерева
      </text>
      {stages.map((s, i) => (
        <g key={s.b}>
          {i < stages.length - 1 && (
            <>
              <line x1={s.x + 56} y1={48} x2={stages[i + 1].x} y2={48} stroke="oklch(0.7 0.15 260)" strokeWidth={1.4} />
              <text x={(s.x + 56 + stages[i + 1].x) / 2} y={42} textAnchor="middle" className="fill-primary text-[7.5px]">CONTINUE</text>
            </>
          )}
          <rect
            x={s.x} y={28} width={72} height={40} rx={6}
            className={i === 0 ? 'fill-background' : 'fill-primary/10'}
            stroke={i === stages.length - 1 ? 'oklch(0.75 0.18 150)' : 'oklch(0.7 0.15 260)'}
            strokeWidth={i === stages.length - 1 ? 2 : 1.3}
          />
          <text x={s.x + 36} y={44} textAnchor="middle" className="fill-foreground text-[10px] font-semibold">B = {s.b}</text>
          <text x={s.x + 36} y={57} textAnchor="middle" className="fill-muted-foreground text-[8px]">{s.title}</text>
          <text x={s.x + 36} y={82} textAnchor="middle" className="fill-muted-foreground text-[7.5px]">{s.detail}</text>
        </g>
      ))}
      <text x={168} y={108} textAnchor="middle" className="fill-amber-500 text-[8px]">
        VoC: P(CONTINUE) &lt; 0.5 → STOP на текущем чекпоинте · safety floor не даёт откатиться ниже открытой стадии
      </text>
      <text x={168} y={124} textAnchor="middle" className="fill-muted-foreground text-[8px]">
        shadow-старт: всегда executed = 32, пока VoC не докажет, что ранняя остановка безопасна
      </text>
    </svg>
  )
}

/** Supervised learning vs RL — the first distinction a newcomer needs. */
export function SupervisedVsRlDiagram() {
  const cols = [
    { title: 'Обучение с учителем', detail: 'датасет (x, y*) заранее', signal: 'правильный ответ y*' },
    { title: 'Обучение с подкреплением', detail: 'ответа нет, есть только r', signal: 'скаляр награды, часто позже' },
  ]
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {cols.map((col) => (
        <div key={col.title} className="rounded-xl border border-border/70 bg-background/60 p-3">
          <div className="text-[12px] font-semibold text-foreground">{col.title}</div>
          <div className="mt-1 text-[11px] text-muted-foreground">{col.detail}</div>
          <div className="mt-2 rounded-md bg-muted/50 px-2 py-1.5 font-mono text-[10px] text-foreground/80">{col.signal}</div>
        </div>
      ))}
    </div>
  )
}

/** Discounted return as a weighted sum of future rewards. */
export function ReturnTimelineDiagram() {
  const steps = [
    { r: 'r₁ = 1', w: 'γ⁰ = 1' },
    { r: 'r₂ = 1', w: 'γ¹ = 0.99' },
    { r: 'r₃ = 1', w: 'γ² ≈ 0.98' },
    { r: '…', w: 'γᵗ' },
    { r: 'r₅₀₀', w: 'γ⁴⁹⁹ ≈ 0.007' },
  ]
  return (
    <div className="space-y-2 py-1">
      <div className="flex flex-wrap items-center gap-1.5">
        {steps.map((step, i) => (
          <Fragment key={step.w}>
            {i > 0 && <span className="text-[11px] text-muted-foreground/60">+</span>}
            <div className="rounded-md border border-border/70 bg-background/60 px-2 py-1.5 text-center">
              <div className="font-mono text-[11px] text-foreground">{step.r}</div>
              <div className="text-[9px] text-muted-foreground">× {step.w}</div>
            </div>
          </Fragment>
        ))}
      </div>
      <p className="text-[10px] text-muted-foreground">
        Для CartPole каждый шаг даёт +1. При γ=0.99 пятисотый шаг почти ничего не стоит в возврате —
        агент всё равно может его «хотеть», потому что без него эпизод уже давно упал бы.
      </p>
    </div>
  )
}

/** One-step Bellman backup: value now = reward + discounted value later. */
export function BellmanBackupDiagram() {
  return (
    <svg viewBox="0 0 336 128" className="h-32 w-full max-w-xl">
      <rect x={8} y={36} width={64} height={40} rx={6} className="fill-primary/10" stroke="oklch(0.7 0.15 260)" strokeWidth={1.4} />
      <text x={40} y={54} textAnchor="middle" className="fill-foreground text-[11px] font-semibold">s</text>
      <text x={40} y={68} textAnchor="middle" className="fill-muted-foreground text-[8px]">V(s), Q(s,a)</text>
      <line x1={72} y1={56} x2={118} y2={56} stroke="oklch(0.45 0 0)" strokeWidth={1.4} />
      <text x={95} y={48} textAnchor="middle" className="fill-muted-foreground text-[8px]">a</text>
      <rect x={122} y={18} width={72} height={32} rx={6} className="fill-amber-500/10" stroke="oklch(0.75 0.15 70)" strokeWidth={1.3} />
      <text x={158} y={38} textAnchor="middle" className="fill-foreground text-[10px]">награда r</text>
      <rect x={122} y={62} width={72} height={32} rx={6} className="fill-background" stroke="oklch(0.45 0 0)" strokeWidth={1.3} />
      <text x={158} y={82} textAnchor="middle" className="fill-foreground text-[10px]">s′</text>
      <line x1={194} y1={78} x2={236} y2={78} stroke="oklch(0.45 0 0)" strokeWidth={1.4} />
      <rect x={240} y={58} width={88} height={40} rx={6} className="fill-emerald-500/10" stroke="oklch(0.75 0.18 150)" strokeWidth={1.4} />
      <text x={284} y={76} textAnchor="middle" className="fill-foreground text-[10px] font-semibold">γ · V(s′)</text>
      <text x={284} y={90} textAnchor="middle" className="fill-muted-foreground text-[8px]">или γ max Q(s′,·)</text>
      <text x={168} y={118} textAnchor="middle" className="fill-muted-foreground text-[9px]">
        backup: оценка «сейчас» подтягивается к r + оценка «потом»
      </text>
    </svg>
  )
}

/** Three bandit arms — visit counts vs mean reward, the explore/exploit cartoon. */
export function BanditArmsDiagram() {
  const arms = [
    { x: 24, n: 40, mu: '0.2', label: 'A', dim: true },
    { x: 132, n: 8, mu: '?', label: 'B', dim: false },
    { x: 240, n: 31, mu: '0.7', label: 'C', best: true },
  ]
  return (
    <svg viewBox="0 0 336 118" className="h-32 w-full max-w-xl">
      <text x={168} y={14} textAnchor="middle" className="fill-muted-foreground text-[9px]">
        три автомата: среднее известно только у тех, что уже крутили
      </text>
      {arms.map((arm) => (
        <g key={arm.label} opacity={arm.dim ? 0.55 : 1}>
          <rect
            x={arm.x} y={28} width={72} height={52} rx={6}
            className={arm.best ? 'fill-emerald-500/15' : 'fill-background'}
            stroke={arm.best ? 'oklch(0.75 0.18 150)' : 'oklch(0.7 0.15 260)'}
            strokeWidth={arm.best ? 2 : 1.3}
          />
          <text x={arm.x + 36} y={48} textAnchor="middle" className="fill-foreground text-[12px] font-semibold">рука {arm.label}</text>
          <text x={arm.x + 36} y={64} textAnchor="middle" className="fill-muted-foreground text-[8px]">N={arm.n} · μ={arm.mu}</text>
        </g>
      ))}
      <text x={168} y={102} textAnchor="middle" className="fill-muted-foreground text-[8px]">
        жадно брать C — упускаете, что B (8 попыток) может быть лучше. ε-greedy иногда берёт случайную руку.
      </text>
    </svg>
  )
}

/** Monte-Carlo waits for the episode; TD bootstraps after one step. */
export function TdMcCompareDiagram() {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <div className="rounded-xl border border-border/70 p-3">
        <div className="text-[11px] font-semibold text-foreground">Monte Carlo</div>
        <div className="mt-2 flex flex-wrap items-center gap-1 text-[10px]">
          {['s', 'a', 'r', 's′', 'a′', '…', 'конец'].map((tok) => (
            <span key={tok} className="rounded border border-border/60 px-1.5 py-0.5 font-mono">{tok}</span>
          ))}
        </div>
        <p className="mt-2 text-[10px] text-muted-foreground">Обновление только когда эпизод закончился: цель = полный возврат G. Несмещённо, очень шумно.</p>
      </div>
      <div className="rounded-xl border border-primary/30 bg-primary/5 p-3">
        <div className="text-[11px] font-semibold text-foreground">TD(0) / Q-learning</div>
        <div className="mt-2 flex flex-wrap items-center gap-1 text-[10px]">
          {['s', 'a', 'r', 's′'].map((tok) => (
            <span key={tok} className="rounded border border-primary/40 px-1.5 py-0.5 font-mono">{tok}</span>
          ))}
          <span className="text-muted-foreground">+ bootstrap V(s′) или max Q</span>
        </div>
        <p className="mt-2 text-[10px] text-muted-foreground">Обновление после одного перехода. Смещённо (опирается на текущую оценку), зато можно учиться онлайн.</p>
      </div>
    </div>
  )
}

/** Designer graph as three named nodes — matches Experiment Designer. */
export function DesignerPipelineDiagram() {
  const nodes = [
    { title: 'Среда', detail: 'CartPole-v1' },
    { title: 'Алгоритм', detail: 'PPO' },
    { title: 'Training', detail: '50 000 шагов' },
  ]
  return (
    <div className="flex flex-wrap items-center gap-2 py-1">
      {nodes.map((node, i) => (
        <Fragment key={node.title}>
          {i > 0 && <ArrowRight className="h-4 w-4 text-muted-foreground/50" />}
          <div className="min-w-[7.5rem] rounded-lg border border-primary/30 bg-primary/5 px-3 py-2 text-center">
            <div className="text-[12px] font-semibold text-foreground">{node.title}</div>
            <div className="text-[10px] text-muted-foreground">{node.detail}</div>
          </div>
        </Fragment>
      ))}
    </div>
  )
}

/** Worked numerical example — same callout chrome, but for “посчитаем руками”. */
export function WorkedExample({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="space-y-2.5 rounded-lg border border-primary/25 bg-primary/5 px-3.5 py-3">
      <div className="text-xs font-semibold text-foreground">{title}</div>
      <div className="space-y-2.5 text-[13px] leading-relaxed text-foreground/85">{children}</div>
    </div>
  )
}

export function MiniTable({
  headers,
  rows,
  caption,
}: {
  headers: string[]
  rows: string[][]
  caption?: string
}) {
  return (
    <div className="space-y-1">
      <div className="overflow-x-auto rounded-lg border border-border/70">
        <table className="w-full text-left text-[12px]">
          <thead className="bg-muted/50 text-[11px] text-muted-foreground">
            <tr>
              {headers.map((header) => (
                <th key={header} className="px-2.5 py-1.5 font-medium">{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="border-t border-border/50">
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex} className="px-2.5 py-1.5 font-mono text-foreground/90">{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {caption && <p className="text-[11px] text-muted-foreground">{caption}</p>}
    </div>
  )
}

/** Tiny deterministic chain used in Bellman / DP / TD numerical examples. */
export function ChainMdpDiagram() {
  const cells = [
    { title: 'A', detail: 'старт' },
    { title: 'B', detail: 'середина' },
    { title: 'цель', detail: 'терминал, V = 0' },
  ]
  return (
    <div className="space-y-2 py-1">
      <div className="flex flex-wrap items-center gap-2">
        {cells.map((cell, i) => (
          <Fragment key={cell.title}>
            {i > 0 && (
              <div className="flex flex-col items-center text-muted-foreground">
                <ArrowRight className="h-4 w-4" />
                <span className="text-[9px]">{i === 1 ? 'Right, r = 0' : 'Right, r = 1'}</span>
              </div>
            )}
            <div className="min-w-[5.5rem] rounded-lg border border-border/70 bg-background/60 px-3 py-2 text-center">
              <div className="text-[12px] font-semibold text-foreground">{cell.title}</div>
              <div className="text-[10px] text-muted-foreground">{cell.detail}</div>
            </div>
          </Fragment>
        ))}
      </div>
      <p className="text-[10px] text-muted-foreground">
        Одна действие — Right. γ = 0,9. Из цели шагов нет. Это не FrozenLake: лёд не скользит, P детерминирована.
      </p>
    </div>
  )
}

import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Loader2, Boxes, CircleCheck, CircleX } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useEnvironments } from '@/api/hooks'
import { api } from '@/api/client'
import type { EnvSpec } from '@/api/types'

const CATEGORY_LABELS: Record<string, string> = {
  classic_control: 'Classic Control',
  toy_text: 'Toy Text',
  box2d: 'Box2D / Physics',
  mujoco: 'MuJoCo / Robotics',
  atari: 'Atari 2600',
  minigrid: 'MiniGrid (навигация/память)',
  highway_env: 'Highway-env (вождение)',
  nethack: 'NetHack / MiniHack (dungeon crawler)',
  robotics: 'Robotics (goal-conditioned навигация)',
  industrial: 'Industrial (планирование/логистика)',
  trading: 'Trading (позиции и портфели)',
  pomdp: 'POMDP (нужна память)',
  board_game: 'Настольная игра',
}

const EXTRA_HINT: Record<string, string> = {
  box2d: 'pip install "gymnasium[box2d]"',
  mujoco: 'pip install "gymnasium[mujoco]"',
  atari: 'pip install "gymnasium[atari]" && ale-import-roms',
  minigrid: 'pip install minigrid',
  highway_env: 'pip install highway-env',
  nle: 'pip install nle (только Linux, готовые wheel-ы)',
  minihack: 'pip install minihack (только Linux, готовые wheel-ы)',
  gymnasium_robotics: 'pip install gymnasium-robotics',
}

const CATEGORY_ORDER = [
  'classic_control', 'toy_text', 'box2d', 'mujoco', 'atari', 'minigrid', 'highway_env', 'nethack', 'robotics',
  'industrial', 'trading', 'pomdp', 'board_game',
]

function EnvPreview({ env }: { env: EnvSpec }) {
  const rootRef = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  const [thumb, setThumb] = useState<string>()
  const [full, setFull] = useState<string>()
  const [hovered, setHovered] = useState(false)
  const [thumbFailed, setThumbFailed] = useState(false)
  const [thumbLoaded, setThumbLoaded] = useState(false)
  const [fullLoaded, setFullLoaded] = useState(false)

  useEffect(() => {
    const node = rootRef.current
    if (!node) return
    const io = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible(true) },
      { rootMargin: '200px' },
    )
    io.observe(node)
    return () => io.disconnect()
  }, [])

  useEffect(() => {
    if (!visible) return
    let cancelled = false
    setThumbFailed(false)

    const thumbPath = env.preview_thumb_url ?? env.preview_url
    if (!thumbPath) return

    api.resolveUrl(thumbPath).then((url) => {
      if (!cancelled) setThumb(url)
    })
    return () => { cancelled = true }
  }, [visible, env.id, env.preview_thumb_url, env.preview_url])

  useEffect(() => {
    if (!hovered || full || !env.preview_url || env.preview_url === env.preview_thumb_url) return
    let cancelled = false
    api.resolveUrl(env.preview_url).then((url) => {
      if (!cancelled) setFull(url)
    })
    return () => { cancelled = true }
  }, [hovered, full, env.preview_url, env.preview_thumb_url])

  return (
    <div
      ref={rootRef}
      className="relative aspect-video w-full overflow-hidden rounded-md bg-muted/40"
      onMouseEnter={() => setHovered(true)}
    >
      {!thumb && !thumbFailed && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Boxes className="h-8 w-8 text-muted-foreground/30" />
        </div>
      )}
      {thumb && !thumbFailed && (
        <img
          src={thumb}
          alt={env.name}
          decoding="async"
          // object-contain (not cover): source GIFs come from Farama's docs
          // in wildly different aspect ratios (0.76 for Atari, 3.0 for Cliff
          // Walking, ...) — cover would crop them into our fixed 16:9 card,
          // and for *animated* previews (e.g. Blackjack dealing cards near
          // the frame edges) that crop made content look like it was
          // randomly appearing/disappearing as the animation played.
          className={`absolute inset-0 h-full w-full object-contain transition-opacity duration-300 ${thumbLoaded ? 'opacity-100' : 'opacity-0'}`}
          onLoad={() => setThumbLoaded(true)}
          onError={() => setThumbFailed(true)}
        />
      )}
      {full && (
        <img
          src={full}
          alt=""
          decoding="async"
          // Fades in over the (still-mounted) thumb instead of popping in the
          // instant the GIF finishes downloading, so the swap reads as a
          // smooth upgrade rather than a flicker.
          className={`absolute inset-0 h-full w-full object-contain transition-opacity duration-300 ${fullLoaded ? 'opacity-100' : 'opacity-0'}`}
          onLoad={() => setFullLoaded(true)}
        />
      )}
    </div>
  )
}

export function Environments() {
  const navigate = useNavigate()
  const { data, isLoading } = useEnvironments()
  const environments = data?.environments ?? []
  const [categoryFilter, setCategoryFilter] = useState<string>('all')

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  const visible = categoryFilter === 'all'
    ? environments
    : environments.filter((e) => e.category === categoryFilter)

  const groups = visible.reduce<Record<string, typeof environments>>((acc, env) => {
    ;(acc[env.category] ??= []).push(env)
    return acc
  }, {})

  const usedCategories = CATEGORY_ORDER.filter((c) => environments.some((e) => e.category === c))

  return (
    <div className="mx-auto max-w-6xl space-y-8 p-8">
      <div className="space-y-1">
        <h2 className="text-2xl font-semibold">Галерея сред</h2>
        <p className="text-sm text-muted-foreground">
          {environments.length} сред: Classic Control, Toy Text, Box2D, MuJoCo, Atari и настольные игры для AlphaZero.
          Картинки — официальные демо из репозитория Farama Foundation.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button
          size="sm"
          variant={categoryFilter === 'all' ? 'default' : 'outline'}
          onClick={() => setCategoryFilter('all')}
        >
          Все · {environments.length}
        </Button>
        {usedCategories.map((c) => {
          const n = environments.filter((e) => e.category === c).length
          return (
            <Button
              key={c}
              size="sm"
              variant={categoryFilter === c ? 'default' : 'outline'}
              onClick={() => setCategoryFilter(c)}
            >
              {CATEGORY_LABELS[c] ?? c} · {n}
            </Button>
          )
        })}
      </div>

      {CATEGORY_ORDER.filter((c) => groups[c]).map((category) => {
        const envs = groups[category]
        return (
        <div key={category} className="space-y-3">
          <h3 className="text-sm font-medium text-muted-foreground">
            {CATEGORY_LABELS[category] ?? category}
          </h3>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {envs.map((env) => (
              <Card key={env.id} className={!env.available ? 'opacity-60' : ''}>
                <CardHeader className="flex-row items-start justify-between space-y-0">
                  <div className="flex items-center gap-2">
                    <Boxes className="h-4 w-4 text-primary" />
                    <CardTitle className="text-sm">{env.name}</CardTitle>
                  </div>
                  {env.available ? (
                    <CircleCheck className="h-4 w-4 text-success" />
                  ) : (
                    <CircleX className="h-4 w-4 text-muted-foreground" />
                  )}
                </CardHeader>
                <CardContent className="space-y-3">
                  <EnvPreview env={env} />
                  <CardDescription>{env.description}</CardDescription>
                  <div className="flex flex-wrap gap-1">
                    <Badge variant="outline" className="text-[10px]">{env.action_kind}</Badge>
                    {env.compatible_algorithms.map((a) => (
                      <Badge key={a} variant="secondary" className="text-[10px] uppercase">{a}</Badge>
                    ))}
                  </div>
                  {!env.available && env.extra_requirement && (
                    <p className="text-xs text-muted-foreground">
                      Не установлен extra.{' '}
                      <code className="font-mono">{EXTRA_HINT[env.extra_requirement] ?? env.extra_requirement}</code>
                    </p>
                  )}
                  <Button
                    size="sm"
                    variant="outline"
                    className="w-full"
                    disabled={!env.available}
                    onClick={() => navigate(`/designer?env=${encodeURIComponent(env.id)}`)}
                  >
                    Использовать в дизайнере
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
        )
      })}
    </div>
  )
}

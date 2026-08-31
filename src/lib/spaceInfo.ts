import type { SpaceInfo } from '@/api/types'

/** The literal number of scalars the network reads/writes for this space —
 * a `Discrete(n)` observation is one-hot encoded (`n` inputs, see
 * `rl_core/inspect.py::_io_shapes`), a `Discrete(n)` action space has `n`
 * possible choices (`n` output logits), and a `Box` space of any shape is
 * flattened into the product of its dimensions. This is the number users
 * mean by "сколько входов/выходов у среды", as opposed to the space's
 * type/shape notation (`Box[4]`, `Discrete(n=2)`), which is exact but not
 * self-explanatory to non-experts. */
export function spaceSize(space?: SpaceInfo | null): number | null {
  if (!space) return null
  if (space.shape && space.shape.length > 0) {
    return space.shape.reduce((acc, dim) => acc * dim, 1)
  }
  if (space.n != null) return space.n
  return null
}

/** Short human label for what kind of space this is — "непрерывное" (Box)
 * vs "дискретное, N вариантов" (Discrete) — shown alongside the plain
 * count above as supporting detail, not instead of it. */
export function spaceKindLabel(space?: SpaceInfo | null): string {
  if (!space) return '—'
  if (space.n != null) return `дискретное, ${space.n} вариант${pluralSuffix(space.n)}`
  if (space.shape && space.shape.length > 1) return `непрерывное, форма [${space.shape.join('×')}]`
  return 'непрерывное'
}

function pluralSuffix(n: number): string {
  const mod10 = n % 10
  const mod100 = n % 100
  if (mod100 >= 11 && mod100 <= 14) return 'ов'
  if (mod10 === 1) return ''
  if (mod10 >= 2 && mod10 <= 4) return 'а'
  return 'ов'
}

/** Exact type/shape notation (`Box[4]`, `Discrete(n=2)`) — the precise
 * technical detail shown in smaller/muted text next to the plain count
 * from `spaceSize`. */
export function formatSpace(space?: SpaceInfo | null): string {
  if (!space) return '—'
  if (space.n !== undefined) return `${space.type}(n=${space.n})`
  if (space.shape && space.shape.length > 0) return `${space.type}[${space.shape.join('×')}]`
  return space.type
}

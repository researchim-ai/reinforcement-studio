/** Smoke coverage for `lessons.tsx` — every lesson's `content` is plain
 * JSX built once at module load (no hooks/data-fetching of its own, see
 * `AlgorithmDiagram`'s own props-only API), so the main risk is a typo in
 * hand-written SVG/diagram code (`visuals.tsx`) throwing at render time.
 * Rendering every lesson catches that immediately, instead of only
 * discovering it interactively in the Academy page. */
import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { LESSONS } from '@/components/academy/lessons'

describe('Academy lessons', () => {
  it.each(LESSONS.map((l) => [l.id, l] as const))('renders %s without throwing', (_id, lesson) => {
    expect(() => render(<>{lesson.content}</>)).not.toThrow()
  })

  it('efficientzero renders its new training-unroll/Gumbel-halving/two-hot illustrations', () => {
    const lesson = LESSONS.find((l) => l.id === 'efficientzero')
    expect(lesson).toBeTruthy()
    const { container } = render(<>{lesson!.content}</>)
    // `LatentChainDiagram` (CSS/grid, no svg) + `GumbelHalvingDiagram` +
    // `TwoHotBinsChart` (svg via recharts) + `AlgorithmDiagram`'s own loop
    // chips - at least a few <svg> elements should be present.
    expect(container.querySelectorAll('svg').length).toBeGreaterThan(0)
    expect(container.textContent).toContain('env_action = a₂')
    expect(container.textContent).toContain('только цель consistency')
  })

  it('unizero renders its new token-attention illustration', () => {
    const lesson = LESSONS.find((l) => l.id === 'unizero')
    expect(lesson).toBeTruthy()
    const { container } = render(<>{lesson!.content}</>)
    expect(container.querySelectorAll('svg').length).toBeGreaterThan(0)
    expect(container.textContent).toContain('policy/value здесь')
    expect(container.textContent).toContain('reward здесь')
  })
})

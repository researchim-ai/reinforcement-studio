/** Smoke coverage for `lessons.tsx` — every lesson's `content` is plain
 * JSX built once at module load (no hooks/data-fetching of its own, see
 * `AlgorithmDiagram`'s own props-only API), so the main risk is a typo in
 * hand-written SVG/diagram code (`visuals.tsx`) throwing at render time.
 * Rendering every lesson catches that immediately, instead of only
 * discovering it interactively in the Academy page. */
import { render } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { LESSONS } from '@/components/academy/lessons'

function renderLesson(content: LessonContent) {
  return render(<MemoryRouter>{content}</MemoryRouter>)
}

type LessonContent = (typeof LESSONS)[number]['content']

describe('Academy lessons', () => {
  it.each(LESSONS.map((l) => [l.id, l] as const))('renders %s without throwing', (_id, lesson) => {
    expect(() => renderLesson(lesson.content)).not.toThrow()
  })

  it('covers every built-in algorithm with a dedicated or shared lesson', () => {
    const ids = new Set(LESSONS.map((lesson) => lesson.id))
    for (const id of [
      'dqn', 'rainbow_dqn', 'a2c', 'ppo', 'sac', 'ddpg', 'td3', 'es', 'alphazero',
      'dreamer', 'mbpo_pets', 'world_models_ha', 'efficientzero', 'unizero',
      'researchimzero', 'latentimzero', 'ippo', 'qmix',
    ]) {
      expect(ids.has(id)).toBe(true)
    }
  })

  it('teaches foundations before algorithms: return, Bellman, TD, REINFORCE, first run', () => {
    const ids = LESSONS.map((lesson) => lesson.id)
    expect(new Set(ids).size).toBe(ids.length)
    expect(ids.length).toBeGreaterThanOrEqual(40)
    for (const id of [
      'fundamentals', 'what_is_rl', 'first_experiment',
      'agent_env', 'rewards_returns', 'policies', 'value_functions', 'explore_exploit', 'on_off_policy',
      'dynamic_programming', 'monte_carlo', 'td_learning', 'function_approx',
      'reinforce', 'reading_curves', 'hyperparams', 'failure_modes',
    ]) {
      expect(ids).toContain(id)
    }
    expect(ids.indexOf('td_learning')).toBeLessThan(ids.indexOf('dqn'))
    expect(ids.indexOf('reinforce')).toBeLessThan(ids.indexOf('a2c'))
    expect(ids.indexOf('first_experiment')).toBeLessThan(ids.indexOf('dqn'))
  })

  it('first experiment and Bellman lessons contain the practical vocabulary', () => {
    const first = LESSONS.find((l) => l.id === 'first_experiment')
    const bellman = LESSONS.find((l) => l.id === 'value_functions')
    const td = LESSONS.find((l) => l.id === 'td_learning')
    expect(renderLesson(first!.content).container.textContent).toContain('CartPole')
    expect(renderLesson(first!.content).container.textContent).toContain('Дизайнер')
    expect(renderLesson(bellman!.content).container.textContent).toContain('Беллман')
    expect(renderLesson(td!.content).container.textContent).toContain('Q-learning')
  })

  it('foundation and tabular lessons include worked numerical examples', () => {
    const rewards = LESSONS.find((l) => l.id === 'rewards_returns')
    const bellman = LESSONS.find((l) => l.id === 'value_functions')
    const td = LESSONS.find((l) => l.id === 'td_learning')
    const dp = LESSONS.find((l) => l.id === 'dynamic_programming')
    expect(renderLesson(rewards!.content).container.textContent).toContain('5,23')
    expect(renderLesson(bellman!.content).container.textContent).toContain('0,9 · 1')
    expect(renderLesson(td!.content).container.textContent).toContain('Q(A,→) = 0,9')
    expect(renderLesson(dp!.content).container.textContent).toContain('синхронное обновление')
  })

  it('efficientzero renders its new training-unroll/Gumbel-halving/two-hot illustrations', () => {
    const lesson = LESSONS.find((l) => l.id === 'efficientzero')
    expect(lesson).toBeTruthy()
    const { container } = renderLesson(lesson!.content)
    expect(container.querySelectorAll('svg').length).toBeGreaterThan(0)
    expect(container.textContent).toContain('env_action = a₂')
    expect(container.textContent).toContain('только цель consistency')
  })

  it('unizero renders its new token-attention illustration', () => {
    const lesson = LESSONS.find((l) => l.id === 'unizero')
    expect(lesson).toBeTruthy()
    const { container } = renderLesson(lesson!.content)
    expect(container.querySelectorAll('svg').length).toBeGreaterThan(0)
    expect(container.textContent).toContain('policy/value здесь')
    expect(container.textContent).toContain('reward здесь')
  })

  it('researchimzero explains closed-loop overshooting against UniZero', () => {
    const lesson = LESSONS.find((l) => l.id === 'researchimzero')
    expect(lesson).toBeTruthy()
    const { container } = renderLesson(lesson!.content)
    expect(container.textContent).toContain('Closed-loop')
    expect(container.textContent).toContain('Teacher forcing')
    expect(container.textContent).toContain('max(TD, search)')
  })

  it('latentimzero explains VoC checkpoints and sidecar isolation', () => {
    const lesson = LESSONS.find((l) => l.id === 'latentimzero')
    expect(lesson).toBeTruthy()
    const { container } = renderLesson(lesson!.content)
    expect(container.textContent).toContain('VoC')
    expect(container.textContent).toContain('force_research_mode')
    expect(container.textContent).not.toContain('Prior imagination')
    expect(container.textContent).toContain('B = 0')
    expect(container.textContent).toContain('B = 32')
  })
})

/** Short path through the Academy: enough to run CartPole, read Bellman,
 * understand Q-learning/PPO, then debug a real training curve.
 * Algorithm encyclopedia (SAC, Zero, MARL, …) stays on the full course. */
export const BEGINNER_TRACK = [
  'fundamentals',
  'what_is_rl',
  'first_experiment',
  'agent_env',
  'rewards_returns',
  'policies',
  'value_functions',
  'explore_exploit',
  'on_off_policy',
  'dynamic_programming',
  'monte_carlo',
  'td_learning',
  'function_approx',
  'dqn',
  'reinforce',
  'a2c',
  'ppo',
  'reading_curves',
  'hyperparams',
  'failure_modes',
  'cheatsheet',
] as const

export const BEGINNER_SET = new Set<string>(BEGINNER_TRACK)

export function isBeginnerLesson(id: string): boolean {
  return BEGINNER_SET.has(id)
}

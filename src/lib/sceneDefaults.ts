import type { SceneAgentGroup, SceneSpec } from '@/api/types'

let _id = 0
export function newId(prefix: string): string {
  _id += 1
  return `${prefix}_${_id}`
}

const GROUP_PALETTE = ['#3b82f6', '#ef4444', '#eab308', '#22c55e', '#a855f7', '#f97316']

/** A fresh agent group for the "+ команда" button — `index` picks a
 * distinct default team id/color so a newly added group never silently
 * overlaps an existing one (both visually and in `team_ids`/reward
 * bookkeeping — see `_agent_groups` in rl_core/envs/scene_env.py). */
export function newAgentGroup(index: number): SceneAgentGroup {
  return {
    id: newId('group'),
    count: 2,
    team: String.fromCharCode(97 + (index % 26)), // 'a', 'b', 'c', ...
    role: 'agent',
    spawn: { center: [index % 2 === 0 ? -5 : 5, 0, 0], radius: 3 },
    body_radius: 0.4,
    movement: { type: 'discrete4', speed: 0.5 },
    sensors: { type: 'nearest_k', k: 4, range: 10 },
    shape: 'capsule',
    material: { pattern: 'solid', color: GROUP_PALETTE[index % GROUP_PALETTE.length] },
  }
}

export function defaultSceneSpec(name = 'Новая сцена'): SceneSpec {
  return {
    name,
    description: '',
    world: { width: 20, depth: 20, wall_height: 2 },
    objects: [
      { id: 'wall_n', type: 'wall', position: [0, 1, -10], size: [20, 2, 1], shape: 'box', material: { pattern: 'brick', color: '#9ca3af', color2: '#4b5563' } },
      { id: 'wall_s', type: 'wall', position: [0, 1, 10], size: [20, 2, 1], shape: 'box', material: { pattern: 'brick', color: '#9ca3af', color2: '#4b5563' } },
      { id: 'wall_w', type: 'wall', position: [-10, 1, 0], size: [1, 2, 20], shape: 'box', material: { pattern: 'brick', color: '#9ca3af', color2: '#4b5563' } },
      { id: 'wall_e', type: 'wall', position: [10, 1, 0], size: [1, 2, 20], shape: 'box', material: { pattern: 'brick', color: '#9ca3af', color2: '#4b5563' } },
      { id: 'pillar', type: 'prop', position: [6, 1, -4], size: [1, 2, 1], shape: 'cylinder', material: { pattern: 'stripes', color: '#c084fc', color2: '#4c1d95' } },
    ],
    items: [
      {
        id: 'coin1', type: 'reward', position: [4, 0, 4], radius: 0.5, reward: 1, respawn: true, cooldown_steps: 40,
        shape: 'crystal', material: { pattern: 'dots', color: '#22c55e', color2: '#14532d', emissive: true },
      },
      {
        id: 'coin2', type: 'reward', position: [-4, 0, -3], radius: 0.5, reward: 1, respawn: true, cooldown_steps: 40,
        shape: 'pyramid', material: { pattern: 'checker', color: '#eab308', color2: '#713f12', emissive: true },
      },
      {
        id: 'hazard1', type: 'hazard', position: [0, 0, 6], radius: 0.6, reward: -1, terminate: true,
        shape: 'sphere', material: { pattern: 'noise', color: '#ef4444', color2: '#7f1d1d', emissive: true },
      },
    ],
    agents: [
      {
        id: 'team_a',
        count: 2,
        team: 'a',
        spawn: { center: [0, 0, 0], radius: 4 },
        body_radius: 0.4,
        movement: { type: 'discrete4', speed: 0.5 },
        sensors: { type: 'nearest_k', k: 4, range: 10 },
        shape: 'capsule',
        material: { pattern: 'solid', color: '#3b82f6' },
      },
    ],
    episode: { max_steps: 500 },
  }
}

export function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_|_$/g, '')
    .slice(0, 64) || 'scene'
}

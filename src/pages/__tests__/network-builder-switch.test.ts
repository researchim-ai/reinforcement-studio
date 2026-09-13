import { describe, expect, it, vi } from 'vitest'

import { defaultCompositeSpec, defaultSpecForFamily, isFlatSpec } from '@/lib/networkBuilder'
import { buildGraph } from '@/pages/NetworkBuilder'

const handlers = {
  updateTrunkLayer: vi.fn(),
  removeTrunkLayer: vi.fn(),
  moveTrunkLayer: vi.fn(),
  addTrunkLayer: vi.fn(),
  updateHeadLayer: vi.fn(),
  removeHeadLayer: vi.fn(),
  moveHeadLayer: vi.fn(),
  addHeadLayer: vi.fn(),
}

describe('network builder graph', () => {
  it('does not crash when a stale Zero-family preview is applied to a PPO spec', () => {
    const spec = defaultSpecForFamily('actor_critic')
    expect(() => buildGraph(spec, {
      ok: true,
      error: null,
      input_shape: [4],
      output_shape: [2],
      components: [{ name: 'transformer', type: 'T', role: 'online', params: 1, trainable_params: 1, layers: [] }],
      total_params: 1,
      trainable_params: 1,
      fixed_outputs: { policy: [2] },
    } as never, handlers)).not.toThrow()
  })

  it('does not crash when the spec itself is missing trunk/heads', () => {
    expect(isFlatSpec(defaultCompositeSpec('latentimzero'))).toBe(false)
    expect(() => buildGraph(defaultCompositeSpec('latentimzero') as never, undefined, handlers)).not.toThrow()
  })
})

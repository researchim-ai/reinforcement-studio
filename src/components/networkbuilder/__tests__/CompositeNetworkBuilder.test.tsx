import { fireEvent, render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { CompositeNetworkBuilder } from '@/components/networkbuilder/CompositeNetworkBuilder'
import {
  defaultCompositeSpec,
  isCompositeFamily,
  isCompositeSpec,
  networkFamilyFor,
} from '@/lib/networkBuilder'

describe('typed composite network builder', () => {
  it('maps each Zero algorithm to its own compatible family', () => {
    expect(networkFamilyFor('efficientzero', 'gym')).toBe('efficientzero')
    expect(networkFamilyFor('unizero', 'gym')).toBe('unizero')
    expect(networkFamilyFor('researchimzero', 'gym')).toBe('researchimzero')
    expect(isCompositeFamily('researchimzero')).toBe(true)
  })

  it('creates complete versioned defaults without exposing algorithm outputs', () => {
    const efficientzero = defaultCompositeSpec('efficientzero')
    const unizero = defaultCompositeSpec('unizero')
    const researchimzero = defaultCompositeSpec('researchimzero')

    expect(isCompositeSpec(efficientzero)).toBe(true)
    expect(Object.keys(efficientzero.components)).toEqual([
      'representation', 'dynamics', 'prediction', 'projector', 'predictor',
    ])
    expect(unizero.components).toHaveProperty('tokenizer')
    expect(unizero.components).not.toHaveProperty('projector')
    expect(researchimzero.components.projector.batch_norm).toBe(true)
    expect(researchimzero.components.predictor.batch_norm).toBe(true)
  })

  it('renders the protected skeleton, editable slots and real preview', () => {
    const onChange = vi.fn()
    const spec = defaultCompositeSpec('researchimzero')
    const { container } = render(
      <CompositeNetworkBuilder
        spec={spec}
        onChange={onChange}
        preview={{
          ok: true,
          error: null,
          input_shape: [4],
          output_shape: [2],
          trunk: [],
          trunk_error: null,
          trunk_error_index: null,
          heads: {},
          total_params: 100,
          trainable_params: 50,
          fixed_outputs: { policy: [2], value: [601], next_token: ['embed_dim'] },
          components: [{
            name: 'transformer',
            type: '_CausalTransformer',
            role: 'online',
            params: 100,
            trainable_params: 50,
            layers: [{
              path: 'blocks.0.attn.query',
              type: 'Linear',
              detail: '128→128',
              params: 10,
              trainable_params: 10,
            }],
          }],
        }}
      />,
    )

    expect(container.textContent).toContain('Фиксированный корректный каркас')
    expect(container.textContent).toContain('Causal Transformer + KV-cache')
    expect(container.textContent).toContain('next_token → [embed_dim] · auto')
    expect(container.textContent).toContain('blocks.0.attn.query')

    const embedInput = Array.from(container.querySelectorAll('input')).find((input) => input.value === '128')
    expect(embedInput).toBeTruthy()
    fireEvent.change(embedInput!, { target: { value: '64' } })
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      dimensions: expect.objectContaining({ embed_dim: 64 }),
    }))
  })
})

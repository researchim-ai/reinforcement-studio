import { fireEvent, render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { CompositeNetworkBuilder, parseHiddenSizesDraft } from '@/components/networkbuilder/CompositeNetworkBuilder'
import { buildCompositeGraph } from '@/lib/compositeGraph'
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
    expect(networkFamilyFor('latentimzero', 'gym')).toBe('latentimzero')
    expect(isCompositeFamily('latentimzero')).toBe(true)
  })

  it('creates complete versioned defaults without exposing algorithm outputs', () => {
    const efficientzero = defaultCompositeSpec('efficientzero')
    const unizero = defaultCompositeSpec('unizero')
    const researchimzero = defaultCompositeSpec('researchimzero')
    const latentimzero = defaultCompositeSpec('latentimzero')

    expect(isCompositeSpec(efficientzero)).toBe(true)
    expect(Object.keys(efficientzero.components)).toEqual([
      'representation', 'dynamics', 'prediction', 'projector', 'predictor',
    ])
    expect(unizero.components).toHaveProperty('tokenizer')
    expect(unizero.components).not.toHaveProperty('projector')
    expect(researchimzero.components.projector.batch_norm).toBe(true)
    expect(researchimzero.components.predictor.batch_norm).toBe(true)
    expect(latentimzero.dimensions.proj_dim).toBe(64)
    expect(latentimzero.components).toHaveProperty('tokenizer')
    expect(latentimzero.components).toHaveProperty('heads')
    expect(latentimzero.components.projector.batch_norm).toBe(true)
    expect(latentimzero.components.predictor.batch_norm).toBe(true)
    expect(latentimzero.components).not.toHaveProperty('stochastic_prior')
  })

  it('builds a connected architecture graph for every Zero family', () => {
    for (const family of ['efficientzero', 'unizero', 'researchimzero', 'latentimzero'] as const) {
      const { nodes, edges } = buildCompositeGraph(defaultCompositeSpec(family), undefined, null, () => {})
      const ids = new Set(nodes.map((node) => node.id))
      expect(nodes.length).toBeGreaterThan(5)
      expect(edges.length).toBeGreaterThan(4)
      for (const edge of edges) {
        expect(ids.has(edge.source)).toBe(true)
        expect(ids.has(edge.target)).toBe(true)
      }
    }
    const latent = buildCompositeGraph(defaultCompositeSpec('latentimzero'), undefined, null, () => {})
    expect(latent.nodes.map((node) => node.id)).toEqual(expect.arrayContaining([
      'tokenizer', 'transformer', 'heads', 'projector', 'predictor', 'uncertainty_probe', 'planner',
    ]))
  })

  it('renders the algorithm graph instead of a layer dump', () => {
    const onChange = vi.fn()
    const spec = defaultCompositeSpec('researchimzero')
    const { container, getByText } = render(
      <div style={{ width: 960, height: 720 }}>
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
        />
      </div>,
    )

    expect(container.textContent).toContain('Каркас алгоритма')
    expect(container.textContent).toContain('Causal Transformer')
    expect(container.textContent).toContain('Энкодер')
    expect(container.textContent).toContain('next_token → [embed_dim]')
    expect(container.textContent).not.toContain('Фиксированный корректный каркас')
    expect(container.textContent).not.toContain('blocks.0.attn.query')

    const embedInput = Array.from(container.querySelectorAll('input')).find((input) => input.value === '128')
    expect(embedInput).toBeTruthy()
    fireEvent.change(embedInput!, { target: { value: '64' } })
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      dimensions: expect.objectContaining({ embed_dim: 64 }),
    }))

    fireEvent.click(getByText('Энкодер'))
    expect(container.textContent).toContain('Входной энкодер')
  })

  it('parses hidden-size drafts without treating a trailing comma as complete', () => {
    expect(parseHiddenSizesDraft('')).toEqual([])
    expect(parseHiddenSizesDraft('128')).toEqual([128])
    expect(parseHiddenSizesDraft('128, 64')).toEqual([128, 64])
    expect(parseHiddenSizesDraft('256,')).toBeNull()
    expect(parseHiddenSizesDraft('256, ')).toBeNull()
    expect(parseHiddenSizesDraft('128,,64')).toBeNull()
    expect(parseHiddenSizesDraft('128, abc')).toBeNull()
    expect(parseHiddenSizesDraft('0')).toBeNull()
  })

  it('keeps a trailing comma in the hidden-sizes field instead of snapping back', () => {
    const onChange = vi.fn()
    const spec = defaultCompositeSpec('researchimzero')
    const { getByText, getByDisplayValue } = render(
      <div style={{ width: 960, height: 720 }}>
        <CompositeNetworkBuilder spec={spec} onChange={onChange} />
      </div>,
    )

    fireEvent.click(getByText('Tokenizer'))
    const hiddenSizes = getByDisplayValue('256') as HTMLInputElement
    fireEvent.focus(hiddenSizes)
    fireEvent.change(hiddenSizes, { target: { value: '256,' } })
    expect(hiddenSizes.value).toBe('256,')
    expect(onChange).not.toHaveBeenCalled()

    fireEvent.change(hiddenSizes, { target: { value: '256, 64' } })
    expect(hiddenSizes.value).toBe('256, 64')
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      components: expect.objectContaining({
        tokenizer: expect.objectContaining({ hidden_sizes: [256, 64] }),
      }),
    }))

    fireEvent.change(hiddenSizes, { target: { value: '256,' } })
    expect(hiddenSizes.value).toBe('256,')
    fireEvent.blur(hiddenSizes)
    expect(hiddenSizes.value).toBe('256')
  })
})

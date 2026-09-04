import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { AlgorithmDiagram } from '@/components/AlgorithmDiagram'

describe('AlgorithmDiagram', () => {
  it('renders every component and every layer of a compound world model', () => {
    const { container } = render(
      <AlgorithmDiagram
        show={{ loop: false, network: true }}
        network={{
          inputShape: [21, 79, 1],
          outputShape: [23],
          totalParams: 200,
          trainableParams: 100,
          components: [
            {
              name: 'transformer',
              type: '_CausalTransformer',
              role: 'online',
              params: 100,
              trainable_params: 100,
              layers: Array.from({ length: 12 }, (_, index) => ({
                path: `blocks.${index}.attn.q_proj`,
                type: 'Linear',
                detail: '128→128',
                params: 10,
                trainable_params: 10,
              })),
            },
            {
              name: 'target_transformer',
              type: '_CausalTransformer',
              role: 'target',
              params: 100,
              trainable_params: 0,
              layers: [{
                path: 'blocks.0.attn.q_proj',
                type: 'Linear',
                detail: '128→128',
                params: 10,
                trainable_params: 0,
              }],
            },
          ],
        }}
      />,
    )

    expect(container.textContent).toContain('Causal Transformer')
    expect(container.textContent).toContain('EMA Causal Transformer')
    expect(container.textContent).toContain('TARGET / EMA')
    expect(container.textContent).toContain('blocks.11.attn.q_proj')
    expect(container.textContent).toContain('Всего параметров:')
    expect(container.textContent).toContain('Обучаемых:')
  })
})

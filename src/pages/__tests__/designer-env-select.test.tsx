import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MemoryRouter, Routes, Route, Link } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { describe, expect, it, vi, beforeAll } from 'vitest'
import { Environments } from '@/pages/Environments'
import { ExperimentDesigner } from '@/pages/ExperimentDesigner'

vi.mock('@/api/client', () => ({
  api: {
    listEnvironments: async () => ({
      environments: [
        {
          id: 'CartPole-v1', name: 'CartPole', category: 'classic_control', description: 'cartpole desc',
          action_kind: 'discrete', compatible_algorithms: ['ppo'], kind: 'gym', available: true,
        },
        {
          id: 'Pendulum-v1', name: 'Pendulum', category: 'classic_control', description: 'pendulum desc',
          action_kind: 'continuous', compatible_algorithms: ['ppo'], kind: 'gym', available: true,
        },
        // A stand-in for a `petting:`/`scene:` multi-team env — same
        // `compatible_algorithms` ordering the real registry produces
        // (base single-policy algos first, `ippo`/`qmix` appended last;
        // see `list_environments()` in rl_core/envs/registry.py), so the
        // Designer can't just default to `compatibleAlgorithms[0]`.
        {
          id: 'petting:simple_tag', name: 'MPE: Simple Tag', category: 'marl', description: 'tag desc',
          action_kind: 'discrete', compatible_algorithms: ['dqn', 'ppo', 'a2c', 'ippo', 'qmix'], kind: 'gym',
          available: true, scene_team_count: 2, scene_agent_count: 5,
        },
      ],
    }),
    listWrappers: async () => ({ wrappers: [] }),
    listAlgorithms: async () => ({
      algorithms: [
        { id: 'ppo', name: 'PPO', kind: 'gym', description: '', hyperparams: [] },
        { id: 'dqn', name: 'DQN', kind: 'gym', description: '', hyperparams: [] },
        { id: 'a2c', name: 'A2C', kind: 'gym', description: '', hyperparams: [] },
        { id: 'ippo', name: 'IPPO', kind: 'gym', description: '', hyperparams: [] },
        { id: 'qmix', name: 'QMIX', kind: 'gym', description: '', hyperparams: [] },
      ],
    }),
    resolveUrl: async (p: string) => p,
  },
  createMetricsWebSocket: vi.fn(),
}))

beforeAll(() => {
  // @xyflow/react needs ResizeObserver, and Environments' lazy-preview cards
  // need IntersectionObserver — neither is implemented by jsdom.
  if (!('ResizeObserver' in globalThis)) {
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  }
  if (!('IntersectionObserver' in globalThis)) {
    (globalThis as unknown as { IntersectionObserver: unknown }).IntersectionObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  }
})

function Harness() {
  const qc = new QueryClient()
  return (
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={['/environments']}>
        <Routes>
          <Route path="/environments" element={<Environments />} />
          <Route
            path="/designer"
            element={
              <>
                <ExperimentDesigner />
                {/* Test-only escape hatch to simulate the user navigating back
                    to the gallery via the sidebar and picking another env. */}
                <Link to="/environments">back-to-gallery</Link>
              </>
            }
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('Environments -> ExperimentDesigner env handoff', () => {
  it('preselects the environment the user clicked "Использовать в дизайнере" on', async () => {
    render(<Harness />)

    await waitFor(() => expect(screen.getAllByText('Pendulum').length).toBeGreaterThan(0))

    const buttons = screen.getAllByRole('button', { name: 'Использовать в дизайнере' })
    // Cards render in catalog order: CartPole first, Pendulum second.
    fireEvent.click(buttons[1])

    // ReactFlow measures nodes via ResizeObserver before marking them visible,
    // which our stub doesn't drive — query the DOM directly instead of via
    // role, since accessibility queries exclude visibility:hidden ancestors.
    await waitFor(() => {
      const select = document.querySelector('select') as HTMLSelectElement | null
      expect(select).not.toBeNull()
      expect(select!.value).toBe('Pendulum-v1')
    })
  })

  it('picks up a newly requested env on repeated round-trips to the gallery', async () => {
    render(<Harness />)

    await waitFor(() => expect(screen.getAllByText('Pendulum').length).toBeGreaterThan(0))

    // Round 1: pick Pendulum.
    fireEvent.click(screen.getAllByRole('button', { name: 'Использовать в дизайнере' })[1])
    await waitFor(() => {
      expect((document.querySelector('select') as HTMLSelectElement).value).toBe('Pendulum-v1')
    })

    // Back to the gallery, then pick CartPole this time.
    fireEvent.click(screen.getByText('back-to-gallery'))
    await waitFor(() => expect(screen.getAllByText('CartPole').length).toBeGreaterThan(0))
    fireEvent.click(screen.getAllByRole('button', { name: 'Использовать в дизайнере' })[0])
    await waitFor(() => {
      expect((document.querySelector('select') as HTMLSelectElement).value).toBe('CartPole-v1')
    })

    // Back again, and pick Pendulum a second time — this is the exact repro
    // for "я опять нажал на среду, но всё равно CartPole по дефолту".
    fireEvent.click(screen.getByText('back-to-gallery'))
    await waitFor(() => expect(screen.getAllByText('Pendulum').length).toBeGreaterThan(0))
    fireEvent.click(screen.getAllByRole('button', { name: 'Использовать в дизайнере' })[1])
    await waitFor(() => {
      expect((document.querySelector('select') as HTMLSelectElement).value).toBe('Pendulum-v1')
    })
  })

  it('defaults to ippo (not dqn/ppo) when picking a multi-team MARL env', async () => {
    render(<Harness />)

    await waitFor(() => expect(screen.getAllByText('MPE: Simple Tag').length).toBeGreaterThan(0))
    fireEvent.click(screen.getAllByRole('button', { name: 'Использовать в дизайнере' })[2])

    await waitFor(() => {
      const selects = Array.from(document.querySelectorAll('select'))
      const algoSelect = selects.find((s) => Array.from(s.options).some((o) => o.value === 'ippo'))
      expect(algoSelect).toBeTruthy()
      expect(algoSelect!.value).toBe('ippo')
    })
  })
})

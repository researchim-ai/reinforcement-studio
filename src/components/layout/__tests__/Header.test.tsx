import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import '@/i18n'
import { Header } from '@/components/layout/Header'

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Header />
    </MemoryRouter>,
  )
}

describe('Header titles', () => {
  it('uses the academy title, not the dashboard fallback', () => {
    renderAt('/academy')
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Учебный центр')
  })
})

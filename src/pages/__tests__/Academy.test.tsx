import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it } from 'vitest'

import { BEGINNER_TRACK } from '@/components/academy/beginnerTrack'
import { LESSON_EXERCISES } from '@/components/academy/exerciseBank'
import { LESSONS } from '@/components/academy/lessons'
import { Academy } from '@/pages/Academy'

function renderAcademy(path = '/academy') {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/academy" element={<Academy />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('Academy page', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('opens a deep-linked lesson and can move to the next one', () => {
    renderAcademy('/academy?lesson=researchimzero')
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('ResearchImZero')
    fireEvent.click(screen.getByRole('button', { name: /Дальше/ }))
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('LatentImZero')
  })

  it('filters the sidebar by search query even on the beginner track', () => {
    renderAcademy('/academy?lesson=fundamentals')
    fireEvent.change(screen.getByLabelText('Поиск по урокам'), { target: { value: 'qmix' } })
    expect(screen.getByRole('button', { name: /QMIX/ })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /^DQN —/ })).toBeNull()
  })

  it('hides encyclopedia lessons on the beginner track until search or toggle', () => {
    renderAcademy('/academy?lesson=fundamentals')
    expect(screen.getByRole('switch', { name: 'Трек новичка' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.queryByRole('button', { name: /QMIX/ })).toBeNull()
    expect(screen.queryByRole('button', { name: /SAC —/ })).toBeNull()
    expect(screen.getByRole('button', { name: /PPO —/ })).toBeTruthy()
  })

  it('sends a beginner from PPO to reading curves, not SAC', () => {
    renderAcademy('/academy?lesson=ppo')
    fireEvent.click(screen.getByRole('button', { name: /Дальше/ }))
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Как читать графики')
  })

  it('restores full-course order when the beginner track is off', () => {
    localStorage.setItem('rs.academy.trackMode', 'all')
    renderAcademy('/academy?lesson=ppo')
    fireEvent.click(screen.getByRole('button', { name: /Дальше/ }))
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('SAC')
  })
})

describe('Academy exercises', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('accepts a numeric return with a comma decimal and marks it solved', () => {
    renderAcademy('/academy?lesson=rewards_returns')
    expect(screen.getByRole('heading', { name: /Проверьте себя/ })).toBeTruthy()
    fireEvent.change(screen.getByLabelText('Ответ на вопрос 1'), { target: { value: '5,23' } })
    fireEvent.click(screen.getAllByRole('button', { name: 'Проверить' })[0])
    expect(screen.getByText('Верно')).toBeTruthy()
    expect(localStorage.getItem('rs.academy.exercises')).toContain('rewards_returns::g')
  })

  it('keeps a wrong numeric answer retryable without revealing the key', () => {
    renderAcademy('/academy?lesson=value_functions')
    fireEvent.change(screen.getByLabelText('Ответ на вопрос 1'), { target: { value: '0' } })
    fireEvent.click(screen.getAllByRole('button', { name: 'Проверить' })[0])
    expect(screen.getByText(/Неверно/)).toBeTruthy()
    expect(screen.queryByText(/bootstrap с терминала/)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Показать решение' }))
    expect(screen.getByText(/bootstrap с терминала/)).toBeTruthy()
    expect(localStorage.getItem('rs.academy.exercises') ?? '[]').not.toContain('value_functions::vb')
  })

  it('accepts a multiple-choice answer', () => {
    renderAcademy('/academy?lesson=what_is_rl')
    fireEvent.click(screen.getByRole('radio', { name: /нет правильного действия в датасете/i }))
    fireEvent.click(screen.getAllByRole('button', { name: 'Проверить' })[0])
    expect(screen.getByText('Верно')).toBeTruthy()
  })
})

describe('beginner track inventory', () => {
  it('only lists existing lessons, and each has at least one exercise', () => {
    const ids = new Set(LESSONS.map((lesson) => lesson.id))
    for (const id of BEGINNER_TRACK) {
      expect(ids.has(id)).toBe(true)
      expect(LESSON_EXERCISES[id]?.length ?? 0).toBeGreaterThan(0)
    }
  })
})

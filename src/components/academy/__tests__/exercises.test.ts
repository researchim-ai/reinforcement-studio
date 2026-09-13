import { describe, expect, it } from 'vitest'
import { numberMatches, parseExerciseNumber } from '@/components/academy/exercises'

describe('exercise number parsing', () => {
  it('accepts a comma as the decimal separator', () => {
    expect(parseExerciseNumber('5,23')).toBe(5.23)
    expect(parseExerciseNumber(' 0.9 ')).toBe(0.9)
    expect(parseExerciseNumber('')).toBeNull()
  })

  it('compares with tolerance', () => {
    expect(numberMatches(0.933, 0.933, 0.001)).toBe(true)
    expect(numberMatches(0.9333, 0.933, 0.001)).toBe(true)
    expect(numberMatches(0.93, 0.933, 0.001)).toBe(false)
    expect(numberMatches(5.23, 5.23, 0.015)).toBe(true)
  })
})

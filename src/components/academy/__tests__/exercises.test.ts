import { describe, expect, it } from 'vitest'
import { LESSON_EXERCISES } from '@/components/academy/exerciseBank'
import { numberMatches, parseExerciseNumber } from '@/components/academy/exercises'
import { LESSONS } from '@/components/academy/lessons'

describe('exercise bank', () => {
  it('has exercises for every lesson and no orphan entries', () => {
    const lessonIds = new Set(LESSONS.map((l) => l.id))
    const missing = [...lessonIds].filter((id) => !(LESSON_EXERCISES[id]?.length))
    expect(missing).toEqual([])
    expect(Object.keys(LESSON_EXERCISES).filter((id) => !lessonIds.has(id))).toEqual([])
  })

  it.each(Object.entries(LESSON_EXERCISES))('%s: ids unique, answers well-formed', (_lessonId, items) => {
    const ids = items.map((item) => item.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const item of items) {
      if (item.kind === 'choice') {
        expect(item.options.length).toBeGreaterThanOrEqual(2)
        expect(new Set(item.options).size).toBe(item.options.length)
        expect(Number.isInteger(item.correct)).toBe(true)
        expect(item.correct).toBeGreaterThanOrEqual(0)
        expect(item.correct).toBeLessThan(item.options.length)
      } else {
        expect(Number.isFinite(item.answer)).toBe(true)
      }
    }
  })
})

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

import { useCallback, useState, type ReactNode } from 'react'
import { Check, ListChecks, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

export const EXERCISES_KEY = 'rs.academy.exercises'
export const EXERCISES_EVENT = 'rs-academy-exercises'

export type ChoiceExercise = {
  id: string
  kind: 'choice'
  prompt: ReactNode
  options: string[]
  correct: number
  explain: ReactNode
}

export type NumberExercise = {
  id: string
  kind: 'number'
  prompt: ReactNode
  answer: number
  tolerance?: number
  placeholder?: string
  explain: ReactNode
}

export type Exercise = ChoiceExercise | NumberExercise

export function exerciseKey(lessonId: string, exerciseId: string): string {
  return `${lessonId}::${exerciseId}`
}

export function readSolvedKeys(): Set<string> {
  try {
    const parsed = JSON.parse(localStorage.getItem(EXERCISES_KEY) ?? '[]') as unknown
    return new Set(Array.isArray(parsed) ? parsed.filter((id): id is string => typeof id === 'string') : [])
  } catch {
    return new Set()
  }
}

function writeSolvedKeys(keys: Set<string>) {
  localStorage.setItem(EXERCISES_KEY, JSON.stringify([...keys]))
  window.dispatchEvent(new Event(EXERCISES_EVENT))
}

export function markExerciseSolved(lessonId: string, exerciseId: string) {
  const next = readSolvedKeys()
  next.add(exerciseKey(lessonId, exerciseId))
  writeSolvedKeys(next)
}

export function parseExerciseNumber(raw: string): number | null {
  const trimmed = raw.trim().replace(',', '.')
  if (!trimmed) return null
  const value = Number(trimmed)
  return Number.isFinite(value) ? value : null
}

export function numberMatches(value: number, answer: number, tolerance = 1e-6): boolean {
  return Math.abs(value - answer) <= tolerance
}

export function lessonExerciseProgress(
  lessonId: string,
  items: Exercise[] | undefined,
  solved = readSolvedKeys(),
): { solved: number; total: number } {
  const total = items?.length ?? 0
  if (!total) return { solved: 0, total: 0 }
  let n = 0
  for (const item of items!) {
    if (solved.has(exerciseKey(lessonId, item.id))) n += 1
  }
  return { solved: n, total }
}

export function bankExerciseProgress(
  bank: Record<string, Exercise[]>,
  solved = readSolvedKeys(),
): { solved: number; total: number } {
  let total = 0
  let n = 0
  for (const [lessonId, items] of Object.entries(bank)) {
    total += items.length
    for (const item of items) {
      if (solved.has(exerciseKey(lessonId, item.id))) n += 1
    }
  }
  return { solved: n, total }
}

function ExerciseCard({
  lessonId,
  index,
  item,
}: {
  lessonId: string
  index: number
  item: Exercise
}) {
  const already = readSolvedKeys().has(exerciseKey(lessonId, item.id))
  const [picked, setPicked] = useState<number | null>(already && item.kind === 'choice' ? item.correct : null)
  const [draft, setDraft] = useState(already && item.kind === 'number' ? String(item.answer) : '')
  const [status, setStatus] = useState<'idle' | 'correct' | 'wrong'>(already ? 'correct' : 'idle')
  const [reveal, setReveal] = useState(already)
  const [tried, setTried] = useState(already)

  const check = useCallback(() => {
    let ok = false
    if (item.kind === 'choice') {
      if (picked === null) return
      ok = picked === item.correct
    } else {
      const value = parseExerciseNumber(draft)
      if (value === null) return
      ok = numberMatches(value, item.answer, item.tolerance)
    }
    setTried(true)
    if (ok) {
      setStatus('correct')
      setReveal(true)
      markExerciseSolved(lessonId, item.id)
    } else {
      setStatus('wrong')
    }
  }, [draft, item, lessonId, picked])

  return (
    <div
      className={cn(
        'space-y-3 rounded-lg border px-3.5 py-3',
        status === 'correct' && 'border-emerald-500/40 bg-emerald-500/5',
        status === 'wrong' && 'border-destructive/40 bg-destructive/5',
        status === 'idle' && 'border-border/70 bg-muted/20',
      )}
    >
      <div className="flex items-start gap-2">
        <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-background text-[11px] font-semibold text-muted-foreground">
          {index + 1}
        </span>
        <div className="min-w-0 flex-1 space-y-1 text-[13px] leading-relaxed text-foreground/90">{item.prompt}</div>
      </div>

      {item.kind === 'choice' ? (
        <div className="flex flex-col gap-1.5" role="radiogroup" aria-label={`Вопрос ${index + 1}`}>
          {item.options.map((option, optionIndex) => {
            const selected = picked === optionIndex
            const showMark = status === 'correct' && optionIndex === item.correct
            return (
              <button
                key={option}
                type="button"
                role="radio"
                aria-checked={selected}
                onClick={() => {
                  setPicked(optionIndex)
                  if (status === 'wrong') setStatus('idle')
                }}
                className={cn(
                  'rounded-md border px-3 py-2 text-left text-[13px] transition-colors',
                  selected ? 'border-primary bg-primary/10 text-foreground' : 'border-border/60 bg-background/60 text-foreground/85 hover:bg-accent/50',
                  showMark && 'border-emerald-500/50',
                )}
              >
                {option}
              </button>
            )
          })}
        </div>
      ) : (
        <Input
          value={draft}
          onChange={(event) => {
            setDraft(event.target.value)
            if (status === 'wrong') setStatus('idle')
          }}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault()
              check()
            }
          }}
          placeholder={item.placeholder ?? 'число'}
          inputMode="decimal"
          autoComplete="off"
          aria-label={`Ответ на вопрос ${index + 1}`}
          className="h-9 max-w-[12rem] text-sm"
        />
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Button type="button" size="sm" onClick={check}>
          Проверить
        </Button>
        {tried && !reveal && (
          <Button type="button" size="sm" variant="ghost" onClick={() => setReveal(true)}>
            Показать решение
          </Button>
        )}
        {status === 'correct' && (
          <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-500">
            <Check className="h-3.5 w-3.5" /> Верно
          </span>
        )}
        {status === 'wrong' && (
          <span className="inline-flex items-center gap-1 text-xs font-medium text-destructive">
            <X className="h-3.5 w-3.5" /> Неверно, попробуйте ещё
          </span>
        )}
      </div>

      {reveal && (
        <div className="rounded-md bg-background/50 px-3 py-2 text-[12px] leading-relaxed text-muted-foreground">
          {item.explain}
        </div>
      )}
    </div>
  )
}

export function LessonExercises({ lessonId, items }: { lessonId: string; items: Exercise[] }) {
  if (items.length === 0) return null
  return (
    <section className="space-y-3 border-t border-border pt-6">
      <div className="space-y-1">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-foreground">
          <ListChecks className="h-4 w-4 text-primary" />
          Проверьте себя
        </h2>
        <p className="text-[12px] text-muted-foreground">
          Считается руками. В Дизайнере нет табличного Q-learning — после основ запускайте DQN или PPO
          на CartPole, а не «покликайте таблицу».
        </p>
      </div>
      <div className="space-y-3">
        {items.map((item, index) => (
          <ExerciseCard key={item.id} lessonId={lessonId} index={index} item={item} />
        ))}
      </div>
    </section>
  )
}

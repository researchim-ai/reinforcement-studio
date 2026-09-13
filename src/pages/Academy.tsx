import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Check, ChevronLeft, ChevronRight, GraduationCap, Search } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Switch } from '@/components/ui/switch'
import { cn } from '@/lib/utils'
import { BEGINNER_SET, BEGINNER_TRACK } from '@/components/academy/beginnerTrack'
import { LESSON_EXERCISES } from '@/components/academy/exerciseBank'
import {
  EXERCISES_EVENT,
  LessonExercises,
  bankExerciseProgress,
  lessonExerciseProgress,
  readSolvedKeys,
} from '@/components/academy/exercises'
import { LESSONS, LESSON_GROUPS, type Lesson } from '@/components/academy/lessons'

const LAST_KEY = 'rs.academy.lastLesson'
const VIEWED_KEY = 'rs.academy.viewed'
const TRACK_MODE_KEY = 'rs.academy.trackMode'

function readViewed(): Set<string> {
  try {
    const parsed = JSON.parse(localStorage.getItem(VIEWED_KEY) ?? '[]') as unknown
    return new Set(Array.isArray(parsed) ? parsed.filter((id): id is string => typeof id === 'string') : [])
  } catch {
    return new Set()
  }
}

function writeViewed(viewed: Set<string>) {
  localStorage.setItem(VIEWED_KEY, JSON.stringify([...viewed]))
}

function readBeginnerMode(): boolean {
  const raw = localStorage.getItem(TRACK_MODE_KEY)
  if (raw === 'all') return false
  if (raw === 'beginner') return true
  return true
}

function lessonFromParam(raw: string | null): Lesson | undefined {
  if (!raw) return undefined
  return LESSONS.find((lesson) => lesson.id === raw)
}

function matchesQuery(lesson: Lesson, query: string): boolean {
  if (!query) return true
  const haystack = [lesson.id, lesson.title, lesson.tagline, ...lesson.badges].join(' ').toLowerCase()
  return haystack.includes(query)
}

function beginnerLessons(): Lesson[] {
  return BEGINNER_TRACK
    .map((id) => LESSONS.find((lesson) => lesson.id === id))
    .filter((lesson): lesson is Lesson => lesson !== undefined)
}

export function Academy() {
  const [searchParams, setSearchParams] = useSearchParams()
  const requested = lessonFromParam(searchParams.get('lesson'))
  const stored = typeof window === 'undefined' ? undefined : lessonFromParam(localStorage.getItem(LAST_KEY))
  const active = requested ?? stored ?? LESSONS[0]
  const inBeginnerTrack = BEGINNER_SET.has(active.id)

  const [query, setQuery] = useState('')
  const [viewed, setViewed] = useState<Set<string>>(readViewed)
  const [beginnerMode, setBeginnerMode] = useState(readBeginnerMode)
  const [solvedKeys, setSolvedKeys] = useState<Set<string>>(readSolvedKeys)
  const contentRef = useRef<HTMLDivElement>(null)
  const normalizedQuery = query.trim().toLowerCase()

  const navLessons = useMemo(() => {
    if (beginnerMode && inBeginnerTrack) return beginnerLessons()
    return LESSONS
  }, [beginnerMode, inBeginnerTrack])

  const activeIndex = navLessons.findIndex((lesson) => lesson.id === active.id)
  const catalogIndex = LESSONS.findIndex((lesson) => lesson.id === active.id)
  const prev = activeIndex > 0 ? navLessons[activeIndex - 1] : undefined
  const next = activeIndex >= 0 && activeIndex < navLessons.length - 1 ? navLessons[activeIndex + 1] : undefined
  const exercises = LESSON_EXERCISES[active.id]
  const lessonExercises = lessonExerciseProgress(active.id, exercises, solvedKeys)
  const bankStats = bankExerciseProgress(
    beginnerMode
      ? Object.fromEntries(BEGINNER_TRACK.map((id) => [id, LESSON_EXERCISES[id] ?? []]))
      : LESSON_EXERCISES,
    solvedKeys,
  )

  const goTo = useCallback((id: string) => {
    setSearchParams({ lesson: id })
  }, [setSearchParams])

  const toggleBeginner = useCallback((checked: boolean) => {
    setBeginnerMode(checked)
    localStorage.setItem(TRACK_MODE_KEY, checked ? 'beginner' : 'all')
  }, [])

  useEffect(() => {
    if (!requested) {
      setSearchParams({ lesson: active.id }, { replace: true })
    }
  }, [active.id, requested, setSearchParams])

  useEffect(() => {
    localStorage.setItem(LAST_KEY, active.id)
    setViewed((current) => {
      if (current.has(active.id)) return current
      const nextViewed = new Set(current)
      nextViewed.add(active.id)
      writeViewed(nextViewed)
      return nextViewed
    })
    contentRef.current?.scrollTo?.({ top: 0 })
  }, [active.id])

  useEffect(() => {
    const onExercises = () => setSolvedKeys(readSolvedKeys())
    window.addEventListener(EXERCISES_EVENT, onExercises)
    return () => window.removeEventListener(EXERCISES_EVENT, onExercises)
  }, [])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return
      if (event.key === 'ArrowLeft' && prev) {
        event.preventDefault()
        goTo(prev.id)
      }
      if (event.key === 'ArrowRight' && next) {
        event.preventDefault()
        goTo(next.id)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [goTo, next, prev])

  const grouped = useMemo(() => {
    const pool = beginnerMode && !normalizedQuery ? beginnerLessons() : LESSONS
    const map = new Map<string, Lesson[]>()
    for (const group of LESSON_GROUPS) map.set(group, [])
    for (const lesson of pool) {
      if (!matchesQuery(lesson, normalizedQuery)) continue
      map.get(lesson.group)?.push(lesson)
    }
    if (beginnerMode && !normalizedQuery && !inBeginnerTrack) {
      const extra = map.get(active.group) ?? []
      if (!extra.some((lesson) => lesson.id === active.id)) {
        extra.unshift(active)
        map.set(active.group, extra)
      }
    }
    return map
  }, [active, beginnerMode, inBeginnerTrack, normalizedQuery])

  const viewedLabel = beginnerMode
    ? `Трек: ${[...viewed].filter((id) => BEGINNER_SET.has(id)).length} из ${BEGINNER_TRACK.length}`
    : `Просмотрено ${viewed.size} из ${LESSONS.length}`

  const positionLabel = beginnerMode && inBeginnerTrack
    ? `Трек новичка · урок ${activeIndex + 1} из ${navLessons.length}`
    : `Урок ${catalogIndex + 1} из ${LESSONS.length}`

  return (
    <div className="flex h-full">
      <aside className="flex w-72 shrink-0 flex-col border-r border-border">
        <div className="border-b border-border px-4 py-3">
          <h2 className="flex items-center gap-2 text-sm font-semibold">
            <GraduationCap className="h-4 w-4 text-primary" /> Учебный центр
          </h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Курс по RL: от агента и Беллмана до первого запуска и каждого алгоритма в Дизайнере.
          </p>
          <div className="mt-2 flex items-center justify-between gap-2 text-xs">
            <span className="text-foreground">Трек новичка</span>
            <Switch
              checked={beginnerMode}
              onCheckedChange={toggleBeginner}
              aria-label="Трек новичка"
            />
          </div>
          <p className="mt-2 text-[11px] text-muted-foreground">
            {viewedLabel}
            {bankStats.total > 0 && (
              <>
                {' · '}упражнения {bankStats.solved} из {bankStats.total}
              </>
            )}
          </p>
          <div className="relative mt-2">
            <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Поиск по урокам"
              className="h-8 pl-7 text-xs"
              aria-label="Поиск по урокам"
            />
          </div>
        </div>
        <ScrollArea className="flex-1">
          <div className="space-y-4 p-3">
            {LESSON_GROUPS.map((group) => {
              const items = grouped.get(group) ?? []
              if (items.length === 0) return null
              return (
                <div key={group}>
                  <div className="mb-1.5 px-1 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                    {group}
                  </div>
                  <div className="space-y-0.5">
                    {items.map((lesson) => {
                      const isActive = lesson.id === active.id
                      const isViewed = viewed.has(lesson.id)
                      const ex = lessonExerciseProgress(lesson.id, LESSON_EXERCISES[lesson.id], solvedKeys)
                      return (
                        <button
                          key={lesson.id}
                          onClick={() => goTo(lesson.id)}
                          className={cn(
                            'flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors',
                            isActive
                              ? 'bg-accent text-accent-foreground'
                              : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
                          )}
                        >
                          <lesson.icon className="h-3.5 w-3.5 shrink-0" />
                          <span className="min-w-0 flex-1 truncate">{lesson.title}</span>
                          {ex.total > 0 && (
                            <span
                              className={cn(
                                'shrink-0 text-[10px] tabular-nums',
                                ex.solved === ex.total ? 'text-emerald-500' : 'text-muted-foreground/70',
                              )}
                              aria-label={`Упражнения ${ex.solved} из ${ex.total}`}
                            >
                              {ex.solved}/{ex.total}
                            </span>
                          )}
                          {isViewed && !isActive && (
                            <Check className="h-3 w-3 shrink-0 text-emerald-500/80" aria-label="Просмотрено" />
                          )}
                        </button>
                      )
                    })}
                  </div>
                </div>
              )
            })}
            {normalizedQuery && [...grouped.values()].every((items) => items.length === 0) && (
              <p className="px-1 text-xs text-muted-foreground">Ничего не найдено.</p>
            )}
          </div>
        </ScrollArea>
      </aside>

      <ScrollArea ref={contentRef} className="flex-1">
        <div className="mx-auto max-w-3xl space-y-6 p-8">
          <div className="space-y-2 border-b border-border pb-5">
            <div className="flex flex-wrap items-center gap-1.5">
              {active.badges.map((b) => (
                <Badge key={b} variant="secondary" className="text-[10px]">{b}</Badge>
              ))}
              {beginnerMode && inBeginnerTrack && (
                <Badge variant="secondary" className="text-[10px]">Трек новичка</Badge>
              )}
            </div>
            <h1 className="flex items-center gap-2.5 text-2xl font-semibold">
              <active.icon className="h-6 w-6 shrink-0 text-primary" />
              {active.title}
            </h1>
            <p className="text-sm text-muted-foreground">{active.tagline}</p>
            <p className="text-[11px] text-muted-foreground">
              {positionLabel}
              {lessonExercises.total > 0 && (
                <span className="ml-2">
                  упражнения {lessonExercises.solved} из {lessonExercises.total}
                </span>
              )}
              <span className="ml-2 text-muted-foreground/70">← → листать с клавиатуры</span>
            </p>
            {beginnerMode && !inBeginnerTrack && (
              <p className="text-[12px] text-muted-foreground">
                Этот урок не входит в трек новичка. Поиск и «Дальше» идут по полному курсу, либо
                выключите фильтр в сайдбаре.
              </p>
            )}
          </div>
          <div className="space-y-6 pb-4">{active.content}</div>
          {exercises && <LessonExercises lessonId={active.id} items={exercises} />}
          <div className="flex items-stretch justify-between gap-3 border-t border-border pt-5 pb-8">
            {prev ? (
              <Button variant="outline" className="h-auto min-w-0 flex-1 justify-start py-3 text-left" onClick={() => goTo(prev.id)}>
                <ChevronLeft className="h-4 w-4 shrink-0" />
                <span className="min-w-0">
                  <span className="block text-[10px] uppercase tracking-wide text-muted-foreground">Назад</span>
                  <span className="block truncate text-sm">{prev.title}</span>
                </span>
              </Button>
            ) : <span className="flex-1" />}
            {next ? (
              <Button variant="outline" className="h-auto min-w-0 flex-1 justify-end py-3 text-right" onClick={() => goTo(next.id)}>
                <span className="min-w-0">
                  <span className="block text-[10px] uppercase tracking-wide text-muted-foreground">Дальше</span>
                  <span className="block truncate text-sm">{next.title}</span>
                </span>
                <ChevronRight className="h-4 w-4 shrink-0" />
              </Button>
            ) : (
              beginnerMode && inBeginnerTrack ? (
                <Button
                  variant="outline"
                  className="h-auto min-w-0 flex-1 justify-end py-3 text-right"
                  onClick={() => toggleBeginner(false)}
                >
                  <span className="min-w-0">
                    <span className="block text-[10px] uppercase tracking-wide text-muted-foreground">Трек пройден</span>
                    <span className="block truncate text-sm">Открыть весь курс</span>
                  </span>
                  <ChevronRight className="h-4 w-4 shrink-0" />
                </Button>
              ) : <span className="flex-1" />
            )}
          </div>
        </div>
      </ScrollArea>
    </div>
  )
}

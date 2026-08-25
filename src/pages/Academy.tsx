import { useMemo, useState } from 'react'
import { GraduationCap } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'
import { LESSONS, LESSON_GROUPS, type Lesson } from '@/components/academy/lessons'

export function Academy() {
  const [activeId, setActiveId] = useState<string>(LESSONS[0].id)
  const active = useMemo<Lesson>(() => LESSONS.find((l) => l.id === activeId) ?? LESSONS[0], [activeId])

  const grouped = useMemo(() => {
    const map = new Map<string, Lesson[]>()
    for (const group of LESSON_GROUPS) map.set(group, [])
    for (const lesson of LESSONS) map.get(lesson.group)?.push(lesson)
    return map
  }, [])

  return (
    <div className="flex h-full">
      <aside className="flex w-72 shrink-0 flex-col border-r border-border">
        <div className="border-b border-border px-4 py-3">
          <h2 className="flex items-center gap-2 text-sm font-semibold">
            <GraduationCap className="h-4 w-4 text-primary" /> Учебный центр
          </h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Курс по RL: как работает каждый алгоритм — с формулами и визуальными схемами, шаг за шагом.
          </p>
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
                      const isActive = lesson.id === activeId
                      return (
                        <button
                          key={lesson.id}
                          onClick={() => setActiveId(lesson.id)}
                          className={cn(
                            'flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors',
                            isActive
                              ? 'bg-accent text-accent-foreground'
                              : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
                          )}
                        >
                          <lesson.icon className="h-3.5 w-3.5 shrink-0" />
                          <span className="truncate">{lesson.title}</span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </ScrollArea>
      </aside>

      <ScrollArea className="flex-1">
        <div className="mx-auto max-w-3xl space-y-6 p-8">
          <div className="space-y-2 border-b border-border pb-5">
            <div className="flex flex-wrap items-center gap-1.5">
              {active.badges.map((b) => (
                <Badge key={b} variant="secondary" className="text-[10px]">{b}</Badge>
              ))}
            </div>
            <h1 className="flex items-center gap-2.5 text-2xl font-semibold">
              <active.icon className="h-6 w-6 shrink-0 text-primary" />
              {active.title}
            </h1>
            <p className="text-sm text-muted-foreground">{active.tagline}</p>
          </div>
          <div className="space-y-6 pb-8">{active.content}</div>
        </div>
      </ScrollArea>
    </div>
  )
}

import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import type { LucideIcon } from 'lucide-react'

export interface Lesson {
  id: string
  title: string
  tagline: string
  icon: LucideIcon
  group: string
  badges: string[]
  content: ReactNode
}

export function lessonPath(id: string): string {
  return `/academy?lesson=${id}`
}

export function LessonLink({ id, children }: { id: string; children: ReactNode }) {
  return (
    <Link
      to={lessonPath(id)}
      className="font-medium text-primary underline decoration-primary/30 underline-offset-2 hover:decoration-primary"
    >
      {children}
    </Link>
  )
}

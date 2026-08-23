import * as React from 'react'
import { cn } from '@/lib/utils'
import { X } from 'lucide-react'

interface DialogProps {
  open: boolean
  onClose: () => void
  title?: string
  children: React.ReactNode
  className?: string
}

function Dialog({ open, onClose, title, children, className }: DialogProps) {
  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 animate-fade-in" onClick={onClose}>
      <div
        className={cn('w-full max-w-md rounded-xl border border-border bg-card p-6 shadow-lg animate-slide-in', className)}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          {title && <h2 className="text-base font-semibold">{title}</h2>}
          <button onClick={onClose} className="ml-auto rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        </div>
        {children}
      </div>
    </div>
  )
}

export { Dialog }

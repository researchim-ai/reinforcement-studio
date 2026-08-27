import { useEffect, useState } from 'react'
import { Input, type InputProps } from '@/components/ui/input'

export interface NumericInputProps extends Omit<InputProps, 'type' | 'value' | 'onChange' | 'min' | 'max' | 'step'> {
  value: number
  onChange: (value: number) => void
  /** Round/commit as integer (default: false). */
  integer?: boolean
}

function formatValue(value: number, integer: boolean): string {
  if (!Number.isFinite(value)) return ''
  return integer ? String(Math.round(value)) : String(value)
}

function isValidPartial(text: string, integer: boolean): boolean {
  if (text === '' || text === '-') return true
  if (integer) return /^-?\d*$/.test(text)
  return /^-?(\d+\.?\d*|\.\d*)$/.test(text)
}

function parseDraft(text: string, integer: boolean): number | null {
  const trimmed = text.trim()
  if (trimmed === '' || trimmed === '-' || trimmed === '.' || trimmed === '-.') return null
  const n = integer ? Number.parseInt(trimmed, 10) : Number.parseFloat(trimmed)
  return Number.isFinite(n) ? n : null
}

/** Text-based numeric field — no native min/max, no spinner auto-repeat,
 * and the value can be fully cleared while typing. Commits on blur if the
 * draft is empty or invalid. */
export function NumericInput({
  value,
  onChange,
  integer = false,
  className,
  disabled,
  onFocus,
  onBlur,
  ...props
}: NumericInputProps) {
  const [focused, setFocused] = useState(false)
  const [draft, setDraft] = useState(() => formatValue(value, integer))

  useEffect(() => {
    if (!focused) setDraft(formatValue(value, integer))
  }, [value, focused, integer])

  return (
    <Input
      {...props}
      type="text"
      inputMode={integer ? 'numeric' : 'decimal'}
      autoComplete="off"
      disabled={disabled}
      className={className}
      value={focused ? draft : formatValue(value, integer)}
      onFocus={(e) => {
        setFocused(true)
        setDraft(formatValue(value, integer))
        onFocus?.(e)
      }}
      onChange={(e) => {
        const next = e.target.value
        if (!isValidPartial(next, integer)) return
        setDraft(next)
        const parsed = parseDraft(next, integer)
        if (parsed !== null) onChange(parsed)
      }}
      onBlur={(e) => {
        setFocused(false)
        const parsed = parseDraft(draft, integer)
        if (parsed !== null) {
          onChange(parsed)
          setDraft(formatValue(parsed, integer))
        } else {
          setDraft(formatValue(value, integer))
        }
        onBlur?.(e)
      }}
    />
  )
}

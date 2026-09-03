import { useMemo, useState } from 'react'
import { toast } from 'sonner'
import { Loader2, FlaskConical } from 'lucide-react'
import { Dialog } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { Tooltip } from '@/components/ui/tooltip'
import { api } from '@/api/client'
import { useSettingsStore } from '@/stores/settingsStore'
import type { AlgorithmSpec, EnvKind, WrapperNode } from '@/api/types'

function parseNumberList(text: string): number[] {
  return text
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number)
    .filter((n) => Number.isFinite(n))
}

/** Grid hyperparameter sweep + multi-seed launcher — builds one queued run
 * per (grid combination × seed), all sharing a `sweep_id` the backend
 * scheduler runs one at a time (see `backend/sweep_manager.py`), and hands
 * off to the Sweeps page to watch/compare them once created. */
export function SweepDialog({
  open, onClose, kind, environmentId, wrappers, algorithm, baseHyperparams, training, onCreated,
}: {
  open: boolean
  onClose: () => void
  kind: EnvKind
  environmentId: string
  wrappers: WrapperNode[]
  algorithm: AlgorithmSpec | undefined
  baseHyperparams: Record<string, number>
  training: Record<string, unknown>
  onCreated: (sweepId: string) => void
}) {
  const language = useSettingsStore((s) => s.language)
  const [name, setName] = useState('Sweep')
  const [enabledKeys, setEnabledKeys] = useState<Set<string>>(new Set())
  const [values, setValues] = useState<Record<string, string>>({})
  const [seedsText, setSeedsText] = useState('42')
  const [submitting, setSubmitting] = useState(false)

  const numericHyperparams = useMemo(
    () => (algorithm?.hyperparams ?? []).filter((hp) => !hp.options),
    [algorithm],
  )

  const grid = useMemo(() => {
    const out: Record<string, number[]> = {}
    for (const key of enabledKeys) {
      const parsed = parseNumberList(values[key] ?? '')
      if (parsed.length > 0) out[key] = parsed
    }
    return out
  }, [enabledKeys, values])

  const seeds = useMemo(() => parseNumberList(seedsText), [seedsText])
  const comboCount = Object.values(grid).reduce((acc, list) => acc * list.length, 1)
  const totalRuns = comboCount * Math.max(1, seeds.length)

  const toggleKey = (key: string) => {
    setEnabledKeys((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else {
        next.add(key)
        setValues((v) => (v[key] != null ? v : { ...v, [key]: String(baseHyperparams[key] ?? '') }))
      }
      return next
    })
  }

  const handleSubmit = async () => {
    if (totalRuns < 1) return
    setSubmitting(true)
    try {
      const res = await api.startSweep({
        kind,
        name,
        environment: { id: environmentId, wrappers },
        algorithm: { id: algorithm!.id, hyperparams: baseHyperparams },
        training,
        grid,
        seeds: seeds.length > 0 ? seeds : [Number(training.seed ?? 42)],
      })
      toast.success(`Sweep запущен: ${res.total} запусков в очереди`)
      onCreated(res.sweep_id)
      onClose()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось запустить sweep')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog open={open} onClose={onClose} title="Запустить как sweep" className="max-w-lg">
      <div className="space-y-4">
        <p className="text-xs text-muted-foreground">
          Перебирает сетку значений гиперпараметров × seeds — каждая комбинация запускается как
          отдельный обычный прогон, по одному за раз (чтобы не бороться за CPU/GPU). Прогресс и
          сравнение результатов — на странице «Sweeps».
        </p>

        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Название sweep</Label>
          <Input value={name} onChange={(e) => setName(e.target.value)} className="h-8 text-sm" />
        </div>

        <div className="max-h-56 space-y-2 overflow-auto pr-1">
          {numericHyperparams.length === 0 && (
            <p className="text-xs text-muted-foreground">У этого алгоритма нет числовых гиперпараметров для сетки.</p>
          )}
          {numericHyperparams.map((hp) => (
            <div key={hp.key} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={enabledKeys.has(hp.key)}
                onChange={() => toggleKey(hp.key)}
                className="accent-primary"
              />
              {hp.desc ? (
                <Tooltip content={hp.desc[language]} side="top">
                  <Label className="w-32 shrink-0 cursor-help truncate text-[11px] text-muted-foreground underline decoration-dotted decoration-muted-foreground/50 underline-offset-2">
                    {hp.label}
                  </Label>
                </Tooltip>
              ) : (
                <Label className="w-32 shrink-0 truncate text-[11px] text-muted-foreground">{hp.label}</Label>
              )}
              <Input
                disabled={!enabledKeys.has(hp.key)}
                value={values[hp.key] ?? String(baseHyperparams[hp.key] ?? hp.default)}
                onChange={(e) => setValues((v) => ({ ...v, [hp.key]: e.target.value }))}
                placeholder="0.0003, 0.001, 0.003"
                className="h-7 flex-1 text-xs"
              />
            </div>
          ))}
        </div>

        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Seeds (через запятую)</Label>
          <Input value={seedsText} onChange={(e) => setSeedsText(e.target.value)} className="h-8 text-sm" placeholder="42, 1, 2" />
        </div>

        <div className="rounded-lg border border-border p-2 text-center text-xs">
          Итого запусков: <span className="font-semibold">{totalRuns}</span>
          {totalRuns > 24 && <p className="mt-1 text-warning">Это может занять очень много времени — запуски идут строго по одному.</p>}
        </div>

        <Button className="w-full" onClick={handleSubmit} disabled={submitting || totalRuns < 1 || !algorithm}>
          {submitting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FlaskConical className="h-3.5 w-3.5" />}
          Запустить {totalRuns} {totalRuns === 1 ? 'прогон' : 'прогонов'}
        </Button>
      </div>
    </Dialog>
  )
}

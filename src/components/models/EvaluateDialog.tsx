import { useState } from 'react'
import { toast } from 'sonner'
import { Loader2, PlayCircle } from 'lucide-react'
import { Dialog } from '@/components/ui/dialog'
import { NumericInput } from '@/components/ui/numeric-input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { api } from '@/api/client'
import type { EvaluateResult } from '@/api/types'

/** "Прогнать N эпизодов без обучения" — separate from the periodic
 * in-training preview GIF (which never reports reward statistics): this
 * is the only place the app tells you how an already-trained model
 * actually performs, with a proper mean/std/best/worst over several
 * fresh episodes instead of a single anecdotal playthrough. Shared by the
 * Training Monitor (source="run") and Model Zoo (source="run" or
 * "checkpoint"). */
export function EvaluateDialog({
  open, onClose, source, id, label,
}: {
  open: boolean
  onClose: () => void
  source: 'run' | 'checkpoint'
  id: string
  label: string
}) {
  const [episodes, setEpisodes] = useState(10)
  const [recordGif, setRecordGif] = useState(true)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<EvaluateResult | null>(null)

  const handleClose = () => {
    if (running) return
    setResult(null)
    onClose()
  }

  const handleRun = async () => {
    setRunning(true)
    setResult(null)
    try {
      const res = await api.evaluateModel({ source, id, episodes, record_gif: recordGif })
      setResult(res)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось оценить модель')
    } finally {
      setRunning(false)
    }
  }

  return (
    <Dialog open={open} onClose={handleClose} title={`Оценить: ${label}`} className="max-w-lg">
      <div className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Прогоняет модель детерминированно (без обучения) заданное число полных эпизодов и
          показывает статистику награды — отдельно от «живого просмотра» во время тренировки.
        </p>

        {!result && (
          <div className="space-y-3">
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">Число эпизодов</Label>
              <NumericInput
                integer
                value={episodes}
                onChange={setEpisodes}
                className="h-8 text-sm"
              />
            </div>
            <div className="flex items-center justify-between">
              <Label className="text-xs text-muted-foreground">Записать GIF первого эпизода</Label>
              <Switch checked={recordGif} onCheckedChange={setRecordGif} />
            </div>
            <Button className="w-full" onClick={handleRun} disabled={running}>
              {running ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <PlayCircle className="h-3.5 w-3.5" />}
              {running ? 'Оцениваем...' : 'Запустить оценку'}
            </Button>
          </div>
        )}

        {result && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2 text-center sm:grid-cols-4">
              <Stat label="Средняя" value={result.reward_mean?.toFixed(2) ?? '—'} />
              <Stat label="Ст. откл." value={result.reward_std?.toFixed(2) ?? '—'} />
              <Stat label="Мин" value={result.reward_min?.toFixed(2) ?? '—'} />
              <Stat label="Макс" value={result.reward_max?.toFixed(2) ?? '—'} />
            </div>
            <p className="text-center text-xs text-muted-foreground">
              {result.episodes} эпизодов · средняя длина {result.length_mean?.toFixed(0) ?? '—'} шагов
            </p>
            {result.episode_gif_base64 && (
              <div className="flex justify-center">
                <img
                  src={`data:image/gif;base64,${result.episode_gif_base64}`}
                  alt="evaluation episode"
                  className="max-h-56 rounded-lg border border-border"
                />
              </div>
            )}
            <Button variant="outline" className="w-full" onClick={() => setResult(null)}>
              Оценить снова
            </Button>
          </div>
        )}
      </div>
    </Dialog>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border p-2">
      <div className="text-sm font-semibold">{value}</div>
      <div className="text-[10px] text-muted-foreground">{label}</div>
    </div>
  )
}

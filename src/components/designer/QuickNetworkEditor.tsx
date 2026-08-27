import { Plus, RotateCcw, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { NumericInput } from '@/components/ui/numeric-input'
import { Label } from '@/components/ui/label'

export interface QuickNetworkEditorProps {
  /** Shown (and edited from) whenever `hiddenLayers` is `null` — the
   * family's own default trunk, i.e. "nothing customized yet". */
  defaultSizes: number[]
  hiddenLayers: number[] | null
  onChange: (layers: number[] | null) => void
}

const MAX_LAYERS = 6

/** Compact inline editor for just the trunk's hidden-layer sizes — sits
 * right in the Algorithm node, under the "Архитектура сети" picker, so
 * quickly nudging the depth/width of the default network doesn't require
 * a trip to the full Network Builder page. The very first edit "activates"
 * a hand-designed `NetworkSpec` for the run (see `onChangeQuickLayers` in
 * `ExperimentDesigner.tsx`) — before that, `hiddenLayers` stays `null` and
 * the algorithm's original hardcoded architecture is used, unchanged. */
export function QuickNetworkEditor({ defaultSizes, hiddenLayers, onChange }: QuickNetworkEditorProps) {
  const sizes = hiddenLayers ?? defaultSizes
  const isCustomized = hiddenLayers != null

  const updateSize = (index: number, value: number) => {
    onChange(sizes.map((size, i) => (i === index ? value : size)))
  }
  const removeLayer = (index: number) => {
    onChange(sizes.filter((_, i) => i !== index))
  }
  const addLayer = () => {
    onChange([...sizes, sizes[sizes.length - 1] ?? 64])
  }

  return (
    <div className="space-y-1.5 border-t border-border pt-2">
      <div className="flex items-baseline justify-between gap-2">
        <Label className="text-[11px] text-muted-foreground">Скрытые слои (быстрая настройка)</Label>
        {isCustomized && (
          <button
            type="button"
            className="flex items-center gap-0.5 text-[10px] text-muted-foreground hover:text-foreground"
            onClick={() => onChange(null)}
          >
            <RotateCcw className="h-2.5 w-2.5" /> сбросить
          </button>
        )}
      </div>
      <div className="space-y-1">
        {sizes.map((size, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <span className="w-3 shrink-0 text-center text-[10px] text-muted-foreground/60">{i + 1}</span>
            <NumericInput
              integer
              value={size}
              onChange={(v) => updateSize(i, v)}
              className="h-6 flex-1 text-[11px]"
            />
            <span className="shrink-0 text-[9px] text-muted-foreground/50">нейронов</span>
            <button
              type="button"
              className="shrink-0 rounded p-0.5 text-muted-foreground hover:text-destructive"
              onClick={() => removeLayer(i)}
              aria-label="Удалить слой"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        ))}
        {sizes.length === 0 && (
          <p className="text-[10px] italic text-muted-foreground/60">
            Без скрытых слоёв — входы идут прямо на выходы сети
          </p>
        )}
      </div>
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="h-6 w-full text-[10px]"
        onClick={addLayer}
        disabled={sizes.length >= MAX_LAYERS}
      >
        <Plus className="h-3 w-3" /> Добавить слой
      </Button>
      <p className="text-[10px] text-muted-foreground/60">
        Меняет только основную часть сети — входные/выходные головы подстраиваются под среду сами. Для полного контроля (свёртки, LSTM и т.д.) откройте конструктор «своя».
      </p>
    </div>
  )
}

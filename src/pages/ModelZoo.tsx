import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { Loader2, Archive, Trash2, LineChart, Network, Sparkles, PlayCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useModels, useRunNetwork, useCheckpointNetwork } from '@/api/hooks'
import { SaveArchitectureDialog } from '@/components/networkbuilder/SaveArchitectureDialog'
import { EvaluateDialog } from '@/components/models/EvaluateDialog'
import { RunFolderLink } from '@/components/monitor/RunFolderLink'
import { api } from '@/api/client'
import { useQueryClient } from '@tanstack/react-query'
import { formatBytes } from '@/lib/utils'
import type { ModelInfo } from '@/api/types'

export function ModelZoo() {
  const { data, isLoading } = useModels()
  const models = data?.models ?? []

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-8">
      <div className="space-y-1">
        <h2 className="text-2xl font-semibold">Model Zoo</h2>
        <p className="text-sm text-muted-foreground">
          Все обученные агенты: сырые запуски и модели, сохранённые вручную из Мониторинга.
        </p>
      </div>

      {models.length === 0 ? (
        <Card>
          <CardContent className="p-8 text-center text-sm text-muted-foreground">
            Пока нет обученных моделей. Запусти эксперимент в Дизайнере.
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {models.map((model) => (
            <ModelCard key={`${model.source}-${model.id}`} model={model} />
          ))}
        </div>
      )}
    </div>
  )
}

function ModelCard({ model }: { model: ModelInfo }) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [saveArchOpen, setSaveArchOpen] = useState(false)
  const [evaluateOpen, setEvaluateOpen] = useState(false)

  // The network the source run/checkpoint actually trained with (see
  // `write_network_snapshot` / `promote_run`'s copy of `network.json`) —
  // only one of these two hooks is ever "enabled" depending on `source`.
  const runNetwork = useRunNetwork(model.source === 'run' ? model.id : undefined)
  const checkpointNetwork = useCheckpointNetwork(model.source === 'checkpoint' ? model.id : undefined)
  const network = model.source === 'run' ? runNetwork.data : checkpointNetwork.data
  const canSaveArchitecture = !!network?.family && !!network?.spec

  const handleDelete = async () => {
    try {
      await api.deleteCheckpoint(model.id)
      queryClient.invalidateQueries({ queryKey: ['models'] })
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось удалить')
    }
  }

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between space-y-0">
        <div className="flex items-center gap-2">
          <Archive className="h-4 w-4 text-primary" />
          <CardTitle className="text-sm">{model.label}</CardTitle>
        </div>
        <Badge variant={model.source === 'checkpoint' ? 'success' : 'outline'} className="text-[10px]">
          {model.source === 'checkpoint' ? 'сохранена' : 'run'}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-1 text-xs text-muted-foreground">
          <Badge variant="outline" className="text-[10px]">{model.environment_id}</Badge>
          <Badge variant="outline" className="text-[10px] uppercase">{model.algorithm_id}</Badge>
          <span>{formatBytes(model.size_bytes)}</span>
        </div>
        {model.source === 'run' && (
          <RunFolderLink runId={model.id} className="text-[10px]" />
        )}
        <div className="flex flex-wrap gap-2">
          {model.kind === 'gym' && model.source === 'run' && (
            <Button size="sm" variant="outline" onClick={() => navigate(`/monitor?run=${model.id}`)}>
              <LineChart className="h-3.5 w-3.5" /> Метрики
            </Button>
          )}
          {model.kind === 'gym' && (
            <Button size="sm" variant="outline" onClick={() => setEvaluateOpen(true)}>
              <PlayCircle className="h-3.5 w-3.5" /> Оценить
            </Button>
          )}
          {model.kind === 'gym' && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => navigate(model.source === 'run' ? `/designer?resumeRun=${model.id}` : `/designer?resumeCheckpoint=${model.id}`)}
            >
              <Sparkles className="h-3.5 w-3.5" /> Дообучить
            </Button>
          )}
          {canSaveArchitecture && (
            <Button size="sm" variant="outline" onClick={() => setSaveArchOpen(true)}>
              <Network className="h-3.5 w-3.5" /> Архитектура
            </Button>
          )}
          {model.source === 'checkpoint' && (
            <Button size="sm" variant="destructive" onClick={handleDelete}>
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </CardContent>
      {canSaveArchitecture && network && (
        <SaveArchitectureDialog
          open={saveArchOpen}
          onClose={() => setSaveArchOpen(false)}
          family={network.family!}
          spec={network.spec!}
          defaultName={`${model.label} — сеть`}
        />
      )}
      <EvaluateDialog
        open={evaluateOpen}
        onClose={() => setEvaluateOpen(false)}
        source={model.source}
        id={model.id}
        label={model.label}
      />
    </Card>
  )
}

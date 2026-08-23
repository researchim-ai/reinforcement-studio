import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { Loader2, Archive, Trash2, LineChart, Swords } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useModels } from '@/api/hooks'
import { api } from '@/api/client'
import { useQueryClient } from '@tanstack/react-query'
import { formatBytes } from '@/lib/utils'

export function ModelZoo() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { data, isLoading } = useModels()
  const models = data?.models ?? []

  const handleDelete = async (id: string) => {
    try {
      await api.deleteCheckpoint(id)
      queryClient.invalidateQueries({ queryKey: ['models'] })
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось удалить')
    }
  }

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
            <Card key={`${model.source}-${model.id}`}>
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
                <div className="flex gap-2">
                  {model.kind === 'gym' && model.source === 'run' && (
                    <Button size="sm" variant="outline" onClick={() => navigate(`/monitor?run=${model.id}`)}>
                      <LineChart className="h-3.5 w-3.5" /> Метрики
                    </Button>
                  )}
                  {model.kind === 'alphazero' && (
                    <Button size="sm" variant="outline" onClick={() => navigate('/arena')}>
                      <Swords className="h-3.5 w-3.5" /> Играть
                    </Button>
                  )}
                  {model.source === 'checkpoint' && (
                    <Button size="sm" variant="destructive" onClick={() => handleDelete(model.id)}>
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}

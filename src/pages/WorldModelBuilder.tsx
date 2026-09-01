import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import {
  AlertTriangle, Boxes, Brain, CheckCircle2, Loader2, Paperclip, PlayCircle, Plus, Save, Trash2,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select } from '@/components/ui/select'
import { NumericInput } from '@/components/ui/numeric-input'
import { Dialog } from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { api } from '@/api/client'
import { useEnvironments, useWorldModelTypes, useWorldModels } from '@/api/hooks'
import { ALGORITHMS_FOR_TYPE, WORLD_MODEL_TYPE_DESCRIPTIONS, WORLD_MODEL_TYPE_LABELS, defaultConfigForType } from '@/lib/worldModels'
import type { WorldModelConfig, WorldModelMeta, WorldModelType } from '@/api/types'

const SLUG_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/
const WORLD_MODEL_TYPES: WorldModelType[] = ['rssm', 'ensemble', 'vae_mdnrnn']

interface Selected {
  slug: string
  isNew: boolean
}

export function WorldModelBuilder() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { data: worldModelsData } = useWorldModels()
  const worldModels: WorldModelMeta[] = worldModelsData?.world_models ?? []
  const { data: typesData } = useWorldModelTypes()
  const typeInfo = typesData?.types ?? []
  const { data: envData } = useEnvironments()
  const gymEnvs = useMemo(() => (envData?.environments ?? []).filter((e) => e.kind === 'gym'), [envData])

  const [selected, setSelected] = useState<Selected | null>(null)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [type, setType] = useState<WorldModelType>('rssm')
  const [config, setConfig] = useState<WorldModelConfig>({})
  const [environmentId, setEnvironmentId] = useState<string | null>(null)
  const [meta, setMeta] = useState<WorldModelMeta | null>(null)
  const [savedSnapshot, setSavedSnapshot] = useState('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)

  const [newDialogOpen, setNewDialogOpen] = useState(false)
  const [newSlug, setNewSlug] = useState('')
  const [newType, setNewType] = useState<WorldModelType>('rssm')

  const [attachRunId, setAttachRunId] = useState('')
  const [attaching, setAttaching] = useState(false)

  // Standalone "Train" run controls — a plain `kind: "world_model"` run
  // (rl_core/world_models/trainer.py), same run-tracking pipeline as any
  // other algorithm.
  const [trainEnvId, setTrainEnvId] = useState('')
  const [totalTimesteps, setTotalTimesteps] = useState(20_000)
  const [seed, setSeed] = useState(42)
  const [batchSize, setBatchSize] = useState(32)
  const [seqLen, setSeqLen] = useState(50)
  const [collectStepsPerIter, setCollectStepsPerIter] = useState(200)
  const [trainStepsPerIter, setTrainStepsPerIter] = useState(40)
  const [learningRate, setLearningRate] = useState(1e-3)
  // Parallel env lanes for data collection (rl_core/world_models/trainer.py
  // — "run several environments in parallel with different seeds"),
  // exactly the same `training.num_envs` lever every other algorithm's run
  // already has in the Designer.
  const [numEnvs, setNumEnvs] = useState(1)
  const [starting, setStarting] = useState(false)

  useEffect(() => {
    if (!trainEnvId && gymEnvs.length) setTrainEnvId(gymEnvs[0].id)
  }, [gymEnvs, trainEnvId])

  useEffect(() => {
    if (!selected || selected.isNew) return
    setLoading(true)
    api
      .getWorldModel(selected.slug)
      .then((doc) => {
        setName(doc.name)
        setDescription(doc.description)
        setType(doc.type)
        setConfig(doc.config)
        setEnvironmentId(doc.environment_id ?? null)
        setSavedSnapshot(JSON.stringify({ name: doc.name, description: doc.description, type: doc.type, config: doc.config, environment_id: doc.environment_id ?? null }))
      })
      .catch((err) => toast.error(err instanceof Error ? err.message : 'Не удалось загрузить World Model'))
      .finally(() => setLoading(false))
  }, [selected])

  useEffect(() => {
    setMeta(selected ? worldModels.find((w) => w.slug === selected.slug) ?? null : null)
  }, [selected, worldModels])

  const dirty = selected
    ? JSON.stringify({ name, description, type, config, environment_id: environmentId }) !== savedSnapshot
    : false

  const fields = typeInfo.find((t) => t.id === type)?.fields ?? []

  const openNewDialog = useCallback(() => {
    setNewSlug('')
    setNewType('rssm')
    setNewDialogOpen(true)
  }, [])

  const handleCreate = useCallback(() => {
    if (!SLUG_RE.test(newSlug)) return
    setSelected({ slug: newSlug, isNew: true })
    setName(newSlug)
    setDescription('')
    setType(newType)
    setConfig(defaultConfigForType(typeInfo, newType))
    setEnvironmentId(null)
    setSavedSnapshot('')
    setNewDialogOpen(false)
  }, [newSlug, newType, typeInfo])

  const handleTypeChange = useCallback(
    (next: WorldModelType) => {
      setType(next)
      setConfig(defaultConfigForType(typeInfo, next))
    },
    [typeInfo],
  )

  const handleSave = useCallback(async () => {
    if (!selected) return
    if (!SLUG_RE.test(selected.slug)) {
      toast.error('Некорректный идентификатор world model')
      return
    }
    setSaving(true)
    try {
      await api.saveWorldModel(selected.slug, { name: name || selected.slug, description, type, config, environment_id: environmentId })
      setSavedSnapshot(JSON.stringify({ name, description, type, config, environment_id: environmentId }))
      setSelected({ ...selected, isNew: false })
      await queryClient.invalidateQueries({ queryKey: ['world-models'] })
      toast.success('World Model сохранён')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось сохранить World Model')
    } finally {
      setSaving(false)
    }
  }, [selected, name, description, type, config, environmentId, queryClient])

  const handleDelete = useCallback(async () => {
    if (!selected || selected.isNew) return
    if (!window.confirm(`Удалить World Model «${selected.slug}»?`)) return
    try {
      await api.deleteWorldModel(selected.slug)
      await queryClient.invalidateQueries({ queryKey: ['world-models'] })
      setSelected(null)
      toast.success('World Model удалён')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось удалить World Model')
    }
  }, [selected, queryClient])

  const handleAttach = useCallback(async () => {
    if (!selected || selected.isNew || !attachRunId.trim()) return
    setAttaching(true)
    try {
      await api.attachWorldModelCheckpoint(selected.slug, attachRunId.trim())
      await queryClient.invalidateQueries({ queryKey: ['world-models'] })
      setAttachRunId('')
      toast.success('Веса прикреплены')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось прикрепить веса (проверьте id запуска)')
    } finally {
      setAttaching(false)
    }
  }, [selected, attachRunId, queryClient])

  const handleTrain = useCallback(async () => {
    if (!selected || !trainEnvId) return
    setStarting(true)
    try {
      const { run_id } = await api.startRun({
        kind: 'world_model',
        name: `${name || selected.slug} — обучение`,
        environment: { id: trainEnvId, wrappers: [] },
        algorithm: { id: type, hyperparams: config, world_model_id: selected.isNew ? null : selected.slug },
        training: {
          total_timesteps: totalTimesteps, seed, batch_size: batchSize, seq_len: seqLen,
          collect_steps_per_iter: collectStepsPerIter, train_steps_per_iter: trainStepsPerIter, learning_rate: learningRate,
          num_envs: numEnvs,
        },
      })
      toast.success(`Запуск создан: ${run_id}`)
      navigate(`/monitor?run=${run_id}`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось запустить обучение')
    } finally {
      setStarting(false)
    }
  }, [
    selected, trainEnvId, name, type, config, totalTimesteps, seed, batchSize, seqLen,
    collectStepsPerIter, trainStepsPerIter, learningRate, numEnvs, navigate,
  ])

  return (
    <div className="flex h-full">
      <aside className="flex w-72 shrink-0 flex-col border-r border-border">
        <div className="border-b border-border px-4 py-3">
          <h2 className="flex items-center gap-2 text-sm font-semibold">
            <Brain className="h-4 w-4 text-primary" /> World Models
          </h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Обучи модель мира отдельно (без агента) и переиспользуй её в Dreamer / MBPO / PETS / World Models (VAE+MDN-RNN) в Дизайнере.
          </p>
        </div>
        <div className="flex items-center justify-between border-b border-border px-4 py-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Мои world models</span>
          <Button variant="ghost" size="icon" className="h-6 w-6" onClick={openNewDialog}>
            <Plus className="h-3.5 w-3.5" />
          </Button>
        </div>
        <ScrollArea className="flex-1">
          <div className="space-y-0.5 p-2">
            {worldModels.length === 0 && <p className="px-2 py-1 text-xs text-muted-foreground">Пока нет world models</p>}
            {worldModels.map((item) => {
              const isActive = selected?.slug === item.slug
              return (
                <button
                  key={item.slug}
                  onClick={() => setSelected({ slug: item.slug, isNew: false })}
                  className={`flex w-full flex-col items-start rounded-md px-2 py-1.5 text-left text-sm transition-colors ${
                    isActive ? 'bg-accent text-accent-foreground' : 'hover:bg-accent/50'
                  }`}
                >
                  <span className="flex w-full items-center gap-1.5 truncate">
                    {item.broken && <AlertTriangle className="h-3 w-3 shrink-0 text-destructive" />}
                    <span className="truncate">{item.name}</span>
                  </span>
                  <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                    <span className="truncate">{item.slug}</span>
                    <Badge variant="secondary" className="text-[9px]">{WORLD_MODEL_TYPE_LABELS[item.type]}</Badge>
                    {item.trained && <CheckCircle2 className="h-3 w-3 shrink-0 text-success" />}
                  </span>
                </button>
              )
            })}
          </div>
        </ScrollArea>
      </aside>

      <div className="flex-1 overflow-auto">
        {!selected ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-muted-foreground">
            <Boxes className="h-8 w-8" />
            <p className="text-sm">Выберите World Model слева или создайте новый</p>
          </div>
        ) : loading ? (
          <div className="flex h-full items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <div className="space-y-4 p-6">
            <div className="flex flex-wrap items-center gap-2">
              <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Название" className="h-8 w-56 text-sm" />
              <Input
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Описание (необязательно)"
                className="h-8 flex-1 min-w-40 text-sm"
              />
              <Select
                value={type}
                onChange={(e) => handleTypeChange(e.target.value as WorldModelType)}
                options={WORLD_MODEL_TYPES.map((t) => ({ value: t, label: WORLD_MODEL_TYPE_LABELS[t] }))}
                className="h-8 w-64 text-xs"
              />
              {dirty && <Badge variant="outline" className="text-[10px]">не сохранено</Badge>}
              <Button size="sm" onClick={handleSave} disabled={saving || !dirty}>
                {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                Сохранить
              </Button>
              {!selected.isNew && (
                <Button variant="ghost" size="icon" onClick={handleDelete} className="text-destructive">
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              )}
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">{WORLD_MODEL_TYPE_LABELS[type]}</CardTitle>
                <p className="text-xs text-muted-foreground">{WORLD_MODEL_TYPE_DESCRIPTIONS[type]}</p>
                <p className="text-[11px] text-muted-foreground">
                  Используется алгоритмами: {ALGORITHMS_FOR_TYPE[type].join(', ')}
                </p>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {fields.map((field) => (
                    <div key={field.key} className="space-y-1">
                      <Label className="text-[11px] text-muted-foreground">{field.label}</Label>
                      <NumericInput
                        integer
                        value={config[field.key] ?? 0}
                        onChange={(v) => setConfig((prev) => ({ ...prev, [field.key]: v }))}
                        className="h-8 text-sm"
                      />
                    </div>
                  ))}
                </div>
                <div className="mt-3 flex items-center gap-2 text-xs">
                  {meta?.trained ? (
                    <>
                      <Badge className="bg-success/15 text-success">обучена</Badge>
                      {meta.trained_at && <span className="text-muted-foreground">{new Date(meta.trained_at).toLocaleString('ru-RU')}</span>}
                      {meta.source_run_id && <span className="font-mono text-muted-foreground">из {meta.source_run_id}</span>}
                    </>
                  ) : (
                    <Badge variant="outline">не обучена — будет построена заново со стандартными весами</Badge>
                  )}
                </div>
              </CardContent>
            </Card>

            {!selected.isNew && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Прикрепить веса из запуска</CardTitle>
                  <p className="text-xs text-muted-foreground">
                    Скопировать обученные веса завершённого запуска (standalone World Model или Dreamer/MBPO/PETS/World Models
                    с этим же world_model_id) в этот spec вручную — обычно не требуется, обучение ниже уже делает это само.
                  </p>
                </CardHeader>
                <CardContent className="flex items-center gap-2">
                  <Input value={attachRunId} onChange={(e) => setAttachRunId(e.target.value)} placeholder="run_id" className="h-8 w-64 text-sm" />
                  <Button size="sm" variant="outline" onClick={handleAttach} disabled={attaching || !attachRunId.trim()}>
                    {attaching ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Paperclip className="h-3.5 w-3.5" />}
                    Прикрепить
                  </Button>
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Обучить world model отдельно</CardTitle>
                <p className="text-xs text-muted-foreground">
                  Без агента — просто "собери опыт случайной политикой, обучи модель мира". Готовые веса автоматически
                  сохранятся в этот spec. Графики ошибки модели, GIF воображённой vs. реальной траектории и латентное
                  пространство доступны в Мониторе обучения.
                </p>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">Среда</Label>
                    <Select
                      value={trainEnvId}
                      onChange={(e) => setTrainEnvId(e.target.value)}
                      options={gymEnvs.map((g) => ({ value: g.id, label: g.name }))}
                      placeholder="Выберите среду"
                      className="h-8 text-xs"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">Всего шагов</Label>
                    <NumericInput integer value={totalTimesteps} onChange={setTotalTimesteps} className="h-8 text-sm" />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">Seed</Label>
                    <NumericInput integer value={seed} onChange={setSeed} className="h-8 text-sm" />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">Learning rate</Label>
                    <NumericInput value={learningRate} onChange={setLearningRate} className="h-8 text-sm" />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">Batch size</Label>
                    <NumericInput integer value={batchSize} onChange={setBatchSize} className="h-8 text-sm" />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">Длина последовательности</Label>
                    <NumericInput integer value={seqLen} onChange={setSeqLen} className="h-8 text-sm" />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">Шагов сбора на итерацию</Label>
                    <NumericInput integer value={collectStepsPerIter} onChange={setCollectStepsPerIter} className="h-8 text-sm" />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">Шагов обучения на итерацию</Label>
                    <NumericInput integer value={trainStepsPerIter} onChange={setTrainStepsPerIter} className="h-8 text-sm" />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground" title="Параллельные среды с разными сидами — ускоряет сбор опыта">
                      Параллельных сред
                    </Label>
                    <NumericInput integer value={numEnvs} onChange={(v) => setNumEnvs(Math.max(1, v))} className="h-8 text-sm" />
                  </div>
                </div>
                <Button onClick={handleTrain} disabled={starting || !trainEnvId || selected.isNew}>
                  {starting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <PlayCircle className="h-3.5 w-3.5" />}
                  Начать обучение
                </Button>
                {selected.isNew && <p className="text-[11px] text-destructive">Сначала сохраните World Model, чтобы обучение автоматически прикрепило веса.</p>}
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      <Dialog open={newDialogOpen} onClose={() => setNewDialogOpen(false)} title="Новый World Model">
        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label className="text-xs">Идентификатор (slug)</Label>
            <Input value={newSlug} onChange={(e) => setNewSlug(e.target.value.toLowerCase())} placeholder="my_world_model" />
            {newSlug && !SLUG_RE.test(newSlug) && (
              <p className="text-[11px] text-destructive">Только a-z, 0-9, «-», «_», начиная с буквы/цифры</p>
            )}
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">Тип модели</Label>
            <Select
              value={newType}
              onChange={(e) => setNewType(e.target.value as WorldModelType)}
              options={WORLD_MODEL_TYPES.map((t) => ({ value: t, label: WORLD_MODEL_TYPE_LABELS[t] }))}
            />
            <p className="text-[11px] text-muted-foreground">{WORLD_MODEL_TYPE_DESCRIPTIONS[newType]}</p>
          </div>
          <Button className="w-full" onClick={handleCreate} disabled={!SLUG_RE.test(newSlug)}>
            Создать
          </Button>
        </div>
      </Dialog>
    </div>
  )
}

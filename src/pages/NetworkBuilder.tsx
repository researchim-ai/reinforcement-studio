import { useCallback, useEffect, useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { ReactFlow, Background, Controls, type Node, type Edge } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { toast } from 'sonner'
import { AlertTriangle, Boxes, Cpu, Loader2, Network, Plus, Save, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select } from '@/components/ui/select'
import { Dialog } from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { api } from '@/api/client'
import { useEnvironments, useNetworkPreview, useNetworks } from '@/api/hooks'
import { useNodePositions } from '@/lib/useNodePositions'
import { LayerNode, type LayerNodeData } from '@/components/networkbuilder/LayerNode'
import { LabelNode, type LabelNodeData } from '@/components/networkbuilder/LabelNode'
import { AddLayerNode, type AddLayerNodeData } from '@/components/networkbuilder/AddLayerNode'
import {
  FAMILY_LABELS,
  HEAD_LABELS,
  defaultLayer,
  defaultSpecForFamily,
  reconcileHeadsForFamily,
} from '@/lib/networkBuilder'
import type { NetworkFamily, NetworkHead, NetworkLayer, NetworkMeta, NetworkPreviewResult, NetworkSpec } from '@/api/types'

const nodeTypes = { layer: LayerNode, label: LabelNode, add: AddLayerNode }

const SLUG_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/
const CENTER_X = 460
const HEAD_GAP = 280
const ROW_H = 110

interface Selected {
  slug: string
  isNew: boolean
}

interface GraphHandlers {
  updateTrunkLayer: (i: number, layer: NetworkLayer) => void
  removeTrunkLayer: (i: number) => void
  moveTrunkLayer: (i: number, dir: -1 | 1) => void
  addTrunkLayer: (index: number) => void
  updateHeadLayer: (headIdx: number, layerIdx: number, layer: NetworkLayer) => void
  removeHeadLayer: (headIdx: number, layerIdx: number) => void
  moveHeadLayer: (headIdx: number, layerIdx: number, dir: -1 | 1) => void
  addHeadLayer: (headIdx: number, index: number) => void
}

function buildGraph(
  spec: NetworkSpec,
  preview: NetworkPreviewResult | undefined,
  handlers: GraphHandlers,
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = []
  const edges: Edge[] = []

  const inputShape = preview?.input_shape
  nodes.push({
    id: 'input',
    type: 'label',
    position: { x: CENTER_X, y: 0 },
    data: { title: 'Вход (наблюдение)', subtitle: inputShape ? `[${inputShape.join('×')}]` : undefined, variant: 'input' } satisfies LabelNodeData,
  })

  // A "+" insertion slot sits in every gap of the chain — before the first
  // layer, between each pair, and after the last one — so a layer can be
  // inserted anywhere, not just appended at the end.
  let lastTrunkId = 'input'
  spec.trunk.forEach((layer, i) => {
    const id = `trunk-${i}`
    const built = i < (preview?.trunk.length ?? 0)
    const failed = preview?.trunk_error_index === i
    nodes.push({
      id: `add-trunk-${i}`,
      type: 'add',
      position: { x: CENTER_X + 190, y: (i + 0.5) * ROW_H },
      data: { onAdd: () => handlers.addTrunkLayer(i) } satisfies AddLayerNodeData,
    })
    nodes.push({
      id,
      type: 'layer',
      position: { x: CENTER_X, y: (i + 1) * ROW_H },
      data: {
        layer,
        outShape: built ? preview!.trunk[i].shape : null,
        error: failed ? preview!.trunk_error : null,
        removable: true,
        onChange: (l: NetworkLayer) => handlers.updateTrunkLayer(i, l),
        onRemove: () => handlers.removeTrunkLayer(i),
        onMoveUp: i > 0 ? () => handlers.moveTrunkLayer(i, -1) : undefined,
        onMoveDown: i < spec.trunk.length - 1 ? () => handlers.moveTrunkLayer(i, 1) : undefined,
      } satisfies LayerNodeData,
    })
    edges.push({ id: `e-${lastTrunkId}-${id}`, source: lastTrunkId, target: id })
    lastTrunkId = id
  })

  nodes.push({
    id: 'add-trunk-end',
    type: 'add',
    position: { x: CENTER_X + 190, y: spec.trunk.length * ROW_H },
    data: { onAdd: () => handlers.addTrunkLayer(spec.trunk.length) } satisfies AddLayerNodeData,
  })

  const headStartY = (spec.trunk.length + 2) * ROW_H
  const heads = spec.heads
  heads.forEach((head: NetworkHead, hi) => {
    const headX = CENTER_X + (hi - (heads.length - 1) / 2) * HEAD_GAP
    const headPreview = preview?.heads[head.name]
    const labelId = `head-label-${hi}`
    nodes.push({
      id: labelId,
      type: 'label',
      position: { x: headX, y: headStartY },
      data: {
        title: HEAD_LABELS[head.name] ?? head.name,
        subtitle: headPreview?.out_shape?.length ? `→ [${headPreview.out_shape.join('×')}]` : undefined,
        variant: 'head',
      } satisfies LabelNodeData,
    })
    edges.push({ id: `e-${lastTrunkId}-${labelId}`, source: lastTrunkId, target: labelId })

    let prevId = labelId
    const nonFinalCount = head.layers.length - 1
    head.layers.forEach((layer, li) => {
      const isFinal = li === head.layers.length - 1
      const id = `head-${hi}-${li}`
      const built = li < (headPreview?.layer_shapes.length ?? 0)
      const failed = headPreview?.error_index === li
      if (!isFinal) {
        nodes.push({
          id: `add-head-${hi}-${li}`,
          type: 'add',
          position: { x: headX + 190, y: headStartY + (li + 0.5) * ROW_H },
          data: { onAdd: () => handlers.addHeadLayer(hi, li) } satisfies AddLayerNodeData,
        })
      }
      nodes.push({
        id,
        type: 'layer',
        position: { x: headX, y: headStartY + (li + 1) * ROW_H },
        data: {
          layer,
          outShape: built ? headPreview!.layer_shapes[li] : null,
          error: failed ? headPreview!.error : null,
          locked: isFinal,
          removable: !isFinal,
          onChange: (l: NetworkLayer) => handlers.updateHeadLayer(hi, li, l),
          onRemove: isFinal ? undefined : () => handlers.removeHeadLayer(hi, li),
          onMoveUp: !isFinal && li > 0 ? () => handlers.moveHeadLayer(hi, li, -1) : undefined,
          onMoveDown: !isFinal && li < nonFinalCount - 1 ? () => handlers.moveHeadLayer(hi, li, 1) : undefined,
        } satisfies LayerNodeData,
      })
      edges.push({ id: `e-${prevId}-${id}`, source: prevId, target: id })
      prevId = id
    })

    nodes.push({
      id: `add-head-${hi}-end`,
      type: 'add',
      position: { x: headX + 190, y: headStartY + nonFinalCount * ROW_H },
      data: { onAdd: () => handlers.addHeadLayer(hi, nonFinalCount) } satisfies AddLayerNodeData,
    })
  })

  return { nodes, edges }
}

export function NetworkBuilderPage() {
  const queryClient = useQueryClient()
  const { data: networksData } = useNetworks()
  const networks: NetworkMeta[] = networksData?.networks ?? []
  const { data: envData } = useEnvironments()
  const environments = envData?.environments ?? []

  const [selected, setSelected] = useState<Selected | null>(null)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [family, setFamily] = useState<NetworkFamily>('actor_critic')
  const [spec, setSpec] = useState<NetworkSpec>(defaultSpecForFamily('actor_critic'))
  const [savedSnapshot, setSavedSnapshot] = useState('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)

  const [newDialogOpen, setNewDialogOpen] = useState(false)
  const [newSlug, setNewSlug] = useState('')
  const [newFamily, setNewFamily] = useState<NetworkFamily>('actor_critic')

  const gymEnvs = useMemo(() => environments.filter((e) => e.kind === 'gym'), [environments])
  const gameEnvs = useMemo(() => environments.filter((e) => e.kind === 'alphazero'), [environments])
  const [previewEnvId, setPreviewEnvId] = useState('')
  const [previewGameId, setPreviewGameId] = useState('')

  useEffect(() => {
    if (family === 'alphazero') {
      if (!previewGameId && gameEnvs.length) setPreviewGameId(gameEnvs[0].id)
    } else if (!previewEnvId && gymEnvs.length) {
      setPreviewEnvId(gymEnvs[0].id)
    }
  }, [family, gymEnvs, gameEnvs, previewEnvId, previewGameId])

  useEffect(() => {
    if (!selected || selected.isNew) return
    setLoading(true)
    api
      .getNetwork(selected.slug)
      .then((doc) => {
        setName(doc.name)
        setDescription(doc.description)
        setFamily(doc.family)
        setSpec(doc.spec)
        setSavedSnapshot(JSON.stringify({ name: doc.name, description: doc.description, family: doc.family, spec: doc.spec }))
      })
      .catch((err) => toast.error(err instanceof Error ? err.message : 'Не удалось загрузить архитектуру'))
      .finally(() => setLoading(false))
  }, [selected])

  const dirty = selected ? JSON.stringify({ name, description, family, spec }) !== savedSnapshot : false

  const preview = useNetworkPreview(
    selected
      ? {
          family,
          spec,
          environmentId: family === 'alphazero' ? null : previewEnvId,
          gameId: family === 'alphazero' ? previewGameId : null,
        }
      : null,
  )
  const previewResult = preview.data

  const openNewDialog = useCallback(() => {
    setNewSlug('')
    setNewFamily('actor_critic')
    setNewDialogOpen(true)
  }, [])

  const handleCreate = useCallback(() => {
    if (!SLUG_RE.test(newSlug)) return
    setSelected({ slug: newSlug, isNew: true })
    setName(newSlug)
    setDescription('')
    setFamily(newFamily)
    setSpec(defaultSpecForFamily(newFamily))
    setSavedSnapshot('')
    setNewDialogOpen(false)
  }, [newSlug, newFamily])

  const handleSave = useCallback(async () => {
    if (!selected) return
    if (!SLUG_RE.test(selected.slug)) {
      toast.error('Некорректный идентификатор архитектуры')
      return
    }
    setSaving(true)
    try {
      await api.saveNetwork(selected.slug, { name: name || selected.slug, description, family, spec })
      setSavedSnapshot(JSON.stringify({ name, description, family, spec }))
      setSelected({ ...selected, isNew: false })
      await queryClient.invalidateQueries({ queryKey: ['networks'] })
      toast.success('Архитектура сохранена')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось сохранить архитектуру')
    } finally {
      setSaving(false)
    }
  }, [selected, name, description, family, spec, queryClient])

  const handleDelete = useCallback(async () => {
    if (!selected || selected.isNew) return
    if (!window.confirm(`Удалить архитектуру «${selected.slug}»?`)) return
    try {
      await api.deleteNetwork(selected.slug)
      await queryClient.invalidateQueries({ queryKey: ['networks'] })
      setSelected(null)
      toast.success('Архитектура удалена')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось удалить архитектуру')
    }
  }, [selected, queryClient])

  const handleFamilyChange = useCallback((next: NetworkFamily) => {
    setFamily(next)
    setSpec((prev) => reconcileHeadsForFamily(prev, next))
  }, [])

  const updateTrunkLayer = useCallback((i: number, layer: NetworkLayer) => {
    setSpec((prev) => ({ ...prev, trunk: prev.trunk.map((l, idx) => (idx === i ? layer : l)) }))
  }, [])
  const removeTrunkLayer = useCallback((i: number) => {
    setSpec((prev) => ({ ...prev, trunk: prev.trunk.filter((_, idx) => idx !== i) }))
  }, [])
  const moveTrunkLayer = useCallback((i: number, dir: -1 | 1) => {
    setSpec((prev) => {
      const next = [...prev.trunk]
      const j = i + dir
      if (j < 0 || j >= next.length) return prev
      ;[next[i], next[j]] = [next[j], next[i]]
      return { ...prev, trunk: next }
    })
  }, [])
  const addTrunkLayer = useCallback((index: number) => {
    setSpec((prev) => {
      const trunk = [...prev.trunk]
      trunk.splice(index, 0, defaultLayer('linear'))
      return { ...prev, trunk }
    })
  }, [])

  const updateHeadLayer = useCallback((headIdx: number, layerIdx: number, layer: NetworkLayer) => {
    setSpec((prev) => ({
      ...prev,
      heads: prev.heads.map((h, hi) =>
        hi !== headIdx ? h : { ...h, layers: h.layers.map((l, li) => (li === layerIdx ? layer : l)) },
      ),
    }))
  }, [])
  const removeHeadLayer = useCallback((headIdx: number, layerIdx: number) => {
    setSpec((prev) => ({
      ...prev,
      heads: prev.heads.map((h, hi) => (hi !== headIdx ? h : { ...h, layers: h.layers.filter((_, li) => li !== layerIdx) })),
    }))
  }, [])
  const moveHeadLayer = useCallback((headIdx: number, layerIdx: number, dir: -1 | 1) => {
    setSpec((prev) => ({
      ...prev,
      heads: prev.heads.map((h, hi) => {
        if (hi !== headIdx) return h
        const next = [...h.layers]
        const j = layerIdx + dir
        if (j < 0 || j >= next.length - 1) return h
        ;[next[layerIdx], next[j]] = [next[j], next[layerIdx]]
        return { ...h, layers: next }
      }),
    }))
  }, [])
  const addHeadLayer = useCallback((headIdx: number, index: number) => {
    setSpec((prev) => ({
      ...prev,
      heads: prev.heads.map((h, hi) => {
        if (hi !== headIdx) return h
        const layers = [...h.layers]
        layers.splice(index, 0, defaultLayer('linear'))
        return { ...h, layers }
      }),
    }))
  }, [])

  const { applyPositions, onNodesChange } = useNodePositions()
  const { nodes: rawNodes, edges } = useMemo(
    () =>
      buildGraph(spec, previewResult, {
        updateTrunkLayer,
        removeTrunkLayer,
        moveTrunkLayer,
        addTrunkLayer,
        updateHeadLayer,
        removeHeadLayer,
        moveHeadLayer,
        addHeadLayer,
      }),
    [spec, previewResult, updateTrunkLayer, removeTrunkLayer, moveTrunkLayer, addTrunkLayer, updateHeadLayer, removeHeadLayer, moveHeadLayer, addHeadLayer],
  )
  const nodes = applyPositions(rawNodes)

  return (
    <div className="flex h-full">
      <aside className="flex w-72 shrink-0 flex-col border-r border-border">
        <div className="border-b border-border px-4 py-3">
          <h2 className="flex items-center gap-2 text-sm font-semibold">
            <Network className="h-4 w-4 text-primary" /> Архитектуры сетей
          </h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Собери свою нейросеть визуально — она появится как опция в узле алгоритма в Дизайнере.
          </p>
        </div>
        <div className="flex items-center justify-between border-b border-border px-4 py-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Мои архитектуры</span>
          <Button variant="ghost" size="icon" className="h-6 w-6" onClick={openNewDialog}>
            <Plus className="h-3.5 w-3.5" />
          </Button>
        </div>
        <ScrollArea className="flex-1">
          <div className="space-y-0.5 p-2">
            {networks.length === 0 && <p className="px-2 py-1 text-xs text-muted-foreground">Пока нет архитектур</p>}
            {networks.map((item) => {
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
                    <Badge variant="secondary" className="text-[9px]">{FAMILY_LABELS[item.family]}</Badge>
                  </span>
                </button>
              )
            })}
          </div>
        </ScrollArea>
      </aside>

      <div className="flex flex-1 flex-col">
        {!selected ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-muted-foreground">
            <Boxes className="h-8 w-8" />
            <p className="text-sm">Выберите архитектуру слева или создайте новую</p>
          </div>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2.5">
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Название"
                className="h-8 w-48 text-sm"
              />
              <Input
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Описание (необязательно)"
                className="h-8 flex-1 min-w-40 text-sm"
              />
              <Select
                value={family}
                onChange={(e) => handleFamilyChange(e.target.value as NetworkFamily)}
                options={Object.entries(FAMILY_LABELS).map(([value, label]) => ({ value, label }))}
                className="h-8 w-56 text-xs"
              />
              {dirty && <Badge variant="outline" className="text-[10px]">не сохранено</Badge>}
              <Button size="sm" onClick={handleSave} disabled={saving || loading || !dirty}>
                {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                Сохранить
              </Button>
              {!selected.isNew && (
                <Button variant="ghost" size="icon" onClick={handleDelete} className="text-destructive">
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-3 border-b border-border bg-muted/30 px-4 py-2">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Cpu className="h-3.5 w-3.5" /> Проверять на:
              </div>
              {family === 'alphazero' ? (
                <Select
                  value={previewGameId}
                  onChange={(e) => setPreviewGameId(e.target.value)}
                  options={gameEnvs.map((g) => ({ value: g.id, label: g.name }))}
                  placeholder="Выберите игру"
                  className="h-7 w-48 text-xs"
                />
              ) : (
                <Select
                  value={previewEnvId}
                  onChange={(e) => setPreviewEnvId(e.target.value)}
                  options={gymEnvs.map((g) => ({ value: g.id, label: g.name }))}
                  placeholder="Выберите среду"
                  className="h-7 w-56 text-xs"
                />
              )}
              <div className="ml-auto flex items-center gap-2 text-xs">
                {previewResult?.total_params != null && (
                  <Badge variant="secondary">{previewResult.total_params.toLocaleString('ru-RU')} параметров</Badge>
                )}
                {previewResult && !previewResult.ok && (
                  <span className="max-w-md truncate text-destructive" title={previewResult.error ?? ''}>
                    {previewResult.error}
                  </span>
                )}
                {previewResult?.ok && <Badge className="bg-success/15 text-success">валидна</Badge>}
              </div>
            </div>

            <div className="min-h-0 flex-1">
              <ReactFlow
                nodes={nodes}
                edges={edges}
                nodeTypes={nodeTypes}
                onNodesChange={onNodesChange}
                nodesConnectable={false}
                elementsSelectable={false}
                fitView
                fitViewOptions={{ padding: 0.3 }}
                proOptions={{ hideAttribution: true }}
              >
                <Background />
                <Controls showInteractive={false} />
              </ReactFlow>
            </div>
          </>
        )}
      </div>

      <Dialog open={newDialogOpen} onClose={() => setNewDialogOpen(false)} title="Новая архитектура">
        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label className="text-xs">Идентификатор (slug)</Label>
            <Input value={newSlug} onChange={(e) => setNewSlug(e.target.value.toLowerCase())} placeholder="my_network" />
            {newSlug && !SLUG_RE.test(newSlug) && (
              <p className="text-[11px] text-destructive">Только a-z, 0-9, «-», «_», начиная с буквы/цифры</p>
            )}
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">Тип сети</Label>
            <Select
              value={newFamily}
              onChange={(e) => setNewFamily(e.target.value as NetworkFamily)}
              options={Object.entries(FAMILY_LABELS).map(([value, label]) => ({ value, label }))}
            />
          </div>
          <Button className="w-full" onClick={handleCreate} disabled={!SLUG_RE.test(newSlug)}>
            Создать
          </Button>
        </div>
      </Dialog>
    </div>
  )
}

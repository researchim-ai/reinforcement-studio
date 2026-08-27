import { Suspense, useCallback, useEffect, useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Grid, OrbitControls } from '@react-three/drei'
import { toast } from 'sonner'
import { Box, BrickWall, Coins, Loader2, Package, Save, Skull, Users } from 'lucide-react'
import type { ThreeEvent } from '@react-three/fiber'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { NumericInput } from '@/components/ui/numeric-input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useScenes } from '@/api/hooks'
import { api } from '@/api/client'
import type {
  SceneItem, SceneMaterial, SceneMaterialPattern, SceneObject, SceneSelection, SceneShape, SceneSpec,
} from '@/api/types'
import { defaultSceneSpec, newId, slugify } from '@/lib/sceneDefaults'
import { PATTERN_OPTIONS, SHAPE_OPTIONS } from '@/lib/sceneMaterials'
import { SelectableMesh } from '@/components/scenebuilder/SceneMeshes'
import { useQueryClient } from '@tanstack/react-query'

function ShapeSelect({ value, onChange }: { value: SceneShape; onChange: (s: SceneShape) => void }) {
  return (
    <select className="h-8 w-full rounded-md border border-border bg-background px-2" value={value} onChange={(e) => onChange(e.target.value as SceneShape)}>
      {SHAPE_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
    </select>
  )
}

function MaterialFields({
  material, onChange, defaultColor,
}: {
  material?: SceneMaterial
  onChange: (m: SceneMaterial) => void
  defaultColor: string
}) {
  const m: SceneMaterial = {
    pattern: material?.pattern ?? 'solid',
    color: material?.color ?? defaultColor,
    color2: material?.color2 ?? '#1f2937',
    roughness: material?.roughness ?? 0.65,
    metalness: material?.metalness ?? 0.05,
    emissive: material?.emissive ?? false,
  }
  return (
    <div className="space-y-2 rounded-md border border-border/60 p-2">
      <p className="text-[10px] font-semibold uppercase text-muted-foreground">Материал</p>
      <div className="space-y-1">
        <Label>Узор</Label>
        <select
          className="h-8 w-full rounded-md border border-border bg-background px-2"
          value={m.pattern}
          onChange={(e) => onChange({ ...m, pattern: e.target.value as SceneMaterialPattern })}
        >
          {PATTERN_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
        </select>
      </div>
      <div className="flex items-center gap-2">
        <Label className="w-16 shrink-0">Цвет 1</Label>
        <input type="color" value={m.color} onChange={(e) => onChange({ ...m, color: e.target.value })} className="h-7 w-full cursor-pointer rounded border border-border bg-transparent" />
      </div>
      {m.pattern !== 'solid' && (
        <div className="flex items-center gap-2">
          <Label className="w-16 shrink-0">Цвет 2</Label>
          <input type="color" value={m.color2} onChange={(e) => onChange({ ...m, color2: e.target.value })} className="h-7 w-full cursor-pointer rounded border border-border bg-transparent" />
        </div>
      )}
      <label className="flex items-center gap-2">
        <input type="checkbox" checked={!!m.emissive} onChange={(e) => onChange({ ...m, emissive: e.target.checked })} />
        Свечение
      </label>
    </div>
  )
}

function SceneViewport({
  spec, selection, onSelect, onUpdateObject, onUpdateItem,
}: {
  spec: SceneSpec
  selection: SceneSelection
  onSelect: (s: SceneSelection) => void
  onUpdateObject: (id: string, patch: Partial<SceneObject>) => void
  onUpdateItem: (id: string, patch: Partial<SceneItem>) => void
}) {
  const agents = spec.agents[0]

  return (
    <Canvas camera={{ position: [18, 16, 18], fov: 45 }} onPointerMissed={() => onSelect(null)} shadows>
      <color attach="background" args={['#1a1a1f']} />
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 20, 10]} intensity={1} castShadow shadow-mapSize={[1024, 1024]} />
      <Grid args={[spec.world.width, spec.world.depth]} cellSize={1} sectionSize={5} fadeDistance={40} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow onClick={() => onSelect(null)}>
        <planeGeometry args={[spec.world.width, spec.world.depth]} />
        <meshStandardMaterial color="#2a2a32" />
      </mesh>

      {spec.objects.map((obj) => (
        <SelectableMesh
          key={obj.id}
          position={obj.position}
          size={obj.size}
          shape={obj.shape ?? 'box'}
          material={obj.material}
          fallbackColor={obj.type === 'prop' ? '#a78bfa' : '#6b7280'}
          selected={selection?.kind === 'object' && selection.id === obj.id}
          onSelect={() => onSelect({ kind: 'object', id: obj.id })}
          onTransform={(pos) => onUpdateObject(obj.id, { position: pos })}
        />
      ))}

      {spec.items.map((item) => (
        <SelectableMesh
          key={item.id}
          position={item.position}
          radius={item.radius}
          shape={item.shape ?? (item.type === 'reward' ? 'crystal' : 'sphere')}
          material={item.material ?? { pattern: 'solid', color: item.type === 'reward' ? '#22c55e' : '#ef4444', emissive: true }}
          fallbackColor={item.type === 'reward' ? '#22c55e' : '#ef4444'}
          selected={selection?.kind === 'item' && selection.id === item.id}
          onSelect={() => onSelect({ kind: 'item', id: item.id })}
          onTransform={(pos) => onUpdateItem(item.id, { position: pos })}
          yLift={item.radius}
        />
      ))}

      {agents && (
        <group onClick={(e: ThreeEvent<MouseEvent>) => { e.stopPropagation(); onSelect({ kind: 'agents' }) }}>
          <mesh position={agents.spawn.center}>
            <cylinderGeometry args={[agents.spawn.radius, agents.spawn.radius, 0.05, 32]} />
            <meshStandardMaterial color={selection?.kind === 'agents' ? '#818cf8' : '#6366f1'} transparent opacity={0.35} />
          </mesh>
          {Array.from({ length: Math.min(agents.count, 8) }).map((_, i) => {
            const ang = (i / Math.max(1, agents.count)) * Math.PI * 2
            const x = agents.spawn.center[0] + Math.cos(ang) * agents.spawn.radius * 0.5
            const z = agents.spawn.center[2] + Math.sin(ang) * agents.spawn.radius * 0.5
            return (
              <SelectableMesh
                key={i}
                position={[x, 0, z]}
                radius={agents.body_radius}
                shape={agents.shape ?? 'capsule'}
                material={agents.material ?? { pattern: 'solid', color: '#3b82f6' }}
                fallbackColor="#3b82f6"
                selected={selection?.kind === 'agents'}
                onSelect={() => onSelect({ kind: 'agents' })}
                onTransform={() => {}}
                yLift={agents.body_radius}
              />
            )
          })}
        </group>
      )}

      <OrbitControls makeDefault maxPolarAngle={Math.PI / 2.1} target={[0, 0, 0]} />
    </Canvas>
  )
}

function Inspector({
  spec, selection, onChangeSpec,
}: {
  spec: SceneSpec
  selection: SceneSelection
  onChangeSpec: (next: SceneSpec) => void
}) {
  if (!selection) {
    return (
      <div className="space-y-3 text-xs text-muted-foreground">
        <p>Кликните объект на сцене или выберите категорию слева.</p>
        <div className="space-y-1">
          <Label>Размер мира (ширина × глубина)</Label>
          <div className="flex gap-2">
            <NumericInput integer value={spec.world.width} onChange={(width) => onChangeSpec({ ...spec, world: { ...spec.world, width } })} className="h-8" />
            <NumericInput integer value={spec.world.depth} onChange={(depth) => onChangeSpec({ ...spec, world: { ...spec.world, depth } })} className="h-8" />
          </div>
        </div>
        <div className="space-y-1">
          <Label>Макс. шагов эпизода</Label>
          <NumericInput integer value={spec.episode?.max_steps ?? 500} onChange={(max_steps) => onChangeSpec({ ...spec, episode: { max_steps } })} className="h-8" />
        </div>
      </div>
    )
  }

  if (selection.kind === 'object') {
    const obj = spec.objects.find((o) => o.id === selection.id)
    if (!obj) return null
    const patch = (p: Partial<SceneObject>) =>
      onChangeSpec({ ...spec, objects: spec.objects.map((o) => (o.id === obj.id ? { ...o, ...p } : o)) })
    return (
      <div className="space-y-2 text-xs">
        <p className="font-medium">{obj.type === 'prop' ? 'Декор / проп' : 'Стена / объект'}</p>
        <div className="space-y-1">
          <Label>Форма</Label>
          <ShapeSelect value={obj.shape ?? 'box'} onChange={(shape) => patch({ shape })} />
        </div>
        {(['x', 'y', 'z'] as const).map((axis, i) => (
          <div key={axis} className="flex items-center gap-2">
            <Label className="w-6">{axis}</Label>
            <NumericInput value={obj.position[i]} onChange={(v) => {
              const pos = [...obj.position] as [number, number, number]
              pos[i] = v
              patch({ position: pos })
            }} className="h-7" />
          </div>
        ))}
        {(['w', 'h', 'd'] as const).map((label, i) => (
          <div key={label} className="flex items-center gap-2">
            <Label className="w-6">{label}</Label>
            <NumericInput value={obj.size[i]} onChange={(v) => {
              const size = [...obj.size] as [number, number, number]
              size[i] = v
              patch({ size })
            }} className="h-7" />
          </div>
        ))}
        <MaterialFields material={obj.material} defaultColor="#6b7280" onChange={(material) => patch({ material })} />
        <Button size="sm" variant="destructive" className="w-full" onClick={() => onChangeSpec({ ...spec, objects: spec.objects.filter((o) => o.id !== obj.id) })}>
          Удалить
        </Button>
      </div>
    )
  }

  if (selection.kind === 'item') {
    const item = spec.items.find((it) => it.id === selection.id)
    if (!item) return null
    const patch = (p: Partial<SceneItem>) =>
      onChangeSpec({ ...spec, items: spec.items.map((it) => (it.id === item.id ? { ...it, ...p } : it)) })
    return (
      <div className="space-y-2 text-xs">
        <p className="font-medium">{item.type === 'reward' ? 'Награда' : 'Опасность'}</p>
        <div className="space-y-1">
          <Label>Форма</Label>
          <ShapeSelect value={item.shape ?? (item.type === 'reward' ? 'crystal' : 'sphere')} onChange={(shape) => patch({ shape })} />
        </div>
        <div className="space-y-1">
          <Label>reward</Label>
          <NumericInput value={item.reward} onChange={(reward) => patch({ reward })} className="h-7" />
        </div>
        <div className="space-y-1">
          <Label>радиус (коллизия)</Label>
          <NumericInput value={item.radius} onChange={(radius) => patch({ radius })} className="h-7" />
        </div>
        {item.type === 'hazard' && (
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={!!item.terminate} onChange={(e) => patch({ terminate: e.target.checked })} />
            Завершать эпизод
          </label>
        )}
        {item.type === 'reward' && (
          <>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={item.respawn !== false} onChange={(e) => patch({ respawn: e.target.checked })} />
              Респавн
            </label>
            <div className="space-y-1">
              <Label>cooldown (шагов)</Label>
              <NumericInput integer value={item.cooldown_steps ?? 30} onChange={(cooldown_steps) => patch({ cooldown_steps })} className="h-7" />
            </div>
          </>
        )}
        <MaterialFields
          material={item.material}
          defaultColor={item.type === 'reward' ? '#22c55e' : '#ef4444'}
          onChange={(material) => patch({ material })}
        />
        <Button size="sm" variant="destructive" className="w-full" onClick={() => onChangeSpec({ ...spec, items: spec.items.filter((it) => it.id !== item.id) })}>
          Удалить
        </Button>
      </div>
    )
  }

  const agents = spec.agents[0]
  if (!agents) return null
  const patchAgents = (p: Partial<typeof agents>) =>
    onChangeSpec({ ...spec, agents: [{ ...agents, ...p }] })

  return (
    <div className="space-y-2 text-xs">
      <p className="font-medium">Агенты (общая политика)</p>
      <div className="space-y-1">
        <Label>Форма</Label>
        <ShapeSelect value={agents.shape ?? 'capsule'} onChange={(shape) => patchAgents({ shape })} />
      </div>
      <div className="space-y-1">
        <Label>Количество</Label>
        <NumericInput integer value={agents.count} onChange={(count) => patchAgents({ count })} className="h-7" />
      </div>
      <div className="space-y-1">
        <Label>Радиус спавна</Label>
        <NumericInput value={agents.spawn.radius} onChange={(radius) => patchAgents({ spawn: { ...agents.spawn, radius } })} className="h-7" />
      </div>
      <div className="space-y-1">
        <Label>Движение</Label>
        <select
          className="h-8 w-full rounded-md border border-border bg-background px-2"
          value={agents.movement.type}
          onChange={(e) => patchAgents({ movement: { ...agents.movement, type: e.target.value as 'discrete4' | 'discrete8' | 'continuous' } })}
        >
          <option value="discrete4">Дискретное 4 направления</option>
          <option value="discrete8">Дискретное 8 направлений</option>
          <option value="continuous">Непрерывное (Box)</option>
        </select>
      </div>
      <div className="space-y-1">
        <Label>Скорость</Label>
        <NumericInput value={agents.movement.speed} onChange={(speed) => patchAgents({ movement: { ...agents.movement, speed } })} className="h-7" />
      </div>
      <div className="space-y-1">
        <Label>Сенсоры k</Label>
        <NumericInput integer value={agents.sensors.k} onChange={(k) => patchAgents({ sensors: { ...agents.sensors, k } })} className="h-7" />
      </div>
      <MaterialFields material={agents.material} defaultColor="#3b82f6" onChange={(material) => patchAgents({ material })} />
    </div>
  )
}

export function SceneBuilder() {
  const qc = useQueryClient()
  const { data: scenesData } = useScenes()
  const scenes = scenesData?.scenes ?? []

  const [spec, setSpec] = useState<SceneSpec>(() => defaultSceneSpec())
  const [slug, setSlug] = useState('my_scene')
  const [selection, setSelection] = useState<SceneSelection>(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    api.getDefaultScene().then(setSpec).catch(() => {})
  }, [])

  const addWall = useCallback(() => {
    const id = newId('wall')
    setSpec((s) => ({
      ...s,
      objects: [...s.objects, {
        id, type: 'wall', position: [0, 1, 0], size: [2, 2, 2], shape: 'box',
        material: { pattern: 'brick', color: '#9ca3af', color2: '#4b5563' },
      }],
    }))
    setSelection({ kind: 'object', id })
  }, [])

  const addProp = useCallback(() => {
    const id = newId('prop')
    setSpec((s) => ({
      ...s,
      objects: [...s.objects, {
        id, type: 'prop', position: [0, 0.75, 0], size: [1.2, 1.5, 1.2], shape: 'cylinder',
        material: { pattern: 'stripes', color: '#c084fc', color2: '#581c87' },
      }],
    }))
    setSelection({ kind: 'object', id })
  }, [])

  const addItem = useCallback((type: 'reward' | 'hazard') => {
    const id = newId(type)
    setSpec((s) => ({
      ...s,
      items: [...s.items, {
        id, type, position: [0, 0, 0], radius: 0.5,
        reward: type === 'reward' ? 1 : -1,
        respawn: type === 'reward',
        cooldown_steps: 40,
        terminate: type === 'hazard',
        shape: type === 'reward' ? 'crystal' : 'sphere',
        material: {
          pattern: type === 'reward' ? 'dots' : 'noise',
          color: type === 'reward' ? '#22c55e' : '#ef4444',
          color2: type === 'reward' ? '#14532d' : '#7f1d1d',
          emissive: true,
        },
      }],
    }))
    setSelection({ kind: 'item', id })
  }, [])

  const updateObject = useCallback((id: string, patch: Partial<SceneObject>) => {
    setSpec((s) => ({ ...s, objects: s.objects.map((o) => (o.id === id ? { ...o, ...patch } : o)) }))
  }, [])

  const updateItem = useCallback((id: string, patch: Partial<SceneItem>) => {
    setSpec((s) => ({ ...s, items: s.items.map((it) => (it.id === id ? { ...it, ...patch } : it)) }))
  }, [])

  const handleSave = async () => {
    setSaving(true)
    try {
      const s = slugify(slug) || 'scene'
      await api.saveScene(s, spec)
      toast.success(`Сцена сохранена: scene:${s}`)
      setSlug(s)
      qc.invalidateQueries({ queryKey: ['scenes'] })
      qc.invalidateQueries({ queryKey: ['environments'] })
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось сохранить')
    } finally {
      setSaving(false)
    }
  }

  const loadScene = async (s: string) => {
    try {
      const doc = await api.getScene(s)
      setSpec(doc)
      setSlug(s)
      setSelection(null)
      toast.success(`Загружена сцена «${doc.name}»`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось загрузить')
    }
  }

  const agentCount = useMemo(() => spec.agents.reduce((n, g) => n + (g.count || 1), 0), [spec.agents])

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2">
        <Input value={spec.name} onChange={(e) => setSpec({ ...spec, name: e.target.value })} className="h-8 w-48" placeholder="Название" />
        <Input value={slug} onChange={(e) => setSlug(e.target.value)} className="h-8 w-36 font-mono text-xs" placeholder="slug" />
        <Button size="sm" onClick={handleSave} disabled={saving}>
          {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
          Сохранить
        </Button>
        <select
          className="h-8 rounded-md border border-border bg-background px-2 text-xs"
          value=""
          onChange={(e) => { if (e.target.value) loadScene(e.target.value) }}
        >
          <option value="">Загрузить…</option>
          {scenes.map((sc) => (
            <option key={sc.slug} value={sc.slug}>{sc.name}</option>
          ))}
        </select>
        <Button size="sm" variant="outline" onClick={() => { setSpec(defaultSceneSpec()); setSelection(null) }}>
          Новая
        </Button>
        <span className="ml-auto text-xs text-muted-foreground">{agentCount} агент(ов) · id: scene:{slugify(slug)}</span>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-[200px_1fr_260px]">
        <div className="space-y-2 overflow-y-auto border-r border-border p-3">
          <p className="text-xs font-semibold uppercase text-muted-foreground">Добавить</p>
          <Button size="sm" variant="outline" className="w-full justify-start gap-2" onClick={addWall}>
            <BrickWall className="h-3.5 w-3.5" /> Стена
          </Button>
          <Button size="sm" variant="outline" className="w-full justify-start gap-2" onClick={addProp}>
            <Package className="h-3.5 w-3.5" /> Проп / декор
          </Button>
          <Button size="sm" variant="outline" className="w-full justify-start gap-2" onClick={() => addItem('reward')}>
            <Coins className="h-3.5 w-3.5" /> Награда
          </Button>
          <Button size="sm" variant="outline" className="w-full justify-start gap-2" onClick={() => addItem('hazard')}>
            <Skull className="h-3.5 w-3.5" /> Опасность
          </Button>
          <Button size="sm" variant="outline" className="w-full justify-start gap-2" onClick={() => setSelection({ kind: 'agents' })}>
            <Users className="h-3.5 w-3.5" /> Агенты
          </Button>
          <p className="pt-2 text-[10px] leading-relaxed text-muted-foreground">
            Формы: куб, сфера, цилиндр, конус, капсула, пирамида, кристалл.
            Узоры генерируются процедурно (шахматка, кирпич, шум…).
          </p>
        </div>

        <div className="relative min-h-[400px]">
          <Suspense fallback={<div className="flex h-full items-center justify-center"><Loader2 className="h-6 w-6 animate-spin" /></div>}>
            <SceneViewport spec={spec} selection={selection} onSelect={setSelection} onUpdateObject={updateObject} onUpdateItem={updateItem} />
          </Suspense>
        </div>

        <Card className="rounded-none border-0 border-l border-border overflow-y-auto">
          <CardHeader className="p-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Box className="h-4 w-4" /> Инспектор
            </CardTitle>
          </CardHeader>
          <CardContent className="p-3 pt-0">
            <Inspector spec={spec} selection={selection} onChangeSpec={setSpec} />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import Editor, { loader as monacoLoader } from '@monaco-editor/react'
import * as monaco from 'monaco-editor'
import { toast } from 'sonner'
import {
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Gamepad2,
  Loader2,
  Plus,
  Puzzle,
  Save,
  Sparkles,
  Trash2,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select } from '@/components/ui/select'
import { Dialog } from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { api } from '@/api/client'
import { usePluginScripts, usePluginTemplates } from '@/api/hooks'
import type { PluginKind, PluginScriptMeta, ValidateResult } from '@/api/types'

interface SectionDef {
  kind: PluginKind
  title: string
  icon: typeof Cpu
  templateKind: 'gym-algorithm' | 'alphazero-algorithm' | 'reward-function'
}

const SECTIONS: SectionDef[] = [
  { kind: 'gym-algorithms', title: 'Gym-алгоритмы', icon: Gamepad2, templateKind: 'gym-algorithm' },
  { kind: 'alphazero-algorithms', title: 'AlphaZero-алгоритмы', icon: Cpu, templateKind: 'alphazero-algorithm' },
  { kind: 'reward-functions', title: 'Reward-функции', icon: Sparkles, templateKind: 'reward-function' },
]

const SLUG_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

// By default @monaco-editor/react fetches the editor from a CDN — this app
// runs offline once installed, so point it at the `monaco-editor` package
// bundled into the app instead. Without a dedicated web worker (Python has
// no built-in Monaco language worker anyway), it just runs the tokenizer on
// the main thread — a bit less snappy for huge files, fine for short scripts.
monacoLoader.config({ monaco })

interface Selected {
  kind: PluginKind
  slug: string
  isNew: boolean
}

export function PluginsPage() {
  const queryClient = useQueryClient()
  const { data: templateData } = usePluginTemplates()
  const templates = templateData?.templates ?? []

  const gymScripts = usePluginScripts('gym-algorithms')
  const azScripts = usePluginScripts('alphazero-algorithms')
  const rewardScripts = usePluginScripts('reward-functions')
  const scriptsByKind: Record<PluginKind, PluginScriptMeta[]> = {
    'gym-algorithms': gymScripts.data?.scripts ?? [],
    'alphazero-algorithms': azScripts.data?.scripts ?? [],
    'reward-functions': rewardScripts.data?.scripts ?? [],
  }

  const [selected, setSelected] = useState<Selected | null>(null)
  const [code, setCode] = useState('')
  const [savedCode, setSavedCode] = useState('')
  const [loadingCode, setLoadingCode] = useState(false)
  const [validating, setValidating] = useState(false)
  const [saving, setSaving] = useState(false)
  const [validateResult, setValidateResult] = useState<ValidateResult | null>(null)

  const [newDialog, setNewDialog] = useState<SectionDef | null>(null)
  const [newSlug, setNewSlug] = useState('')
  const [newTemplateId, setNewTemplateId] = useState('')

  const dirty = code !== savedCode

  const refetchKind = useCallback(
    (kind: PluginKind) => queryClient.invalidateQueries({ queryKey: ['plugin-scripts', kind] }),
    [queryClient],
  )

  useEffect(() => {
    if (!selected || selected.isNew) return
    setLoadingCode(true)
    setValidateResult(null)
    api
      .getPluginScript(selected.kind, selected.slug)
      .then((res) => {
        setCode(res.code)
        setSavedCode(res.code)
      })
      .catch((err) => toast.error(err instanceof Error ? err.message : 'Не удалось загрузить скрипт'))
      .finally(() => setLoadingCode(false))
  }, [selected])

  const handleSelect = useCallback((kind: PluginKind, slug: string) => {
    setSelected({ kind, slug, isNew: false })
  }, [])

  const openNewDialog = useCallback((section: SectionDef) => {
    setNewDialog(section)
    setNewSlug('')
    const first = templates.find((t) => t.kind === section.templateKind)
    setNewTemplateId(first?.id ?? '')
  }, [templates])

  const handleCreate = useCallback(() => {
    if (!newDialog || !SLUG_RE.test(newSlug)) return
    const template = templates.find((t) => t.id === newTemplateId)
    setSelected({ kind: newDialog.kind, slug: newSlug, isNew: true })
    setCode(template?.code ?? '')
    setSavedCode('')
    setValidateResult(null)
    setNewDialog(null)
  }, [newDialog, newSlug, newTemplateId, templates])

  const handleValidate = useCallback(async () => {
    if (!selected) return
    setValidating(true)
    setValidateResult(null)
    try {
      const result = await api.validatePluginScript(selected.kind, selected.slug, code)
      setValidateResult(result)
      if (result.ok) toast.success('Проверка прошла успешно')
      else toast.error('Проверка не прошла — смотрите детали ниже')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось выполнить проверку')
    } finally {
      setValidating(false)
    }
  }, [selected, code])

  const handleSave = useCallback(async () => {
    if (!selected) return
    if (!SLUG_RE.test(selected.slug)) {
      toast.error('Некорректный идентификатор скрипта')
      return
    }
    setSaving(true)
    try {
      await api.savePluginScript(selected.kind, selected.slug, code)
      setSavedCode(code)
      setSelected({ ...selected, isNew: false })
      await refetchKind(selected.kind)
      queryClient.invalidateQueries({ queryKey: ['algorithms'] })
      queryClient.invalidateQueries({ queryKey: ['wrappers'] })
      toast.success('Сохранено')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось сохранить скрипт')
    } finally {
      setSaving(false)
    }
  }, [selected, code, refetchKind, queryClient])

  const handleDelete = useCallback(async () => {
    if (!selected || selected.isNew) return
    if (!window.confirm(`Удалить «${selected.slug}»?`)) return
    try {
      await api.deletePluginScript(selected.kind, selected.slug)
      await refetchKind(selected.kind)
      queryClient.invalidateQueries({ queryKey: ['algorithms'] })
      queryClient.invalidateQueries({ queryKey: ['wrappers'] })
      setSelected(null)
      setCode('')
      setSavedCode('')
      toast.success('Скрипт удалён')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось удалить скрипт')
    }
  }, [selected, refetchKind, queryClient])

  const selectedSection = useMemo(
    () => (selected ? SECTIONS.find((s) => s.kind === selected.kind) : undefined),
    [selected],
  )
  const filteredTemplates = useMemo(
    () => templates.filter((t) => t.kind === newDialog?.templateKind),
    [templates, newDialog],
  )

  return (
    <div className="flex h-full">
      <aside className="flex w-72 shrink-0 flex-col border-r border-border">
        <div className="border-b border-border px-4 py-3">
          <h2 className="flex items-center gap-2 text-sm font-semibold">
            <Puzzle className="h-4 w-4 text-primary" /> Свои плагины
          </h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Пишите свои алгоритмы и reward-функции — они появятся в дизайнере рядом со встроенными.
          </p>
        </div>
        <ScrollArea className="flex-1">
          <div className="space-y-4 p-3">
            {SECTIONS.map((section) => {
              const items = scriptsByKind[section.kind]
              return (
                <div key={section.kind}>
                  <div className="mb-1.5 flex items-center justify-between px-1">
                    <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      <section.icon className="h-3.5 w-3.5" />
                      {section.title}
                    </div>
                    <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => openNewDialog(section)}>
                      <Plus className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                  <div className="space-y-0.5">
                    {items.length === 0 && (
                      <p className="px-1 py-1 text-xs text-muted-foreground">Пока нет скриптов</p>
                    )}
                    {items.map((item) => {
                      const isActive = selected?.kind === section.kind && selected.slug === item.slug
                      return (
                        <button
                          key={item.slug}
                          onClick={() => handleSelect(section.kind, item.slug)}
                          className={`flex w-full flex-col items-start rounded-md px-2 py-1.5 text-left text-sm transition-colors ${
                            isActive ? 'bg-accent text-accent-foreground' : 'hover:bg-accent/50'
                          }`}
                        >
                          <span className="flex w-full items-center gap-1.5 truncate">
                            {item.broken && <AlertTriangle className="h-3 w-3 shrink-0 text-destructive" />}
                            <span className="truncate">{item.name}</span>
                          </span>
                          <span className="truncate text-[11px] text-muted-foreground">{item.slug}</span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </ScrollArea>
      </aside>

      <div className="flex flex-1 flex-col">
        {!selected ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-muted-foreground">
            <Puzzle className="h-8 w-8" />
            <p className="text-sm">Выберите скрипт слева или создайте новый</p>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-3 border-b border-border px-4 py-2.5">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="truncate font-mono text-sm">{selected.slug}</span>
                  {selectedSection && (
                    <Badge variant="secondary" className="text-[10px]">{selectedSection.title}</Badge>
                  )}
                  {dirty && <Badge variant="outline" className="text-[10px]">не сохранено</Badge>}
                </div>
              </div>
              <Button variant="outline" size="sm" onClick={handleValidate} disabled={validating || loadingCode}>
                {validating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
                Проверить
              </Button>
              <Button size="sm" onClick={handleSave} disabled={saving || loadingCode || !dirty}>
                {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                Сохранить
              </Button>
              {!selected.isNew && (
                <Button variant="ghost" size="icon" onClick={handleDelete} className="text-destructive">
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              )}
            </div>

            {validateResult && (
              <div
                className={`mx-4 mt-3 rounded-lg border px-3 py-2 text-xs ${
                  validateResult.ok
                    ? 'border-success/30 bg-success/10 text-success'
                    : 'border-destructive/30 bg-destructive/10 text-destructive'
                }`}
              >
                {validateResult.ok ? (
                  'Скрипт соответствует контракту, dry-run прошёл успешно.'
                ) : (
                  <div className="space-y-1">
                    <p className="font-medium">{validateResult.error}</p>
                    {validateResult.traceback && (
                      <pre className="max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[10px] opacity-90">
                        {validateResult.traceback}
                      </pre>
                    )}
                  </div>
                )}
              </div>
            )}

            <div className="flex-1 overflow-hidden p-4 pt-3">
              {loadingCode ? (
                <div className="flex h-full items-center justify-center">
                  <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                </div>
              ) : (
                <div className="h-full overflow-hidden rounded-lg border border-border">
                  <Editor
                    height="100%"
                    language="python"
                    theme="vs-dark"
                    value={code}
                    onChange={(value) => setCode(value ?? '')}
                    options={{
                      minimap: { enabled: false },
                      fontSize: 13,
                      scrollBeyondLastLine: false,
                      tabSize: 4,
                    }}
                  />
                </div>
              )}
            </div>
          </>
        )}
      </div>

      <Dialog open={!!newDialog} onClose={() => setNewDialog(null)} title={`Новый скрипт: ${newDialog?.title ?? ''}`}>
        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label className="text-xs">Идентификатор (slug)</Label>
            <Input
              value={newSlug}
              onChange={(e) => setNewSlug(e.target.value.toLowerCase())}
              placeholder="my_algorithm"
            />
            {newSlug && !SLUG_RE.test(newSlug) && (
              <p className="text-[11px] text-destructive">Только a-z, 0-9, «-», «_», начиная с буквы/цифры</p>
            )}
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">Шаблон</Label>
            <Select
              value={newTemplateId}
              onChange={(e) => setNewTemplateId(e.target.value)}
              options={filteredTemplates.map((t) => ({ value: t.id, label: t.label }))}
            />
          </div>
          <Button className="w-full" onClick={handleCreate} disabled={!SLUG_RE.test(newSlug) || !newTemplateId}>
            Создать
          </Button>
        </div>
      </Dialog>
    </div>
  )
}

import { useState } from 'react'
import { toast } from 'sonner'
import { Dialog } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { api } from '@/api/client'
import type { NetworkFamily, NetworkSpec } from '@/api/types'

function slugify(name: string): string {
  const slug = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 64)
  return slug || 'network'
}

export interface SaveArchitectureDialogProps {
  open: boolean
  onClose: () => void
  family: NetworkFamily
  spec: NetworkSpec
  defaultName?: string
  onSaved?: (slug: string) => void
}

/** Shared by the Training Monitor and Model Zoo — takes a fully-resolved
 * `network.json` snapshot (see `rl_core/netbuilder_store.py`) and `PUT`s it
 * into the permanent Network Builder catalog under a chosen slug, so a
 * one-off inline/quick-edited architecture from a past run becomes pickable
 * from the "Архитектура сети" dropdown in any future experiment. */
export function SaveArchitectureDialog({ open, onClose, family, spec, defaultName, onSaved }: SaveArchitectureDialogProps) {
  const [name, setName] = useState(defaultName ?? '')
  const [slug, setSlug] = useState(defaultName ? slugify(defaultName) : '')
  const [slugTouched, setSlugTouched] = useState(false)
  const [saving, setSaving] = useState(false)

  const handleNameChange = (value: string) => {
    setName(value)
    if (!slugTouched) setSlug(slugify(value))
  }

  const handleSave = async () => {
    if (!name.trim() || !slug.trim()) return
    setSaving(true)
    try {
      await api.saveNetwork(slug.trim(), { name: name.trim(), description: '', family, spec })
      toast.success('Архитектура сохранена в каталог сетей')
      onSaved?.(slug.trim())
      onClose()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось сохранить архитектуру')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open={open} onClose={onClose} title="Сохранить архитектуру сети">
      <div className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Архитектура появится в конструкторе сетей и будет доступна для выбора в любом новом эксперименте с совместимым алгоритмом.
        </p>
        <div className="space-y-1">
          <Label className="text-xs">Название</Label>
          <Input value={name} onChange={(e) => handleNameChange(e.target.value)} placeholder="Моя сеть" />
        </div>
        <div className="space-y-1">
          <Label className="text-xs">Идентификатор (slug)</Label>
          <Input
            value={slug}
            onChange={(e) => {
              setSlugTouched(true)
              setSlug(e.target.value)
            }}
            placeholder="my-network"
          />
        </div>
        <Button className="w-full" onClick={handleSave} disabled={!name.trim() || !slug.trim() || saving}>
          {saving ? 'Сохранение…' : 'Сохранить'}
        </Button>
      </div>
    </Dialog>
  )
}

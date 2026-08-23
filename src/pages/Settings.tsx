import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Loader2, FileText, Container, RefreshCw } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { useDockerStore } from '@/stores/dockerStore'
import { useSystemInfo } from '@/api/hooks'

const MODES = [
  { id: 'auto' as const, label: 'Auto', desc: 'Сначала Docker, если не вышло — нативный Python' },
  { id: 'docker' as const, label: 'Docker', desc: 'Только контейнер (нужен установленный Docker)' },
  { id: 'native' as const, label: 'Native', desc: 'Только локальный Python-процесс' },
]

export function SettingsPage() {
  const {
    config, backendMode, backendPort, loadConfig, setBackendMode, setDockerGpu, restartBackend,
  } = useDockerStore()
  const { data: sysInfo } = useSystemInfo()
  const [version, setVersion] = useState<string>('')
  const [platform, setPlatform] = useState<string>('')
  const [restarting, setRestarting] = useState(false)

  useEffect(() => {
    loadConfig()
    window.electronAPI?.app.version().then(setVersion)
    window.electronAPI?.app.platform().then(setPlatform)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleRestart = async () => {
    setRestarting(true)
    await restartBackend()
    setRestarting(false)
    toast.success('Backend перезапущен')
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-8">
      <div className="space-y-1">
        <h2 className="text-2xl font-semibold">Настройки</h2>
        <p className="text-sm text-muted-foreground">Режим бэкенда, GPU и информация о системе.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Backend</CardTitle>
          <CardDescription>Текущий режим: <Badge variant="outline" className="ml-1">{backendMode}</Badge> · порт {backendPort}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {MODES.map((m) => (
              <button
                key={m.id}
                onClick={() => setBackendMode(m.id)}
                className={`rounded-lg border p-3 text-left transition-colors ${
                  config?.backendMode === m.id ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/40'
                }`}
              >
                <div className="text-sm font-medium">{m.label}</div>
                <div className="mt-1 text-xs text-muted-foreground">{m.desc}</div>
              </button>
            ))}
          </div>

          <div className="flex items-center justify-between rounded-lg border border-border p-3">
            <div className="flex items-center gap-2">
              <Container className="h-4 w-4 text-muted-foreground" />
              <div>
                <div className="text-sm font-medium">GPU в Docker</div>
                <div className="text-xs text-muted-foreground">Передавать --gpus all в контейнер (нужен NVIDIA Container Toolkit)</div>
              </div>
            </div>
            <Switch checked={config?.dockerGpu ?? false} onCheckedChange={setDockerGpu} />
          </div>

          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleRestart} disabled={restarting}>
              {restarting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
              Перезапустить backend
            </Button>
            <Button variant="ghost" size="sm" onClick={() => window.electronAPI?.backend.openLogs()}>
              <FileText className="h-3.5 w-3.5" /> Логи
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Система</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
          <InfoRow label="Приложение" value={version || '—'} />
          <InfoRow label="Платформа" value={platform || sysInfo?.platform || '—'} />
          <InfoRow label="CPU cores" value={String(sysInfo?.cpu_count ?? '—')} />
          <InfoRow label="CUDA" value={sysInfo?.torch_cuda_available ? 'доступна' : 'нет'} />
        </CardContent>
      </Card>

      {sysInfo?.gpus && sysInfo.gpus.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">GPU</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {sysInfo.gpus.map((gpu) => (
              <div key={gpu.id} className="flex items-center justify-between text-sm">
                <span>{gpu.name}</span>
                <span className="text-muted-foreground">
                  {gpu.memory_used_mb.toFixed(0)} / {gpu.memory_total_mb.toFixed(0)} MB · {gpu.utilization ?? '—'}%
                </span>
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <Label className="text-xs text-muted-foreground">{label}</Label>
      <div className="font-medium">{value}</div>
    </div>
  )
}

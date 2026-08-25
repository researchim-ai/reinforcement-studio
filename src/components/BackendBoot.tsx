import { useEffect, useMemo, useState } from 'react'
import { useDockerStore } from '@/stores/dockerStore'
import {
  Loader2, Server, AlertCircle, RefreshCw, FileText, Package, Container,
  Heart, PlayCircle, Cog, Cpu, Zap,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { BootPhase, DetectedGpu } from '@/types/electron'
import type { SetupDevice } from '@/stores/dockerStore'

interface BackendBootProps {
  children: React.ReactNode
}

function phaseLabel(phase: BootPhase): { title: string; detail: string; icon: React.ReactNode } {
  switch (phase.phase) {
    case 'starting':
      return { title: 'Запуск...', detail: 'Инициализация', icon: <Cog className="h-6 w-6" /> }
    case 'awaiting-setup':
      return { title: 'Первый запуск', detail: 'Выберите, на чём считать', icon: <Cpu className="h-6 w-6" /> }
    case 'checking-docker':
      return { title: 'Поиск Docker', detail: 'Проверяю, доступен ли Docker daemon', icon: <Container className="h-6 w-6" /> }
    case 'docker-unavailable':
      return { title: 'Docker недоступен', detail: `${phase.detail ?? 'Переключаюсь на нативный Python'}`, icon: <Container className="h-6 w-6" /> }
    case 'docker-no-image':
      return { title: 'Docker образ отсутствует', detail: 'Сейчас соберу его — это может занять несколько минут при первом запуске', icon: <Package className="h-6 w-6" /> }
    case 'building-image':
      return { title: 'Сборка Docker образа', detail: phase.line ?? 'Собираю контейнер для RL-бэкенда', icon: <Package className="h-6 w-6" /> }
    case 'starting-container':
      return { title: 'Запуск контейнера', detail: 'Поднимаю контейнер бэкенда', icon: <PlayCircle className="h-6 w-6" /> }
    case 'waiting-container-health':
      return { title: 'Ожидание готовности', detail: `FastAPI внутри контейнера запускается (попытка ${phase.attempt})`, icon: <Heart className="h-6 w-6" /> }
    case 'creating-venv':
      return { title: 'Настройка Python-окружения', detail: phase.line ?? 'Создаю изолированное виртуальное окружение (первый запуск)', icon: <Package className="h-6 w-6" /> }
    case 'installing-dependencies':
      return { title: 'Установка зависимостей', detail: phase.line ?? 'Ставлю gymnasium / stable-baselines3 / torch — это может занять несколько минут при первом запуске', icon: <Package className="h-6 w-6" /> }
    case 'starting-python':
      return { title: 'Запуск Python', detail: 'Нативный режим — запускаю uvicorn', icon: <Cpu className="h-6 w-6" /> }
    case 'python-starting':
      return { title: 'Python запускается', detail: phase.line ?? 'Ожидание готовности uvicorn', icon: <Cpu className="h-6 w-6" /> }
    case 'ready':
      return { title: 'Готово', detail: `Backend (${phase.mode}) на порту ${phase.port}`, icon: <Server className="h-6 w-6" /> }
    case 'failed':
      return { title: 'Не удалось запустить backend', detail: phase.error, icon: <AlertCircle className="h-6 w-6" /> }
    default:
      return { title: 'Инициализация', detail: '', icon: <Cog className="h-6 w-6" /> }
  }
}

function isFailure(phase: BootPhase): boolean {
  return phase.phase === 'failed'
}

export function BackendBoot({ children }: BackendBootProps) {
  const {
    backendOnline, backendError, backendPort, bootPhase, checkBackend, restartBackend,
    subscribeToBootPhase, setBackendMode, config, loadConfig, chooseSetupDevice, choosingSetup,
  } = useDockerStore()
  // Both toggles get set together by chooseSetupDevice, so either is a
  // faithful "what did the user pick" signal for the persistent selector —
  // "GPU" only shows unselected once both are genuinely off.
  const gpuChosen = Boolean(config?.nativeGpu || config?.dockerGpu)
  const [attempts, setAttempts] = useState(0)
  const [buildLog, setBuildLog] = useState<string[]>([])
  const [timeout20, setTimeout20] = useState(false)

  useEffect(() => {
    loadConfig()
    const unsub = subscribeToBootPhase()
    checkBackend()
    return unsub
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (backendOnline) return
    let cancelled = false
    const tick = async () => {
      if (cancelled) return
      await checkBackend()
      if (cancelled) return
      setAttempts((a) => a + 1)
      if (!useDockerStore.getState().backendOnline) {
        setTimeout(tick, 1500)
      }
    }
    tick()
    return () => { cancelled = true }
  }, [backendOnline, checkBackend])

  // "Stuck" detection: only flag as failed if nothing has happened for a
  // while. Long-running phases (installing deps, building the Docker image)
  // reset this timer on every log line, since they legitimately take minutes
  // on a first run.
  useEffect(() => {
    if (backendOnline) return
    setTimeout20(false)
    const isLongRunning = ['installing-dependencies', 'creating-venv', 'building-image', 'waiting-container-health'].includes(bootPhase.phase)
    const timer = setTimeout(() => setTimeout20(true), isLongRunning ? 45_000 : 20_000)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bootPhase, backendOnline])

  useEffect(() => {
    if ((bootPhase.phase === 'building-image' || bootPhase.phase === 'creating-venv' || bootPhase.phase === 'installing-dependencies') && bootPhase.line) {
      setBuildLog((prev) => [...prev.slice(-200), bootPhase.line!])
    }
  }, [bootPhase])

  const { title, detail, icon } = useMemo(() => phaseLabel(bootPhase), [bootPhase])
  const failed = isFailure(bootPhase) || (timeout20 && !backendOnline)

  if (backendOnline) return <>{children}</>

  if (bootPhase.phase === 'awaiting-setup') {
    return <SetupChoice gpus={bootPhase.gpus} busy={choosingSetup} onChoose={chooseSetupDevice} />
  }

  const isBuilding = ['building-image', 'creating-venv', 'installing-dependencies'].includes(bootPhase.phase)
  const isDocker = ['checking-docker', 'docker-no-image', 'building-image', 'starting-container', 'waiting-container-health'].includes(bootPhase.phase)

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-background p-8 overflow-auto">
      <div className="flex flex-col items-center gap-6 max-w-2xl w-full text-center">
        <div className="relative">
          <div className={`flex h-20 w-20 items-center justify-center rounded-2xl ${failed ? 'bg-destructive/10' : 'bg-primary/10'}`}>
            <div className={failed ? 'text-destructive' : 'text-primary'}>
              {icon}
            </div>
          </div>
          {!failed && (
            <Loader2 className="absolute -bottom-1 -right-1 h-6 w-6 animate-spin text-primary" />
          )}
        </div>

        <div className="space-y-2">
          <h1 className="text-xl font-semibold">{title}</h1>
          <p className="text-sm text-muted-foreground max-w-md">
            {failed ? (backendError ?? detail) : detail}
          </p>
        </div>

        {!failed && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <div className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
            <span>
              {isDocker ? 'Docker mode' : 'Native mode'}
              {/* The attempt counter tracks health-check polling (every 1.5s),
                  not install retries — showing it during venv/deps setup, which
                  legitimately takes minutes, made a normal one-shot install look
                  like it was failing and retrying over and over. */}
              {!isBuilding && ` · попытка ${attempts}`}
            </span>
          </div>
        )}

        {isBuilding && buildLog.length > 0 && (
          <div className="w-full max-w-2xl rounded-lg border border-border bg-black/40 p-3 text-left text-xs font-mono text-muted-foreground max-h-56 overflow-auto">
            {buildLog.slice(-20).map((line, i) => (
              <div key={i} className="truncate">{line}</div>
            ))}
          </div>
        )}

        {failed && (
          <div className="flex gap-2 flex-wrap justify-center">
            <Button onClick={() => { setTimeout20(false); setAttempts(0); restartBackend() }}>
              <RefreshCw className="h-4 w-4" />
              Перезапустить
            </Button>
            <Button
              variant="ghost"
              onClick={() => window.electronAPI?.backend.openLogs()}
            >
              <FileText className="h-4 w-4" />
              Логи
            </Button>
          </div>
        )}

        {/* Neither of these two is gated on `failed` — a slow Docker
            build/pip install (Auto mode legitimately preferring Docker
            whenever it's newly available, e.g. right after this Dockerfile
            packaging bug got fixed) used to leave no way to change your
            mind short of force-quitting the app. Both always show, and
            both actually interrupt whatever's currently installing instead
            of silently queuing behind it. */}
        {config && (
          <div className="flex flex-col gap-4 items-center w-full max-w-md rounded-lg border border-border/60 bg-card/40 p-4">
            <div className="flex flex-col gap-2 items-center w-full">
              <p className="text-xs text-muted-foreground">На чём считать:</p>
              <div className="flex gap-2">
                <Button
                  variant={!gpuChosen ? 'default' : 'outline'}
                  size="sm"
                  disabled={choosingSetup}
                  onClick={() => { setTimeout20(false); setAttempts(0); setBuildLog([]); chooseSetupDevice('cpu') }}
                >
                  <Cpu className="h-3.5 w-3.5" /> CPU
                </Button>
                <Button
                  variant={gpuChosen ? 'default' : 'outline'}
                  size="sm"
                  disabled={choosingSetup}
                  onClick={() => { setTimeout20(false); setAttempts(0); setBuildLog([]); chooseSetupDevice('gpu') }}
                >
                  <Zap className="h-3.5 w-3.5" /> GPU (CUDA)
                </Button>
              </div>
            </div>

            <div className="flex flex-col gap-2 items-center w-full">
              <p className="text-xs text-muted-foreground">Режим backend:</p>
              <div className="flex gap-2">
                {(['native', 'docker', 'auto'] as const).map((m) => (
                  <Button
                    key={m}
                    variant={config.backendMode === m ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => { setTimeout20(false); setAttempts(0); setBuildLog([]); setBackendMode(m) }}
                  >
                    {m === 'native' ? 'Native' : m === 'docker' ? 'Docker' : 'Auto'}
                  </Button>
                ))}
              </div>
            </div>

            <p className="text-[10px] text-muted-foreground/60 text-center">
              Любое изменение сразу перезапускает backend и переустанавливает то, что нужно под новый выбор.
            </p>
          </div>
        )}

        {failed && (
          <details className="w-full max-w-xl text-left">
            <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
              Ручной запуск
            </summary>
            <div className="mt-2 rounded-lg border border-border bg-card p-4 text-xs font-mono text-muted-foreground">
              <div className="space-y-1">
                <div><span className="text-primary">$</span> pip install -r backend/requirements.txt -r rl_core/requirements.txt</div>
                <div><span className="text-primary">$</span> uvicorn backend.api:app --port {backendPort}</div>
              </div>
            </div>
          </details>
        )}
      </div>
    </div>
  )
}

interface SetupChoiceProps {
  gpus: DetectedGpu[]
  busy: boolean
  onChoose: (device: SetupDevice) => void
}

function SetupChoice({ gpus, busy, onChoose }: SetupChoiceProps) {
  return (
    <div className="flex h-screen w-screen items-center justify-center bg-background p-8 overflow-auto">
      <div className="flex flex-col items-center gap-6 max-w-2xl w-full text-center">
        <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-primary/10">
          <Cpu className="h-10 w-10 text-primary" />
        </div>

        <div className="space-y-2">
          <h1 className="text-xl font-semibold">Первый запуск — на чём считать?</h1>
          <p className="text-sm text-muted-foreground max-w-lg">
            Приложение ещё ничего не устанавливало. Выберите CPU или GPU — от этого зависит,
            какая версия PyTorch будет поставлена (CPU-версия компактная и ставится быстро,
            CUDA-версия больше, но нужна для быстрого обучения на больших досках/сетях).
            Позже это можно поменять в Settings.
          </p>
        </div>

        {gpus.length > 0 ? (
          <div className="w-full max-w-lg space-y-1.5 rounded-lg border border-border bg-card p-3 text-left">
            <p className="text-xs font-medium text-muted-foreground">Обнаружены видеокарты NVIDIA:</p>
            {gpus.map((gpu) => (
              <div key={gpu.index} className="flex items-center justify-between text-sm">
                <span>{gpu.name}</span>
                <span className="text-muted-foreground">{(gpu.memoryTotalMb / 1024).toFixed(1)} ГБ</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground/70 max-w-lg">
            Видеокарты NVIDIA не обнаружены (или не найден nvidia-smi) — если у вас всё же есть GPU,
            можно выбрать GPU-вариант вручную.
          </p>
        )}

        {busy ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Сохраняю выбор и запускаю установку...
          </div>
        ) : (
          <div className="grid w-full max-w-lg grid-cols-1 gap-3 sm:grid-cols-2">
            <button
              onClick={() => onChoose('cpu')}
              className="flex flex-col items-center gap-2 rounded-xl border border-border p-5 text-center transition-colors hover:border-primary/50 hover:bg-primary/5"
            >
              <Cpu className="h-7 w-7 text-primary" />
              <div className="text-sm font-semibold">CPU</div>
              <div className="text-xs text-muted-foreground">
                Быстрая установка (~200 МБ). Хватает для классики Gymnasium и AlphaZero на маленьких досках.
              </div>
            </button>
            <button
              onClick={() => onChoose('gpu')}
              className="flex flex-col items-center gap-2 rounded-xl border border-border p-5 text-center transition-colors hover:border-primary/50 hover:bg-primary/5"
            >
              <Zap className="h-7 w-7 text-primary" />
              <div className="text-sm font-semibold">GPU (CUDA)</div>
              <div className="text-xs text-muted-foreground">
                Более долгая установка (несколько ГБ). Нужен драйвер NVIDIA — для self-play на больших досках,
                Atari/MuJoCo с CNN.
              </div>
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

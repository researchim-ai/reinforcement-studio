import { Link } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useRuns } from '@/api/hooks'
import { useSystemInfo } from '@/api/hooks'
import { Workflow, LineChart, Boxes, Cpu, MemoryStick, Gauge } from 'lucide-react'
import { RunFolderLink } from '@/components/monitor/RunFolderLink'

// AlphaZero Arena card is temporarily removed — see the comment next to its
// lazy import in App.tsx for how to bring it back.
const QUICK_LINKS = [
  { to: '/designer', icon: Workflow, title: 'Дизайнер экспериментов', desc: 'Собери граф env → wrappers → алгоритм и запусти обучение' },
  { to: '/environments', icon: Boxes, title: 'Среды', desc: 'Gymnasium классика и настольные игры для AlphaZero' },
  { to: '/monitor', icon: LineChart, title: 'Мониторинг', desc: 'Живые графики и управление запусками' },
]

export function Dashboard() {
  const { data: runsData } = useRuns()
  const { data: sysInfo } = useSystemInfo()
  const runs = runsData?.runs ?? []
  const activeRuns = runs.filter((r) => r.running)

  return (
    <div className="mx-auto max-w-6xl space-y-8 p-8">
      <div className="space-y-1">
        <h2 className="text-2xl font-semibold">Добро пожаловать в Reinforcement Studio</h2>
        <p className="text-sm text-muted-foreground">
          Локальная лаборатория для reinforcement learning: классика на Gymnasium (DQN/PPO/A2C) и
          AlphaZero self-play для настольных игр — без единой строчки кода.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card>
          <CardContent className="flex items-center gap-3 p-4">
            <Gauge className="h-5 w-5 text-primary" />
            <div>
              <div className="text-lg font-semibold">{activeRuns.length}</div>
              <div className="text-xs text-muted-foreground">активных запусков</div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="flex items-center gap-3 p-4">
            <Cpu className="h-5 w-5 text-primary" />
            <div>
              <div className="text-lg font-semibold">{sysInfo?.cpu_count ?? '—'}</div>
              <div className="text-xs text-muted-foreground">CPU cores</div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="flex items-center gap-3 p-4">
            <MemoryStick className="h-5 w-5 text-primary" />
            <div>
              <div className="text-lg font-semibold">
                {sysInfo?.torch_cuda_available ? 'CUDA' : 'CPU-only'}
              </div>
              <div className="text-xs text-muted-foreground">
                {sysInfo?.gpus?.length ? `${sysInfo.gpus.length} GPU обнаружен` : 'вычисления'}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div>
        <h3 className="mb-3 text-sm font-medium text-muted-foreground">Быстрый старт</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {QUICK_LINKS.map((link) => (
            <Link key={link.to} to={link.to}>
              <Card className="h-full transition-colors hover:border-primary/50">
                <CardHeader className="flex-row items-center gap-3 space-y-0">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <link.icon className="h-5 w-5" />
                  </div>
                  <div>
                    <CardTitle className="text-sm">{link.title}</CardTitle>
                    <CardDescription>{link.desc}</CardDescription>
                  </div>
                </CardHeader>
              </Card>
            </Link>
          ))}
        </div>
      </div>

      <div>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-medium text-muted-foreground">Последние запуски</h3>
          <Link to="/monitor">
            <Button variant="ghost" size="sm">Все запуски →</Button>
          </Link>
        </div>
        {runs.length === 0 ? (
          <Card>
            <CardContent className="p-6 text-center text-sm text-muted-foreground">
              Пока нет запусков. Собери первый эксперимент в{' '}
              <Link to="/designer" className="text-primary hover:underline">Дизайнере экспериментов</Link>.
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-2">
            {runs.slice(0, 5).map((run) => (
              <Card key={run.run_id}>
                <CardContent className="flex items-center justify-between gap-3 p-4">
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium">{run.name}</div>
                    <div className="text-xs text-muted-foreground">
                      {run.environment_id} · {run.algorithm_id}
                    </div>
                    <RunFolderLink runId={run.run_id} runDir={run.run_dir} className="mt-1" />
                  </div>
                  <Badge variant={run.running ? 'success' : run.status === 'completed' ? 'secondary' : 'outline'}>
                    {run.status}
                  </Badge>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

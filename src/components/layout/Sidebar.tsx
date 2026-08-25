import { useEffect } from 'react'
import { NavLink } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  Workflow,
  LineChart,
  Boxes,
  Swords,
  Archive,
  Settings,
  Activity,
  Puzzle,
  Network,
  GraduationCap,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { useDockerStore } from '@/stores/dockerStore'
import appIcon from '@/assets/icon.png'

const navItems = [
  { path: '/', icon: LayoutDashboard, labelKey: 'nav.dashboard' },
  { path: '/designer', icon: Workflow, labelKey: 'nav.designer' },
  { path: '/monitor', icon: LineChart, labelKey: 'nav.monitor' },
  { path: '/environments', icon: Boxes, labelKey: 'nav.environments' },
  { path: '/arena', icon: Swords, labelKey: 'nav.arena' },
  { path: '/models', icon: Archive, labelKey: 'nav.models' },
  { path: '/plugins', icon: Puzzle, labelKey: 'nav.plugins' },
  { path: '/network-builder', icon: Network, labelKey: 'nav.networkBuilder' },
  { path: '/academy', icon: GraduationCap, labelKey: 'nav.academy' },
  { path: '/settings', icon: Settings, labelKey: 'nav.settings' },
]

export function Sidebar() {
  const { t } = useTranslation()
  const backendOnline = useDockerStore((s) => s.backendOnline)
  const checkBackend = useDockerStore((s) => s.checkBackend)

  useEffect(() => {
    checkBackend()
    const interval = setInterval(checkBackend, 5000)
    return () => clearInterval(interval)
  }, [checkBackend])

  return (
    <aside className="flex h-full w-[230px] flex-col border-r border-border bg-card">
      <div className="flex items-center gap-3 px-5 py-5">
        {/* Imported (not a "/icon.png" public path): an absolute path
            resolves against the filesystem root under the packaged app's
            file:// loadFile(), not the app's own dist folder — same class of
            bug as BrowserRouter. Importing it lets Vite rewrite the URL
            relative to the app, like it already does for JS/CSS. */}
        <img src={appIcon} alt="Reinforcement Studio" className="h-8 w-8 rounded-lg" />
        <div className="flex flex-col">
          <span className="text-sm font-semibold">Reinforcement Studio</span>
          <span className="text-[10px] text-muted-foreground">Reinforcement Learning Lab</span>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 px-3 py-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
                isActive
                  ? 'bg-accent text-accent-foreground font-medium'
                  : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
              )
            }
          >
            <item.icon className="h-4 w-4 shrink-0" />
            <span className="truncate">{t(item.labelKey)}</span>
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <Activity className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">Backend</span>
          <Badge
            variant={backendOnline ? 'success' : 'secondary'}
            className="ml-auto text-[10px] px-1.5 py-0"
          >
            {backendOnline ? 'Online' : 'Offline'}
          </Badge>
        </div>
      </div>
    </aside>
  )
}

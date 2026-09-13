import { useTranslation } from 'react-i18next'
import { useLocation } from 'react-router-dom'

const routeTitles: Record<string, string> = {
  '/': 'nav.dashboard',
  '/designer': 'nav.designer',
  '/monitor': 'nav.monitor',
  '/environments': 'nav.environments',
  '/models': 'nav.models',
  '/sweeps': 'nav.sweeps',
  '/plugins': 'nav.plugins',
  '/network-builder': 'nav.networkBuilder',
  '/world-models': 'nav.worldModels',
  '/scene-builder': 'nav.sceneBuilder',
  '/academy': 'nav.academy',
  '/settings': 'nav.settings',
}

export function Header() {
  const { t } = useTranslation()
  const location = useLocation()
  const titleKey = routeTitles[location.pathname] ?? 'nav.dashboard'

  return (
    <header className="flex h-12 items-center justify-between border-b border-border px-6 app-drag-region">
      <h1 className="text-sm font-semibold">{t(titleKey)}</h1>
    </header>
  )
}

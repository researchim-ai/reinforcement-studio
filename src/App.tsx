import { lazy, Suspense, type ComponentType } from 'react'
import { Routes, Route } from 'react-router-dom'
import { Layout } from '@/components/layout/Layout'
import { BackendBoot } from '@/components/BackendBoot'
import { Loader2 } from 'lucide-react'

// A dynamic import can fail when the dev server rebuilt the module graph (HMR)
// or after a new deploy — the browser holds a stale chunk URL. Self-heal by
// reloading the window once before surfacing the error to the boundary.
function lazyWithReload<T extends ComponentType<unknown>>(
  factory: () => Promise<{ default: T }>,
) {
  return lazy(async () => {
    try {
      const mod = await factory()
      sessionStorage.removeItem('chunk-reload')
      return mod
    } catch (err) {
      if (!sessionStorage.getItem('chunk-reload')) {
        sessionStorage.setItem('chunk-reload', '1')
        window.location.reload()
        return new Promise<{ default: T }>(() => {})
      }
      throw err
    }
  })
}

const Dashboard = lazyWithReload(() => import('@/pages/Dashboard').then((m) => ({ default: m.Dashboard })))
const ExperimentDesigner = lazyWithReload(() => import('@/pages/ExperimentDesigner').then((m) => ({ default: m.ExperimentDesigner })))
const TrainingMonitor = lazyWithReload(() => import('@/pages/TrainingMonitor').then((m) => ({ default: m.TrainingMonitor })))
const Environments = lazyWithReload(() => import('@/pages/Environments').then((m) => ({ default: m.Environments })))
const AlphaZeroArena = lazyWithReload(() => import('@/pages/AlphaZeroArena').then((m) => ({ default: m.AlphaZeroArena })))
const ModelZoo = lazyWithReload(() => import('@/pages/ModelZoo').then((m) => ({ default: m.ModelZoo })))
const PluginsPage = lazyWithReload(() => import('@/pages/Plugins').then((m) => ({ default: m.PluginsPage })))
const SettingsPage = lazyWithReload(() => import('@/pages/Settings').then((m) => ({ default: m.SettingsPage })))

function PageFallback() {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
    </div>
  )
}

export default function App() {
  return (
    <BackendBoot>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Suspense fallback={<PageFallback />}><Dashboard /></Suspense>} />
          <Route path="/designer" element={<Suspense fallback={<PageFallback />}><ExperimentDesigner /></Suspense>} />
          <Route path="/monitor" element={<Suspense fallback={<PageFallback />}><TrainingMonitor /></Suspense>} />
          <Route path="/environments" element={<Suspense fallback={<PageFallback />}><Environments /></Suspense>} />
          <Route path="/arena" element={<Suspense fallback={<PageFallback />}><AlphaZeroArena /></Suspense>} />
          <Route path="/models" element={<Suspense fallback={<PageFallback />}><ModelZoo /></Suspense>} />
          <Route path="/plugins" element={<Suspense fallback={<PageFallback />}><PluginsPage /></Suspense>} />
          <Route path="/settings" element={<Suspense fallback={<PageFallback />}><SettingsPage /></Suspense>} />
        </Route>
      </Routes>
    </BackendBoot>
  )
}

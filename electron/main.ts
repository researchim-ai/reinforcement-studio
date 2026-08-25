import { app, BrowserWindow, ipcMain, shell, dialog, Menu, nativeImage } from 'electron'
import path from 'path'
import fs from 'fs'
import net from 'net'
import { spawn, type ChildProcess } from 'child_process'
import { DockerManager } from './docker'
import { loadConfig, saveConfig, type AppConfig, type BackendMode } from './config'
import { detectGpus, detectMaxCudaVersion, pickTorchCudaChannel, type DetectedGpu } from './gpu'
import { ensurePythonEnv } from './pythonEnv'

if (process.platform === 'linux') {
  app.disableHardwareAcceleration()
  app.commandLine.appendSwitch('disable-gpu')
  app.commandLine.appendSwitch('disable-gpu-compositing')
  app.commandLine.appendSwitch('disable-accelerated-2d-canvas')
  // The GPU *process* sandbox (distinct from --disable-gpu, which only turns
  // off GPU acceleration) is the actual, well-documented cause of a fully
  // black — but otherwise perfectly alive — window on Linux with proprietary
  // NVIDIA drivers: the sandboxed GPU process fails to initialize against the
  // NVIDIA userspace driver, so nothing ever gets painted even though the
  // renderer/backend keep running fine. See e.g. electron/electron#4380 and
  // balena-io/etcher#3639.
  app.commandLine.appendSwitch('disable-gpu-sandbox')
  // Chromium crashes its renderer on the tiny /dev/shm many containers/VMs
  // ship (often 64MB) unless told to use /tmp instead — a common cause of a
  // fully black window with no visible error.
  app.commandLine.appendSwitch('disable-dev-shm-usage')
  // Running as root breaks Chromium's setuid sandbox helper (common for
  // AppImages launched from a root shell/container) and silently kills
  // rendering. Only relax the sandbox in that specific case.
  if (process.getuid && process.getuid() === 0) {
    app.commandLine.appendSwitch('no-sandbox')
  }
}

let mainWindow: BrowserWindow | null = null
let dockerManager: DockerManager | null = null
let backendProcess: ChildProcess | null = null
let backendPort = 8000
let backendStartError: string | null = null
let currentMode: 'native' | 'docker' | 'offline' = 'offline'
let booting = false

// Boot phases — streamed to renderer for splash screen feedback
type BootPhase =
  | { phase: 'starting' }
  | { phase: 'awaiting-setup'; gpus: DetectedGpu[] }
  | { phase: 'checking-docker' }
  | { phase: 'docker-unavailable'; detail?: string }
  | { phase: 'docker-no-image' }
  | { phase: 'building-image'; line?: string }
  | { phase: 'starting-container' }
  | { phase: 'waiting-container-health'; attempt: number }
  | { phase: 'starting-python' }
  | { phase: 'creating-venv'; line?: string }
  | { phase: 'installing-dependencies'; line?: string }
  | { phase: 'python-starting'; line?: string }
  | { phase: 'ready'; mode: 'native' | 'docker'; port: number }
  | { phase: 'failed'; error: string }

let currentBootPhase: BootPhase = { phase: 'starting' }

function emitBootPhase(p: BootPhase) {
  currentBootPhase = p
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend:boot-phase', p)
  }
}

const isDev = !!process.env.VITE_DEV_SERVER_URL
const VITE_DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

function getResourcePath(...segments: string[]): string {
  if (isDev) {
    return path.join(__dirname, '..', ...segments)
  }
  return path.join(process.resourcesPath, ...segments)
}

function getIconPath(): string | undefined {
  const candidates = isDev
    ? [
        path.join(__dirname, '..', 'build', 'icons', '512x512.png'),
        path.join(__dirname, '..', 'build', 'icon.png'),
      ]
    : [
        path.join(process.resourcesPath, 'icons', '512x512.png'),
        path.join(process.resourcesPath, 'icon.png'),
      ]
  for (const p of candidates) {
    if (fs.existsSync(p)) return p
  }
  return undefined
}

// ---------------------------------------------------------------------------
// Port helpers
// ---------------------------------------------------------------------------

function findAvailablePort(start = 8000): Promise<number> {
  return new Promise((resolve) => {
    const srv = net.createServer()
    srv.unref()
    srv.on('error', () => resolve(findAvailablePort(start + 1)))
    srv.listen(start, '127.0.0.1', () => {
      const addr = srv.address()
      const port = typeof addr === 'object' && addr ? addr.port : start
      srv.close(() => resolve(port))
    })
  })
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

function createWindow() {
  const iconPath = getIconPath()

  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    title: 'Reinforcement Studio',
    icon: iconPath ? nativeImage.createFromPath(iconPath) : undefined,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    backgroundColor: '#0e0e12',
    show: false,
  })

  mainWindow.once('ready-to-show', () => mainWindow?.show())

  if (VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(VITE_DEV_SERVER_URL)
    mainWindow.webContents.openDevTools({ mode: 'detach' })
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))
    if (process.env.RL_STUDIO_DEBUG === '1') {
      mainWindow.webContents.openDevTools({ mode: 'detach' })
    }
  }

  mainWindow.webContents.on('render-process-gone', (_e, details) => {
    console.error('[renderer] gone:', JSON.stringify(details))
  })
  mainWindow.webContents.on('did-fail-load', (_e, code, desc, url) => {
    console.error(`[renderer] did-fail-load: ${code} ${desc} ${url}`)
  })
  mainWindow.webContents.on('console-message', (_e, level, message, line, sourceId) => {
    console.log(`[renderer console] [${level}] ${message} (${sourceId}:${line})`)
  })

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  mainWindow.webContents.on('did-finish-load', () => {
    emitBootPhase(currentBootPhase)
  })

  mainWindow.on('closed', () => { mainWindow = null })
}

// ---------------------------------------------------------------------------
// Application menu
// ---------------------------------------------------------------------------

function buildAppMenu() {
  const isMac = process.platform === 'darwin'

  const template: Electron.MenuItemConstructorOptions[] = [
    ...(isMac
      ? [{
          label: app.name,
          submenu: [
            { role: 'about' as const },
            { type: 'separator' as const },
            { role: 'services' as const },
            { type: 'separator' as const },
            { role: 'hide' as const },
            { role: 'hideOthers' as const },
            { role: 'unhide' as const },
            { type: 'separator' as const },
            { role: 'quit' as const },
          ],
        }]
      : []),
    {
      label: 'File',
      submenu: [
        isMac ? { role: 'close' as const } : { role: 'quit' as const },
      ],
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectAll' },
      ],
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    {
      label: 'Backend',
      submenu: [
        {
          label: 'Restart Backend',
          click: async () => {
            await stopBackend()
            await startBackend().catch((err) => console.error('Restart failed:', err))
          },
        },
        {
          label: 'Show Backend Logs',
          click: () => {
            const logPath = path.join(app.getPath('logs'), 'backend.log')
            shell.showItemInFolder(logPath)
          },
        },
      ],
    },
    {
      role: 'help',
      submenu: [
        {
          label: 'About',
          click: () => {
            dialog.showMessageBox(mainWindow!, {
              type: 'info',
              title: 'Reinforcement Studio',
              message: `Reinforcement Studio v${app.getVersion()}`,
              detail: 'Desktop studio for designing and running reinforcement learning experiments.',
              icon: getIconPath() ? nativeImage.createFromPath(getIconPath()!) : undefined,
            })
          },
        },
      ],
    },
  ]

  Menu.setApplicationMenu(Menu.buildFromTemplate(template))
}

// ---------------------------------------------------------------------------
// Backend lifecycle — dispatcher based on config.backendMode
// ---------------------------------------------------------------------------

async function startBackend(): Promise<void> {
  if (booting || backendProcess || currentMode === 'docker') return
  booting = true
  backendStartError = null
  emitBootPhase({ phase: 'starting' })

  const config = loadConfig()
  const mode = config.backendMode

  try {
    if (mode === 'docker') {
      const ok = await startBackendDocker(config)
      if (!ok) throw new Error(backendStartError ?? 'Docker start failed')
      return
    }

    if (mode === 'auto') {
      emitBootPhase({ phase: 'checking-docker' })
      const dockerOk = dockerManager ? await dockerManager.isAvailable() : false
      if (dockerOk && dockerManager?.hasDockerfile()) {
        const ok = await startBackendDocker(config)
        if (ok) return
        console.warn('[backend] Docker attempt failed, falling back to native')
      } else {
        emitBootPhase({
          phase: 'docker-unavailable',
          detail: dockerOk ? 'No Dockerfile' : 'Docker daemon not reachable',
        })
      }
    }

    await startBackendNative()
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    backendStartError = msg
    emitBootPhase({ phase: 'failed', error: msg })
    throw err
  } finally {
    booting = false
  }
}

async function startBackendDocker(config: AppConfig): Promise<boolean> {
  if (!dockerManager) return false

  const ok = await dockerManager.isAvailable()
  if (!ok) {
    emitBootPhase({ phase: 'docker-unavailable', detail: 'Docker daemon not reachable' })
    return false
  }

  if (!dockerManager.hasDockerfile(config.dockerGpu)) {
    emitBootPhase({ phase: 'docker-unavailable', detail: config.dockerGpu ? 'Dockerfile.gpu not found' : 'Dockerfile not found' })
    return false
  }

  const existing = await dockerManager.getStatus()

  if (!(await dockerManager.imageExists(config.dockerGpu))) {
    emitBootPhase({ phase: 'docker-no-image' })
    if (!config.dockerAutoBuild) {
      backendStartError = 'Docker image missing and auto-build is disabled'
      return false
    }
    emitBootPhase({ phase: 'building-image' })
    const torchCudaChannel = config.dockerGpu ? pickTorchCudaChannel(detectMaxCudaVersion()) : undefined
    const build = await dockerManager.buildImage(
      (line) => emitBootPhase({ phase: 'building-image', line }),
      config.dockerGpu,
      torchCudaChannel,
    )
    if (!build.success) {
      backendStartError = build.error ?? 'Image build failed'
      return false
    }
  }

  const hostPort = existing.hostPort ?? (await findAvailablePort(8000))
  emitBootPhase({ phase: 'starting-container' })
  const started = await dockerManager.startContainer({ hostPort, gpu: config.dockerGpu })
  if (!started.success) {
    backendStartError = started.error ?? 'Container start failed'
    return false
  }

  backendPort = started.hostPort ?? hostPort
  const healthy = await dockerManager.waitForHealth(backendPort, {
    timeoutMs: 120_000,
    onTick: (attempt) => emitBootPhase({ phase: 'waiting-container-health', attempt }),
  })
  if (!healthy) {
    backendStartError = 'Container did not become healthy in 2 minutes'
    return false
  }

  currentMode = 'docker'
  emitBootPhase({ phase: 'ready', mode: 'docker', port: backendPort })
  return true
}

async function startBackendNative(): Promise<void> {
  const projectRoot = getResourcePath()
  const venvDir = path.join(app.getPath('userData'), 'pyenv')
  const logPath = path.join(app.getPath('logs'), 'backend.log')
  fs.mkdirSync(path.dirname(logPath), { recursive: true })
  const logStream = fs.createWriteStream(logPath, { flags: 'a' })
  logStream.write(`\n--- Session start ${new Date().toISOString()} ---\n`)

  // First run (or after a requirements change, e.g. toggling the "GPU"
  // setting below) — create a private venv and install rl_core/backend
  // dependencies into it. This is what actually fulfils the "installs
  // everything it needs" promise, instead of relying on the user's system
  // Python already having fastapi/gymnasium/torch.
  const nativeGpu = loadConfig().nativeGpu
  const rlCoreRequirements = nativeGpu ? 'requirements-gpu.txt' : 'requirements.txt'
  // requirements-gpu.txt has no --index-url of its own (see that file's
  // comment for why: PyTorch's cuXXX channels get retired over time and a
  // stale pin silently falls through to whatever PyPI's *current* default
  // is, which can need a newer driver than this machine actually has) — so
  // pick the right one for the driver actually detected on *this* boot.
  const extraPipArgs: string[] = []
  if (nativeGpu) {
    const cudaChannel = pickTorchCudaChannel(detectMaxCudaVersion())
    logStream.write(`[env] GPU включён — ставлю torch с канала ${cudaChannel}\n`)
    extraPipArgs.push('--index-url', `https://download.pytorch.org/whl/${cudaChannel}`, '--extra-index-url', 'https://pypi.org/simple')
  }
  let pythonPath: string
  try {
    const result = await ensurePythonEnv(
      venvDir,
      [path.join(projectRoot, 'rl_core', rlCoreRequirements), path.join(projectRoot, 'backend', 'requirements.txt')],
      (phase, line) => {
        logStream.write(`[env:${phase}] ${line ?? ''}\n`)
        emitBootPhase({ phase, line })
      },
      extraPipArgs,
    )
    pythonPath = result.pythonPath
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    backendStartError = msg
    logStream.write(`--- Python env bootstrap failed: ${msg} ---\n`)
    logStream.end()
    throw err instanceof Error ? err : new Error(msg)
  }

  return new Promise((resolve, reject) => {
    emitBootPhase({ phase: 'starting-python' })

    findAvailablePort(8000).then((port) => {
      backendPort = port
      console.log(`[backend] Starting native on port ${backendPort} with ${pythonPath}`)

      backendProcess = spawn(
        pythonPath,
        [
          '-m', 'uvicorn', 'backend.api:app',
          '--host', '127.0.0.1',
          '--port', String(backendPort),
          ...(isDev ? ['--reload'] : []),
        ],
        {
          cwd: projectRoot,
          env: {
            ...process.env,
            PYTHONUNBUFFERED: '1',
            PYTHONPATH: projectRoot,
            // Packaged apps run from a read-only mount (AppImage squashfs,
            // asar-unpacked resources dir, etc.) — rl_core can't create
            // .runs/checkpoints next to itself there. Point it at a writable
            // per-user directory instead. In dev, the project dir is already
            // writable, so let rl_core use its own default location.
            ...(isDev ? {} : { RL_STUDIO_ROOT: path.join(app.getPath('userData'), 'rl_data') }),
          },
          stdio: ['ignore', 'pipe', 'pipe'],
        },
      )

      let started = false
      const recentLines: string[] = []

      const onData = (buf: Buffer) => {
        const text = buf.toString()
        logStream.write(text)
        process.stdout.write(`[backend] ${text}`)
        for (const line of text.split('\n')) {
          if (!line.trim()) continue
          recentLines.push(line.trim())
          if (recentLines.length > 20) recentLines.shift()
          if (!started) emitBootPhase({ phase: 'python-starting', line: line.trim() })
        }
        if (!started && /Uvicorn running|Application startup complete/.test(text)) {
          started = true
          backendStartError = null
          currentMode = 'native'
          emitBootPhase({ phase: 'ready', mode: 'native', port: backendPort })
          resolve()
        }
      }

      backendProcess.stdout?.on('data', onData)
      backendProcess.stderr?.on('data', onData)

      backendProcess.on('error', (err) => {
        console.error('[backend] spawn error:', err.message)
        backendStartError = err.message
        backendProcess = null
        if (!started) reject(err)
      })

      backendProcess.on('exit', (code, signal) => {
        console.log(`[backend] Exited with code=${code} signal=${signal}`)
        logStream.write(`--- Exit code=${code} signal=${signal} ---\n`)
        logStream.end()
        backendProcess = null
        if (!started) {
          const tail = recentLines.slice(-5).join(' / ')
          backendStartError = tail
            ? `Backend exited early (code ${code}): ${tail}`
            : `Backend exited early (code ${code})`
          reject(new Error(backendStartError))
        }
      })

      const startupTimeout = setTimeout(() => {
        if (!started) {
          const err = new Error('Backend did not become ready in 60 seconds')
          backendStartError = err.message
          emitBootPhase({ phase: 'failed', error: err.message })
          try { backendProcess?.kill('SIGTERM') } catch { /* ignore */ }
          reject(err)
        }
      }, 60_000)

      backendProcess.once('exit', () => clearTimeout(startupTimeout))
    })
  })
}

async function stopBackend() {
  if (currentMode === 'docker' && dockerManager) {
    try {
      await dockerManager.stopContainer()
    } catch (err) {
      console.error('[backend] docker stop error:', err)
    }
    currentMode = 'offline'
    return
  }

  if (!backendProcess) return
  console.log('[backend] Stopping native...')
  try {
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', String(backendProcess.pid), '/f', '/t'])
    } else {
      backendProcess.kill('SIGTERM')
      setTimeout(() => backendProcess?.kill('SIGKILL'), 3000)
    }
  } catch (err) {
    console.error('[backend] Error stopping:', err)
  }
  backendProcess = null
  currentMode = 'offline'
}

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------

const gotSingleInstanceLock = app.requestSingleInstanceLock()
if (!gotSingleInstanceLock) {
  app.quit()
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.focus()
    }
  })
}

app.whenReady().then(async () => {
  buildAppMenu()
  dockerManager = new DockerManager(getResourcePath())
  registerIpcHandlers()

  createWindow()

  // First launch ever (or config wiped) — don't silently start installing
  // anything yet. Detect local NVIDIA GPUs and show the user a CPU/GPU
  // picker on the splash screen instead; startBackend() only runs once
  // they've answered (see the 'setup:choose' IPC handler below).
  if (!loadConfig().setupComplete) {
    emitBootPhase({ phase: 'awaiting-setup', gpus: detectGpus() })
  } else {
    startBackend().catch((err) => {
      console.error('[backend] Could not auto-start:', err)
    })
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', async (e) => {
  if (currentMode === 'docker' && dockerManager) {
    e.preventDefault()
    try { await dockerManager.stopContainer() } catch { /* ignore */ }
    currentMode = 'offline'
    app.quit()
    return
  }
  await stopBackend()
})

// ---------------------------------------------------------------------------
// IPC handlers
// ---------------------------------------------------------------------------

function registerIpcHandlers() {
  ipcMain.handle('backend:status', async () => {
    try {
      const res = await fetch(`http://127.0.0.1:${backendPort}/api/system/health`)
      return {
        running: res.ok,
        port: backendPort,
        error: backendStartError,
        mode: currentMode,
        phase: currentBootPhase,
      }
    } catch {
      return {
        running: false,
        port: backendPort,
        error: backendStartError,
        mode: currentMode,
        phase: currentBootPhase,
      }
    }
  })

  ipcMain.handle('backend:restart', async () => {
    await stopBackend()
    try {
      await startBackend()
      return { success: true, port: backendPort, mode: currentMode }
    } catch (err) {
      return { success: false, error: String(err), port: backendPort, mode: currentMode }
    }
  })

  ipcMain.handle('backend:port', () => backendPort)
  ipcMain.handle('backend:boot-phase', () => currentBootPhase)
  ipcMain.handle('backend:open-logs', () => {
    const logPath = path.join(app.getPath('logs'), 'backend.log')
    shell.showItemInFolder(logPath)
  })

  // Config
  ipcMain.handle('config:get', () => loadConfig())
  ipcMain.handle('config:set', (_e, patch: Partial<AppConfig>) => saveConfig(patch))
  ipcMain.handle('config:set-backend-mode', async (_e, mode: BackendMode) => {
    const next = saveConfig({ backendMode: mode })
    // If Auto/Docker is mid-build (e.g. right after this Dockerfile
    // packaging fix, Auto started actually finding+building the image for
    // the first time — a multi-GB download the user didn't expect), cancel
    // it so this explicit mode switch actually takes effect immediately
    // instead of silently no-op'ing behind the still-running startBackend()
    // call's `booting` guard until the abandoned build finishes on its own.
    dockerManager?.cancelBuild()
    await stopBackend()
    startBackend().catch((err) => console.error('[backend] Restart after mode change failed:', err))
    return next
  })
  // Restarting (rather than just saving) matters here: it's what actually
  // triggers ensurePythonEnv() to notice requirements-gpu.txt vs
  // requirements.txt changed and reinstall torch — see startBackendNative().
  ipcMain.handle('config:set-native-gpu', async (_e, gpu: boolean) => {
    const next = saveConfig({ nativeGpu: gpu })
    await stopBackend()
    startBackend().catch((err) => console.error('[backend] Restart after native GPU toggle failed:', err))
    return next
  })

  // CPU/GPU picker. Shown as a full first-run screen the very first time
  // (see app.whenReady() above) *and* as a persistent control on every boot
  // screen after that (BackendBoot.tsx) — not a one-shot decision, since a
  // driver/GPU setup can change (or the user just wants to flip it) at any
  // later point too. Applies the choice to *both* dockerGpu and nativeGpu —
  // we don't know yet whether Auto mode will land on Docker or native
  // Python, and the user just answered "I want CPU" / "I want GPU" in
  // general, not "for this one specific backend". Whichever mode actually
  // starts will provision the matching (CPU-only or CUDA) PyTorch build.
  ipcMain.handle('setup:detect-gpus', () => detectGpus())
  ipcMain.handle('setup:choose', async (_e, device: 'cpu' | 'gpu') => {
    const gpu = device === 'gpu'
    const next = saveConfig({ nativeGpu: gpu, dockerGpu: gpu, setupComplete: true })
    // Same reasoning as config:set-backend-mode: this can now be clicked
    // while a backend is already running or mid-boot (Docker build,
    // pip install, ...), not just on the very first, backend-less launch —
    // cancel/stop whatever's in flight so the new choice actually takes
    // effect right away instead of queuing invisibly behind it.
    dockerManager?.cancelBuild()
    await stopBackend()
    startBackend().catch((err) => console.error('[backend] Could not start after setup choice:', err))
    return next
  })

  // Docker
  ipcMain.handle('docker:status', async () => dockerManager?.getStatus() ?? { running: false })
  ipcMain.handle('docker:available', async () => (dockerManager ? dockerManager.isAvailable() : false))
  ipcMain.handle('docker:image-exists', async () => (dockerManager ? dockerManager.imageExists(loadConfig().dockerGpu) : false))
  ipcMain.handle('docker:start', async () => {
    if (!dockerManager) return { success: false, error: 'Docker not initialized' }
    const port = await findAvailablePort(8000)
    return dockerManager.startContainer({ hostPort: port, gpu: loadConfig().dockerGpu })
  })
  ipcMain.handle('docker:stop', async () => dockerManager?.stopContainer())
  ipcMain.handle('docker:build', async () => {
    if (!dockerManager) return { success: false, error: 'Docker not initialized' }
    const gpu = loadConfig().dockerGpu
    const torchCudaChannel = gpu ? pickTorchCudaChannel(detectMaxCudaVersion()) : undefined
    return dockerManager.buildImage((line) => mainWindow?.webContents.send('docker:build-progress', line), gpu, torchCudaChannel)
  })
  ipcMain.handle('docker:logs', async (_event, tail?: number) => dockerManager?.getContainerLogs(tail))

  ipcMain.handle('app:version', () => app.getVersion())
  ipcMain.handle('app:platform', () => process.platform)
  ipcMain.handle('app:paths', () => ({
    userData: app.getPath('userData'),
    logs: app.getPath('logs'),
    temp: app.getPath('temp'),
  }))

  ipcMain.handle('shell:openExternal', (_event, url: string) => shell.openExternal(url))
  ipcMain.handle('shell:showInFolder', (_event, p: string) => shell.showItemInFolder(p))

  ipcMain.handle('dialog:pickDirectory', async () => {
    if (!mainWindow) return { canceled: true, filePaths: [] }
    return dialog.showOpenDialog(mainWindow, { properties: ['openDirectory'] })
  })
}

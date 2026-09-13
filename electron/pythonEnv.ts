// Bootstraps a private virtual environment for the Native backend mode so
// the app can genuinely "install everything it needs" on first run, instead
// of assuming the user's system Python already has fastapi/gymnasium/torch.
//
// The interpreter itself is resolved in this order:
//   1. A real CPython 3.10+ already on the machine (PATH, Windows `py -3`
//      launcher, common install dirs). The Microsoft Store `python.exe` stub
//      is skipped — it opens the Store and is not a working interpreter.
//   2. If nothing usable is found, a portable CPython from
//      python-build-standalone is downloaded into `userData/python-runtime`
//      and reused on later boots. That is what makes a Windows .exe install
//      work without asking the user to install Python first.
import { spawn, spawnSync, type SpawnSyncOptions } from 'child_process'
import crypto from 'crypto'
import fs from 'fs'
import https from 'https'
import path from 'path'

export type PythonEnvPhase = 'installing-python' | 'creating-venv' | 'installing-dependencies'

/** Pinned portable CPython used when the machine has no usable interpreter. */
export const MANAGED_CPYTHON_VERSION = '3.12.13'
export const MANAGED_PYTHON_RELEASE = '20260325'

const PROBE_CODE = 'import sys; assert sys.version_info >= (3, 10); print(sys.executable)'
const DOWNLOAD_UA = 'reinforcement-studio'

export function isWindowsStorePython(executable: string): boolean {
  return /\\WindowsApps\\/i.test(executable)
}

export function managedPythonTriple(
  platform: NodeJS.Platform = process.platform,
  arch: string = process.arch,
): string | null {
  if (platform === 'win32' && arch === 'x64') return 'x86_64-pc-windows-msvc'
  if (platform === 'win32' && arch === 'arm64') return 'aarch64-pc-windows-msvc'
  if (platform === 'linux' && arch === 'x64') return 'x86_64-unknown-linux-gnu'
  if (platform === 'linux' && arch === 'arm64') return 'aarch64-unknown-linux-gnu'
  if (platform === 'darwin' && arch === 'x64') return 'x86_64-apple-darwin'
  if (platform === 'darwin' && arch === 'arm64') return 'aarch64-apple-darwin'
  return null
}

export function managedPythonDownloadUrl(
  platform: NodeJS.Platform = process.platform,
  arch: string = process.arch,
): string {
  const triple = managedPythonTriple(platform, arch)
  if (!triple) {
    throw new Error(`Нет встроенного Python для ${platform}/${arch}`)
  }
  const name = `cpython-${MANAGED_CPYTHON_VERSION}+${MANAGED_PYTHON_RELEASE}-${triple}-install_only_stripped.tar.gz`
  return `https://github.com/astral-sh/python-build-standalone/releases/download/${MANAGED_PYTHON_RELEASE}/${name}`
}

export function managedPythonInterpreterPath(
  runtimeDir: string,
  platform: NodeJS.Platform = process.platform,
): string {
  return platform === 'win32'
    ? path.join(runtimeDir, 'python', 'python.exe')
    : path.join(runtimeDir, 'python', 'bin', 'python3')
}

function spawnDefaults(): SpawnSyncOptions {
  return {
    encoding: 'utf8',
    timeout: 8_000,
    windowsHide: true,
    env: process.env,
  }
}

function probePython(command: string, extraArgs: string[] = []): string | null {
  if (path.isAbsolute(command) && !fs.existsSync(command)) return null
  try {
    const result = spawnSync(command, [...extraArgs, '-c', PROBE_CODE], spawnDefaults())
    if (result.status !== 0) return null
    const executable = String(result.stdout ?? '').trim().split(/\r?\n/).filter(Boolean).pop()
    if (!executable) return null
    if (isWindowsStorePython(executable)) return null
    return executable
  } catch {
    return null
  }
}

/** Windows packaged Electron often has a stripped PATH — call tar/where by
 *  absolute System32 path so first-run Python extract still works. */
export function tarExecutable(
  platform: NodeJS.Platform = process.platform,
  systemRoot: string = process.env.SystemRoot || 'C:\\Windows',
  exists: (filePath: string) => boolean = fs.existsSync,
): string {
  if (platform === 'win32') {
    const exe = path.join(systemRoot, 'System32', 'tar.exe')
    if (exists(exe)) return exe
  }
  return 'tar'
}

function windowsWhereBin(): string {
  const exe = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'where.exe')
  return fs.existsSync(exe) ? exe : 'where'
}

function unique(items: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const item of items) {
    const key = item.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(item)
  }
  return out
}

function windowsWhere(name: string): string[] {
  try {
    const result = spawnSync(windowsWhereBin(), [name], spawnDefaults())
    if (result.status !== 0 || !result.stdout) return []
    return String(result.stdout).split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
  } catch {
    return []
  }
}

function windowsCandidateExecutables(): string[] {
  const systemRoot = process.env.SystemRoot || 'C:\\Windows'
  const localAppData = process.env.LOCALAPPDATA
  const programFiles = process.env.ProgramFiles
  const userProfile = process.env.USERPROFILE
  const versions = ['314', '313', '312', '311', '310']
  const candidates: string[] = [
    path.join(systemRoot, 'py.exe'),
    path.join(systemRoot, 'System32', 'py.exe'),
    ...windowsWhere('py'),
    ...windowsWhere('python'),
    ...windowsWhere('python3'),
  ]
  for (const root of [localAppData && path.join(localAppData, 'Programs', 'Python'), programFiles, 'C:\\']) {
    if (!root) continue
    for (const version of versions) {
      candidates.push(path.join(root, `Python${version}`, 'python.exe'))
    }
  }
  if (userProfile) {
    for (const distro of ['miniconda3', 'anaconda3', 'miniforge3']) {
      candidates.push(path.join(userProfile, distro, 'python.exe'))
    }
  }
  return unique(candidates.filter((item) => !isWindowsStorePython(item)))
}

/** Returns an absolute path to a working CPython 3.10+, or null. */
export function resolveSystemPython(): string | null {
  if (process.platform === 'win32') {
    for (const candidate of windowsCandidateExecutables()) {
      const extra = /(?:^|\\|\/)py(?:\.exe)?$/i.test(candidate) ? ['-3'] : []
      const found = probePython(candidate, extra)
      if (found) return found
    }
    return null
  }

  for (const command of ['python3', 'python']) {
    const found = probePython(command)
    if (found) return found
  }
  for (const candidate of ['/usr/bin/python3', '/usr/local/bin/python3']) {
    const found = probePython(candidate)
    if (found) return found
  }
  return null
}

export function venvPythonPath(venvDir: string): string {
  return process.platform === 'win32'
    ? path.join(venvDir, 'Scripts', 'python.exe')
    : path.join(venvDir, 'bin', 'python3')
}

function run(cmd: string, args: string[]): Promise<number> {
  return new Promise((resolve) => {
    const proc = spawn(cmd, args, { stdio: 'ignore', windowsHide: true })
    proc.on('error', () => resolve(1))
    proc.on('exit', (code) => resolve(code ?? 1))
  })
}

/** Checks whether `pip` is actually importable inside the given venv python. */
async function hasPip(pyExe: string): Promise<boolean> {
  if (!fs.existsSync(pyExe)) return false
  const code = await run(pyExe, ['-m', 'pip', '--version'])
  return code === 0
}

function formatMb(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} МБ`
}

/** Downloads a URL to a local file, following redirects. */
function downloadFile(
  url: string,
  destPath: string,
  onProgress?: (received: number, total: number | null) => void,
  redirectsLeft = 8,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const req = https.get(url, {
      timeout: 30_000,
      headers: { 'User-Agent': DOWNLOAD_UA, Accept: '*/*' },
    }, (res) => {
      const status = res.statusCode ?? 0
      if (status >= 300 && status < 400 && res.headers.location && redirectsLeft > 0) {
        res.resume()
        downloadFile(res.headers.location, destPath, onProgress, redirectsLeft - 1).then(resolve, reject)
        return
      }
      if (status !== 200) {
        res.resume()
        reject(new Error(`HTTP ${status} while downloading ${url}`))
        return
      }
      const total = Number(res.headers['content-length'] || 0) || null
      let received = 0
      const file = fs.createWriteStream(destPath)
      res.on('data', (chunk: Buffer) => {
        received += chunk.length
        onProgress?.(received, total)
      })
      res.pipe(file)
      file.on('finish', () => file.close(() => resolve()))
      file.on('error', reject)
    })
    req.on('error', reject)
    req.on('timeout', () => req.destroy(new Error(`Timed out downloading ${url}`)))
  })
}

/**
 * Last-resort pip bootstrap for Python builds that ship without working
 * ensurepip wheels (seen on some Anaconda/Miniconda distributions) — fetches
 * the official installer from bootstrap.pypa.io and runs it directly, so the
 * app can self-heal without asking the user to run any manual commands.
 */
async function bootstrapPipViaGetPip(
  pyExe: string,
  venvDir: string,
  onLine: (line: string) => void,
): Promise<boolean> {
  const getPipPath = path.join(venvDir, 'get-pip.py')
  try {
    onLine('ensurepip недоступен — скачиваю get-pip.py...')
    await downloadFile('https://bootstrap.pypa.io/get-pip.py', getPipPath)
    // get-pip.py itself calls out to PyPI to fetch the pip wheel, so this can
    // hang on a restricted/flaky network — cap it instead of freezing the UI.
    const code = await runStreaming(pyExe, [getPipPath, '--no-input'], onLine, 90_000)
    return code === 0 && (await hasPip(pyExe))
  } catch (err) {
    onLine(`Не удалось скачать/запустить get-pip.py: ${err instanceof Error ? err.message : String(err)}`)
    return false
  } finally {
    fs.rmSync(getPipPath, { force: true })
  }
}

function requirementsHash(files: string[]): string {
  const hash = crypto.createHash('sha256')
  for (const f of files) {
    try {
      hash.update(fs.readFileSync(f))
    } catch {
      // missing requirement file just contributes nothing to the hash
    }
  }
  return hash.digest('hex')
}

/**
 * Runs a command, streaming its combined stdout/stderr line by line.
 * `timeoutMs` guards against steps that can hang indefinitely on a stalled
 * network connection (e.g. pip/get-pip.py waiting on a dead socket) — without
 * this, a single flaky download could freeze the boot screen forever.
 */
function runStreaming(
  cmd: string,
  args: string[],
  onLine?: (line: string) => void,
  timeoutMs = 120_000,
): Promise<number> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true })
    const forward = (buf: Buffer) => {
      const text = buf.toString()
      for (const line of text.split(/\r?\n/)) {
        if (line.trim()) onLine?.(line.trim())
      }
    }
    proc.stdout?.on('data', forward)
    proc.stderr?.on('data', forward)
    proc.on('error', reject)

    const timer = setTimeout(() => {
      onLine?.(`Превышен таймаут (${Math.round(timeoutMs / 1000)}с) — прерываю`)
      proc.kill('SIGKILL')
    }, timeoutMs)

    proc.on('exit', (code) => {
      clearTimeout(timer)
      resolve(code ?? 1)
    })
  })
}

function runtimeId(platform: NodeJS.Platform = process.platform, arch: string = process.arch): string {
  return `cpython-${MANAGED_CPYTHON_VERSION}+${MANAGED_PYTHON_RELEASE}-${managedPythonTriple(platform, arch) ?? 'unknown'}`
}

async function ensureManagedPython(
  runtimeDir: string,
  onProgress: (phase: PythonEnvPhase, line?: string) => void,
): Promise<string> {
  const pyExe = managedPythonInterpreterPath(runtimeDir)
  const markerPath = path.join(runtimeDir, '.runtime-id')
  const expectedId = runtimeId()
  const installedId = fs.existsSync(markerPath) ? fs.readFileSync(markerPath, 'utf-8').trim() : ''
  if (installedId === expectedId && probePython(pyExe)) return pyExe

  const triple = managedPythonTriple()
  if (!triple) {
    throw new Error(
      `Python 3.10+ не найден, а для ${process.platform}/${process.arch} нет встроенного интерпретатора. ` +
      'Установите Python с python.org и перезапустите приложение.',
    )
  }

  onProgress('installing-python', `Скачиваю встроенный Python ${MANAGED_CPYTHON_VERSION}…`)
  const stagingDir = `${runtimeDir}.staging`
  fs.rmSync(stagingDir, { recursive: true, force: true })
  fs.mkdirSync(stagingDir, { recursive: true })
  const archivePath = path.join(stagingDir, 'python.tar.gz')
  try {
    let lastReported = 0
    await downloadFile(managedPythonDownloadUrl(), archivePath, (received, total) => {
      if (received - lastReported < 1024 * 1024 && !(total && received === total)) return
      lastReported = received
      const suffix = total ? ` из ${formatMb(total)}` : ''
      onProgress('installing-python', `Скачиваю встроенный Python ${MANAGED_CPYTHON_VERSION}… ${formatMb(received)}${suffix}`)
    })
    onProgress('installing-python', 'Распаковываю интерпретатор…')
    const tarCode = await runStreaming(tarExecutable(), ['-xf', archivePath, '-C', stagingDir], (line) => onProgress('installing-python', line), 60_000)
    fs.rmSync(archivePath, { force: true })
    if (tarCode !== 0) {
      throw new Error('Не удалось распаковать встроенный Python (tar вернул ошибку).')
    }
    const stagedPython = managedPythonInterpreterPath(stagingDir)
    if (!probePython(stagedPython)) {
      throw new Error('Встроенный Python скачался, но не запускается.')
    }
    fs.rmSync(runtimeDir, { recursive: true, force: true })
    fs.renameSync(stagingDir, runtimeDir)
    fs.writeFileSync(path.join(runtimeDir, '.runtime-id'), expectedId)
    return managedPythonInterpreterPath(runtimeDir)
  } catch (err) {
    fs.rmSync(stagingDir, { recursive: true, force: true })
    const detail = err instanceof Error ? err.message : String(err)
    throw new Error(
      `Не удалось установить встроенный Python (${detail}). Проверьте интернет и перезапустите приложение.`,
    )
  }
}

async function resolveBasePython(
  runtimeDir: string,
  onProgress: (phase: PythonEnvPhase, line?: string) => void,
): Promise<string> {
  const existing = resolveSystemPython()
  if (existing) return existing
  return ensureManagedPython(runtimeDir, onProgress)
}

export interface EnsurePythonEnvResult {
  pythonPath: string
}

/**
 * Ensures a dedicated venv exists under `venvDir` with all packages from
 * `requirementFiles` installed. Skips reinstall if neither the requirement
 * files' content nor `extraPipArgs` (e.g. which CUDA wheel channel to pull
 * torch from) have changed since the last successful install (tracked via
 * a content hash).
 *
 * A working system Python is no longer required: if none is found, a
 * portable CPython is downloaded next to the venv (`../python-runtime`).
 */
export async function ensurePythonEnv(
  venvDir: string,
  requirementFiles: string[],
  onProgress: (phase: PythonEnvPhase, line?: string) => void,
  extraPipArgs: string[] = [],
): Promise<EnsurePythonEnvResult> {
  const pyExe = venvPythonPath(venvDir)
  const existingFiles = requirementFiles.filter((f) => fs.existsSync(f))
  // extraPipArgs folded into the hash too: switching CUDA wheel channel
  // (e.g. detected driver now supports cu130 instead of last time's cu126)
  // must trigger a reinstall even though the requirement *files* themselves
  // didn't change a single byte.
  const currentHash = requirementsHash(existingFiles) + ':' + extraPipArgs.join(' ')
  const markerPath = path.join(venvDir, '.deps-hash')
  const runtimeDir = path.join(path.dirname(venvDir), 'python-runtime')

  // A previous run may have been interrupted (crash, force-quit, killed
  // process) partway through venv creation, leaving a `python3` binary in
  // place but no working `pip` — or the requirements hash marker stale.
  // Re-verify pip actually works every time rather than trusting that the
  // venv directory existing means it's usable; self-heal instead of getting
  // stuck repeating the same failure forever.
  if (fs.existsSync(pyExe) && !(await hasPip(pyExe))) {
    onProgress('creating-venv', 'Обнаружено повреждённое окружение — пересобираю venv...')
    fs.rmSync(venvDir, { recursive: true, force: true })
  }

  if (!fs.existsSync(pyExe)) {
    const basePython = await resolveBasePython(runtimeDir, onProgress)
    onProgress('creating-venv')
    fs.mkdirSync(path.dirname(venvDir), { recursive: true })
    // --system-site-packages lets the venv fall back to the base Python's
    // own site-packages when something isn't installed locally. This is
    // what actually fixes the common Anaconda/Miniconda case: those base
    // environments almost always have a perfectly working `pip` already
    // (conda installs it), even when the *venv module's* bundled ensurepip
    // wheels are stripped/broken. It costs nothing when the base env is
    // "clean" system Python, since pip still installs our packages into the
    // venv's own (isolated) site-packages, not the base one.
    await runStreaming(
      basePython,
      ['-m', 'venv', '--system-site-packages', venvDir],
      (line) => onProgress('creating-venv', line),
      60_000,
    )

    if (!fs.existsSync(pyExe)) {
      throw new Error(
        'Не удалось создать виртуальное окружение Python (python3 -m venv не создал интерпретатор). ' +
        'Убедитесь, что установлен модуль venv для вашего Python.',
      )
    }

    // Note: we deliberately don't gate this on the venv command's own exit
    // code — cpython's venv module usually still leaves a working python3
    // binary in place even when its final ensurepip step fails, which is
    // exactly the case we're repairing here.
    if (!(await hasPip(pyExe))) {
      onProgress('creating-venv', 'pip не установился автоматически, пробую ensurepip...')
      await runStreaming(pyExe, ['-m', 'ensurepip', '--upgrade'], (line) => onProgress('creating-venv', line), 30_000)
    }
    if (!(await hasPip(pyExe))) {
      // Last resort, only reached if the base Python has no usable pip
      // anywhere AND ensurepip's bundled wheels are broken — fetch pip
      // directly instead of asking the user to fix their system Python.
      // Bounded by its own internal timeout so a dead network can't hang
      // the boot screen forever.
      await bootstrapPipViaGetPip(pyExe, venvDir, (line) => onProgress('creating-venv', line))
    }
    if (!(await hasPip(pyExe))) {
      throw new Error(
        'Не удалось создать рабочее виртуальное окружение Python: pip недоступен даже через system-site-packages, ' +
        'ensurepip и загрузку get-pip.py. Проверьте, что у базового Python есть рабочий pip, и что есть доступ в интернет.',
      )
    }
  }

  const installedHash = fs.existsSync(markerPath) ? fs.readFileSync(markerPath, 'utf-8').trim() : null
  if (installedHash !== currentHash && existingFiles.length > 0) {
    onProgress('installing-dependencies')
    const pipArgs = ['-m', 'pip', 'install', '--disable-pip-version-check', '--no-input']
    // --force-reinstall matters specifically for the CPU<->GPU torch swap
    // (toggling the "GPU" setting points this at requirements-gpu.txt
    // instead of requirements.txt, or extraPipArgs' --index-url now points
    // at a different CUDA channel than last time): all of these still
    // satisfy the same `torch>=2.2` constraint, so a plain `pip install`
    // would see the already-installed wheel as "good enough" and silently
    // keep it, leaving the GPU toggle (or a newly-detected driver) with no
    // actual effect. This only runs when the hash changed (first install,
    // app update, or one of these toggles), so the extra reinstall cost is
    // rare, not per-boot.
    pipArgs.push('--force-reinstall')
    pipArgs.push(...extraPipArgs)
    for (const f of existingFiles) pipArgs.push('-r', f)
    // Generous timeout: torch + friends are a real download (hundreds of MB)
    // and can legitimately take several minutes on a slow connection. This
    // only guards against a truly dead/stalled connection, not a slow one —
    // pip's own progress output keeps streaming to the UI the whole time.
    const code = await runStreaming(
      pyExe,
      pipArgs,
      (line) => onProgress('installing-dependencies', line),
      30 * 60_000,
    )
    if (code !== 0) {
      throw new Error(
        'Не удалось установить Python-зависимости (pip install). Подробности — в логах backend.',
      )
    }
    fs.writeFileSync(markerPath, currentHash)
  }

  return { pythonPath: pyExe }
}

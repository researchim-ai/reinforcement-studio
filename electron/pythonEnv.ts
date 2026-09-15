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
export const MANAGED_UV_VERSION = '0.12.9'

export interface ManagedUvAsset {
  triple: string
  archive: string
  sha256: string
}

const UV_ASSETS: Partial<Record<NodeJS.Platform, Partial<Record<string, ManagedUvAsset>>>> = {
  win32: {
    x64: {
      triple: 'x86_64-pc-windows-msvc',
      archive: 'uv-x86_64-pc-windows-msvc.zip',
      sha256: 'ddbfcee1ac615a0499f6aa97b5ec8ebdf3ee4a7714a48055ec2ba0030e3cf810',
    },
    arm64: {
      triple: 'aarch64-pc-windows-msvc',
      archive: 'uv-aarch64-pc-windows-msvc.zip',
      sha256: 'd3360363a3cb671f2c854f4ef48cf4a57fe8664f8ec6a248076d68b797a8acc0',
    },
  },
  linux: {
    x64: {
      triple: 'x86_64-unknown-linux-gnu',
      archive: 'uv-x86_64-unknown-linux-gnu.tar.gz',
      sha256: 'ec7a99cd05e0cd7f80243f135ce1361c76835cb0ee60055d14d20eba8eba1460',
    },
    arm64: {
      triple: 'aarch64-unknown-linux-gnu',
      archive: 'uv-aarch64-unknown-linux-gnu.tar.gz',
      sha256: 'c36fe17937ff6bd16dc42fc13854b5465999fcab2efe0af559381e945e3c6001',
    },
  },
  darwin: {
    x64: {
      triple: 'x86_64-apple-darwin',
      archive: 'uv-x86_64-apple-darwin.tar.gz',
      sha256: 'e1ca175824f1056589ce9908f7631879ebc3c36535b5e63dc06510beb370b4c1',
    },
    arm64: {
      triple: 'aarch64-apple-darwin',
      archive: 'uv-aarch64-apple-darwin.tar.gz',
      sha256: '301f72afaf54060f92da7016cb0115bd077f43a9c8e39c1d8170a0bac80fd398',
    },
  },
}

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

export function managedUvAsset(
  platform: NodeJS.Platform = process.platform,
  arch: string = process.arch,
): ManagedUvAsset {
  const asset = UV_ASSETS[platform]?.[arch]
  if (!asset) throw new Error(`Нет standalone uv для ${platform}/${arch}`)
  return asset
}

export function managedUvDownloadUrl(
  platform: NodeJS.Platform = process.platform,
  arch: string = process.arch,
): string {
  const asset = managedUvAsset(platform, arch)
  return `https://releases.astral.sh/github/uv/releases/download/${MANAGED_UV_VERSION}/${asset.archive}`
}

export function managedUvExecutablePath(
  runtimeDir: string,
  platform: NodeJS.Platform = process.platform,
  arch: string = process.arch,
): string {
  // Official Windows ZIPs contain uv.exe directly at archive root, while
  // Unix tarballs wrap the binaries in an `uv-<target-triple>` directory.
  if (platform === 'win32') return path.join(runtimeDir, 'uv.exe')
  const { triple } = managedUvAsset(platform, arch)
  return path.join(runtimeDir, `uv-${triple}`, 'uv')
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

export function uvPipInstallArgs(
  pyExe: string,
  requirementFiles: string[],
  extraInstallArgs: string[] = [],
  torchInstallArgs: string[] = [],
): string[] {
  const args = ['pip', 'install', '--python', pyExe]
  if (torchInstallArgs.length > 0) {
    // Resolve torch and the rest of the environment in one transaction.
    // A second generic-index pass could otherwise replace cu128 afterwards.
    args.push('--reinstall-package', 'torch', ...torchInstallArgs)
  } else {
    args.push('--reinstall')
  }
  args.push(...extraInstallArgs)
  for (const file of requirementFiles) args.push('-r', file)
  return args
}

function fileSha256(filePath: string): string {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex')
}

/**
 * Runs a command, streaming its combined stdout/stderr line by line.
 * `timeoutMs` guards against steps that can hang indefinitely on a stalled
 * network connection (e.g. uv waiting on a dead package index) — without
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

function uvProbeFailure(uvExe: string): string | null {
  if (!fs.existsSync(uvExe)) return `файл не найден: ${uvExe}`
  try {
    const result = spawnSync(uvExe, ['--version'], spawnDefaults())
    if (result.error) return result.error.message
    const stdout = String(result.stdout ?? '').trim()
    const stderr = String(result.stderr ?? '').trim()
    if (result.status !== 0) {
      return stderr || stdout || `процесс завершился с кодом ${result.status}`
    }
    if (!stdout.includes(`uv ${MANAGED_UV_VERSION}`)) {
      return `неожиданная версия: ${stdout || '(пустой вывод)'}`
    }
    return null
  } catch (err) {
    return err instanceof Error ? err.message : String(err)
  }
}

function probeUv(uvExe: string): boolean {
  return uvProbeFailure(uvExe) === null
}

async function ensureManagedUv(
  runtimeDir: string,
  onProgress: (phase: PythonEnvPhase, line?: string) => void,
): Promise<string> {
  const asset = managedUvAsset()
  const uvExe = managedUvExecutablePath(runtimeDir)
  const markerPath = path.join(runtimeDir, '.uv-id')
  const expectedId = `${MANAGED_UV_VERSION}:${asset.triple}:${asset.sha256}`
  const installedId = fs.existsSync(markerPath) ? fs.readFileSync(markerPath, 'utf-8').trim() : ''
  if (installedId === expectedId && probeUv(uvExe)) return uvExe

  const stagingDir = `${runtimeDir}.staging`
  fs.rmSync(stagingDir, { recursive: true, force: true })
  fs.mkdirSync(stagingDir, { recursive: true })
  const archivePath = path.join(stagingDir, asset.archive)
  try {
    onProgress('installing-dependencies', `Скачиваю uv ${MANAGED_UV_VERSION}…`)
    let lastReported = 0
    await downloadFile(managedUvDownloadUrl(), archivePath, (received, total) => {
      if (received - lastReported < 1024 * 1024 && !(total && received === total)) return
      lastReported = received
      const suffix = total ? ` из ${formatMb(total)}` : ''
      onProgress(
        'installing-dependencies',
        `Скачиваю uv ${MANAGED_UV_VERSION}… ${formatMb(received)}${suffix}`,
      )
    })
    const actualSha256 = fileSha256(archivePath)
    if (actualSha256 !== asset.sha256) {
      throw new Error(`checksum не совпал: ожидался ${asset.sha256}, получен ${actualSha256}`)
    }
    onProgress('installing-dependencies', 'Распаковываю uv…')
    const tarCode = await runStreaming(
      tarExecutable(),
      ['-xf', archivePath, '-C', stagingDir],
      (line) => onProgress('installing-dependencies', line),
      60_000,
    )
    fs.rmSync(archivePath, { force: true })
    if (tarCode !== 0) throw new Error('tar вернул ошибку')
    const stagedUv = managedUvExecutablePath(stagingDir)
    if (process.platform !== 'win32') fs.chmodSync(stagedUv, 0o755)
    const probeFailure = uvProbeFailure(stagedUv)
    if (probeFailure) throw new Error(`скачанный uv не запускается: ${probeFailure}`)
    fs.rmSync(runtimeDir, { recursive: true, force: true })
    fs.renameSync(stagingDir, runtimeDir)
    fs.writeFileSync(path.join(runtimeDir, '.uv-id'), expectedId)
    return managedUvExecutablePath(runtimeDir)
  } catch (err) {
    fs.rmSync(stagingDir, { recursive: true, force: true })
    const detail = err instanceof Error ? err.message : String(err)
    throw new Error(`Не удалось установить uv (${detail}). Проверьте интернет и перезапустите приложение.`)
  }
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
 * files' content, `extraInstallArgs`, nor `torchInstallArgs` (e.g. the
 * CUDA backend selected for torch) have changed since the last
 * successful install (tracked via a content hash).
 *
 * A working system Python is no longer required: if none is found, a
 * portable CPython is downloaded next to the venv (`../python-runtime`).
 */
export async function ensurePythonEnv(
  venvDir: string,
  requirementFiles: string[],
  onProgress: (phase: PythonEnvPhase, line?: string) => void,
  extraInstallArgs: string[] = [],
  torchInstallArgs: string[] = [],
): Promise<EnsurePythonEnvResult> {
  const pyExe = venvPythonPath(venvDir)
  const existingFiles = requirementFiles.filter((f) => fs.existsSync(f))
  // Installer arguments are folded into the hash too: switching CUDA wheel
  // channel must trigger a reinstall even when requirement files did not
  // change.
  const currentHash = [
    requirementsHash(existingFiles),
    extraInstallArgs.join(' '),
    torchInstallArgs.join(' '),
    `uv-install-v2-${MANAGED_UV_VERSION}`,
  ].join(':')
  const markerPath = path.join(venvDir, '.deps-hash')
  const runtimeDir = path.join(path.dirname(venvDir), 'python-runtime')
  const uvRuntimeDir = path.join(path.dirname(venvDir), 'uv-runtime')
  const uvExe = await ensureManagedUv(uvRuntimeDir, onProgress)

  // uv-managed environments do not need pip inside the venv. Probe the
  // interpreter itself so an interrupted creation self-heals without
  // deleting a valid pip-less uv environment on every boot.
  if (fs.existsSync(pyExe) && !probePython(pyExe)) {
    onProgress('creating-venv', 'Обнаружено повреждённое окружение — пересобираю venv...')
    fs.rmSync(venvDir, { recursive: true, force: true })
  }

  if (!fs.existsSync(pyExe)) {
    const basePython = await resolveBasePython(runtimeDir, onProgress)
    onProgress('creating-venv', 'Создаю окружение через uv…')
    fs.mkdirSync(path.dirname(venvDir), { recursive: true })
    const createCode = await runStreaming(
      uvExe,
      ['venv', '--python', basePython, '--clear', venvDir],
      (line) => onProgress('creating-venv', line),
      60_000,
    )

    if (createCode !== 0 || !probePython(pyExe)) {
      throw new Error(
        'Не удалось создать виртуальное окружение Python через uv.',
      )
    }
  }

  const installedHash = fs.existsSync(markerPath) ? fs.readFileSync(markerPath, 'utf-8').trim() : null
  if (installedHash !== currentHash && existingFiles.length > 0) {
    onProgress('installing-dependencies')
    const uvArgs = uvPipInstallArgs(pyExe, existingFiles, extraInstallArgs, torchInstallArgs)
    onProgress(
      'installing-dependencies',
      `Команда: uv ${uvArgs.map((arg) => JSON.stringify(arg)).join(' ')}`,
    )
    // Generous timeout: torch + friends are a real download (hundreds of MB)
    // and can legitimately take several minutes on a slow connection. This
    // only guards against a truly dead/stalled connection, not a slow one —
    // uv's own progress output keeps streaming to the UI the whole time.
    const code = await runStreaming(
      uvExe,
      uvArgs,
      (line) => onProgress('installing-dependencies', line),
      30 * 60_000,
    )
    if (code !== 0) {
      throw new Error(
        'Не удалось установить Python-зависимости через uv. Подробности — в логах backend.',
      )
    }
    const backendFlag = torchInstallArgs.indexOf('--torch-backend')
    const expectedTorchBackend = backendFlag >= 0 ? torchInstallArgs[backendFlag + 1] : null
    if (expectedTorchBackend) {
      const verifyCode = await runStreaming(
        pyExe,
        [
          '-c',
          [
            'import sys, torch',
            'expected = sys.argv[1]',
            'actual = "cpu" if torch.version.cuda is None else "cu" + torch.version.cuda.replace(".", "")',
            'print(f"PyTorch {torch.__version__}; backend={actual}; expected={expected}")',
            'assert actual == expected, f"PyTorch backend mismatch: installed {actual}, expected {expected}"',
          ].join('; '),
          expectedTorchBackend,
        ],
        (line) => onProgress('installing-dependencies', line),
        60_000,
      )
      if (verifyCode !== 0) {
        throw new Error(
          `Установлен неверный backend PyTorch вместо ${expectedTorchBackend}. ` +
          'Окружение не будет помечено готовым; подробности — в логах backend.',
        )
      }
    }
    fs.writeFileSync(markerPath, currentHash)
  }

  return { pythonPath: pyExe }
}

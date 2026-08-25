// Bootstraps a private virtual environment for the Native backend mode so
// the app can genuinely "install everything it needs" on first run, instead
// of assuming the user's system Python already has fastapi/gymnasium/torch.
import { spawn, spawnSync } from 'child_process'
import crypto from 'crypto'
import fs from 'fs'
import https from 'https'
import path from 'path'

export type PythonEnvPhase = 'creating-venv' | 'installing-dependencies'

function candidatePythonCommands(): string[] {
  return process.platform === 'win32' ? ['python', 'py', 'python3'] : ['python3', 'python']
}

export function resolveSystemPython(): string | null {
  for (const cmd of candidatePythonCommands()) {
    try {
      const result = spawnSync(cmd, ['--version'], { stdio: 'pipe' })
      if (result.status === 0) return cmd
    } catch {
      // try next candidate
    }
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
    const proc = spawn(cmd, args, { stdio: 'ignore' })
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

/** Downloads a URL to a local file, following redirects. */
function downloadFile(url: string, destPath: string, redirectsLeft = 5): Promise<void> {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { timeout: 30_000 }, (res) => {
      const status = res.statusCode ?? 0
      if (status >= 300 && status < 400 && res.headers.location && redirectsLeft > 0) {
        res.resume()
        downloadFile(res.headers.location, destPath, redirectsLeft - 1).then(resolve, reject)
        return
      }
      if (status !== 200) {
        res.resume()
        reject(new Error(`HTTP ${status} while downloading ${url}`))
        return
      }
      const file = fs.createWriteStream(destPath)
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
    const proc = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] })
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

export interface EnsurePythonEnvResult {
  pythonPath: string
}

/**
 * Ensures a dedicated venv exists under `venvDir` with all packages from
 * `requirementFiles` installed. Skips reinstall if neither the requirement
 * files' content nor `extraPipArgs` (e.g. which CUDA wheel channel to pull
 * torch from) have changed since the last successful install (tracked via
 * a content hash).
 */
export async function ensurePythonEnv(
  venvDir: string,
  requirementFiles: string[],
  onProgress: (phase: PythonEnvPhase, line?: string) => void,
  extraPipArgs: string[] = [],
): Promise<EnsurePythonEnvResult> {
  const systemPython = resolveSystemPython()
  if (!systemPython) {
    throw new Error(
      'Python 3 не найден в PATH. Установите Python 3.10+ (python.org) и перезапустите приложение.',
    )
  }

  const pyExe = venvPythonPath(venvDir)
  const existingFiles = requirementFiles.filter((f) => fs.existsSync(f))
  // extraPipArgs folded into the hash too: switching CUDA wheel channel
  // (e.g. detected driver now supports cu130 instead of last time's cu126)
  // must trigger a reinstall even though the requirement *files* themselves
  // didn't change a single byte.
  const currentHash = requirementsHash(existingFiles) + ':' + extraPipArgs.join(' ')
  const markerPath = path.join(venvDir, '.deps-hash')

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
      systemPython,
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

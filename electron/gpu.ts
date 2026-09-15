// Best-effort local GPU detection for the first-run setup screen — runs
// `nvidia-smi` directly from the Electron main process (no backend needed
// yet, since this happens *before* we decide whether/how to start one).
import { spawnSync } from 'child_process'

export interface DetectedGpu {
  index: number
  name: string
  memoryTotalMb: number
}

export function detectGpus(): DetectedGpu[] {
  try {
    const result = spawnSync(
      'nvidia-smi',
      ['--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'],
      { encoding: 'utf-8', timeout: 5000 },
    )
    if (result.status !== 0 || !result.stdout) return []

    return result.stdout
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const [idx, name, mem] = line.split(',').map((s) => s.trim())
        return { index: Number(idx) || 0, name: name || 'NVIDIA GPU', memoryTotalMb: Number(mem) || 0 }
      })
  } catch {
    // nvidia-smi not installed / not in PATH / no GPU — not an error, just
    // means we can't show GPU details on the setup screen (the user can
    // still pick "GPU" manually if they know better, e.g. via a driver that
    // doesn't ship nvidia-smi).
    return []
  }
}

/**
 * Max CUDA runtime version the installed NVIDIA driver can run, parsed from
 * plain `nvidia-smi`'s own header line ("CUDA Version: 12.8"). Returns null
 * if nvidia-smi isn't available at all (no driver, not installed, etc).
 */
export function detectMaxCudaVersion(): number | null {
  try {
    const result = spawnSync('nvidia-smi', [], { encoding: 'utf-8', timeout: 5000 })
    if (result.status !== 0 || !result.stdout) return null
    const match = result.stdout.match(/CUDA Version:\s*([\d.]+)/)
    return match ? parseFloat(match[1]) : null
  } catch {
    return null
  }
}

// Pick the newest wheel runtime the detected driver can run. CUDA 12.8 is
// not interchangeable with 12.6 here: RTX 50-series Blackwell GPUs
// (compute capability sm_120) require PyTorch's cu128-or-newer binaries;
// cu126 can install and report CUDA available, then fail at the first
// kernel launch with "no kernel image is available". Ordered newest-first;
// keep this list in sync with https://download.pytorch.org/whl/.
const TORCH_CUDA_CHANNELS: { channel: string; minDriverCuda: number }[] = [
  { channel: 'cu130', minDriverCuda: 13.0 },
  { channel: 'cu128', minDriverCuda: 12.8 },
  { channel: 'cu126', minDriverCuda: 12.6 },
]

/** Always returns a channel — cu126 (PyTorch's own documented "older
 * driver" fallback) if the driver's version is unknown or older than every
 * entry above, rather than guessing further down an increasingly-retired
 * list (cu121, cu118, ...). */
export function pickTorchCudaChannel(maxCudaVersion: number | null): string {
  if (maxCudaVersion != null) {
    for (const { channel, minDriverCuda } of TORCH_CUDA_CHANNELS) {
      if (maxCudaVersion >= minDriverCuda) return channel
    }
  }
  return TORCH_CUDA_CHANNELS[TORCH_CUDA_CHANNELS.length - 1].channel
}

export function torchCudaInstallArgs(channel: string): string[] {
  if (!/^cu\d+$/.test(channel)) throw new Error(`Invalid PyTorch CUDA channel: ${channel}`)
  return [
    '--torch-backend',
    channel,
    'torch>=2.7',
  ]
}

export function torchCpuInstallArgs(): string[] {
  return [
    '--torch-backend',
    'cpu',
    'torch>=2.2',
  ]
}

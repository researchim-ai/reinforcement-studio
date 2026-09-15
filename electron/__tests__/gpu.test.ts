import path from 'path'
import { describe, expect, it } from 'vitest'

import {
  gpuRequiresCuda128,
  nvidiaSmiExecutable,
  pickTorchCudaChannel,
  torchCpuInstallArgs,
  torchCudaInstallArgs,
} from '../gpu'

describe('PyTorch CUDA wheel selection', () => {
  it('selects cu128 for RTX 50-series capable drivers', () => {
    expect(pickTorchCudaChannel(12.8)).toBe('cu128')
    expect(pickTorchCudaChannel(12.9)).toBe('cu128')
  })

  it('selects the newest supported channel', () => {
    expect(pickTorchCudaChannel(13.0)).toBe('cu130')
    expect(pickTorchCudaChannel(13.1)).toBe('cu130')
  })

  it('keeps cu126 as the compatibility fallback', () => {
    expect(pickTorchCudaChannel(12.6)).toBe('cu126')
    expect(pickTorchCudaChannel(12.7)).toBe('cu126')
    expect(pickTorchCudaChannel(null)).toBe('cu126')
  })

  it('uses the absolute System32 nvidia-smi path in packaged Windows apps', () => {
    expect(nvidiaSmiExecutable(
      'win32',
      { SystemRoot: String.raw`C:\Windows` },
      (filePath) => filePath === path.win32.join(String.raw`C:\Windows`, 'System32', 'nvidia-smi.exe'),
    )).toBe(path.win32.join(String.raw`C:\Windows`, 'System32', 'nvidia-smi.exe'))
  })

  it('never selects cu126 for RTX 50-series Blackwell GPUs', () => {
    const gpus = [{ name: 'NVIDIA GeForce RTX 5070 Ti Laptop GPU' }]
    expect(gpuRequiresCuda128(gpus)).toBe(true)
    expect(pickTorchCudaChannel(null, gpus)).toBe('cu128')
    expect(pickTorchCudaChannel(12.6, gpus)).toBe('cu128')
    expect(pickTorchCudaChannel(12.9, gpus)).toBe('cu128')
    expect(pickTorchCudaChannel(13.0, gpus)).toBe('cu130')
  })

  it('installs torch from an isolated CUDA index', () => {
    const args = torchCudaInstallArgs('cu128')
    expect(args).toEqual([
      '--torch-backend',
      'cu128',
      'torch>=2.7',
    ])
    expect(args).not.toContain('--index-url')
  })

  it('installs CPU torch from its isolated index', () => {
    expect(torchCpuInstallArgs()).toEqual([
      '--torch-backend',
      'cpu',
      'torch>=2.2',
    ])
  })
})

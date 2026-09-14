import { describe, expect, it } from 'vitest'

import { pickTorchCudaChannel } from '../gpu'

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
})

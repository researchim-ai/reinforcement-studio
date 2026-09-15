import path from 'path'
import { describe, expect, it } from 'vitest'

import {
  isWindowsStorePython,
  managedPythonDownloadUrl,
  managedPythonInterpreterPath,
  managedPythonTriple,
  managedUvAsset,
  managedUvDownloadUrl,
  managedUvExecutablePath,
  tarExecutable,
  MANAGED_CPYTHON_VERSION,
  MANAGED_PYTHON_RELEASE,
  MANAGED_UV_VERSION,
} from '../pythonEnv'

describe('portable Python bootstrap', () => {
  it('rejects the Microsoft Store python.exe stub', () => {
    expect(isWindowsStorePython(String.raw`C:\Users\me\AppData\Local\Microsoft\WindowsApps\python.exe`)).toBe(true)
    expect(isWindowsStorePython(String.raw`C:\Users\me\AppData\Local\Programs\Python\Python312\python.exe`)).toBe(false)
  })

  it('builds a Windows x64 standalone download URL', () => {
    expect(managedPythonTriple('win32', 'x64')).toBe('x86_64-pc-windows-msvc')
    expect(managedPythonDownloadUrl('win32', 'x64')).toBe(
      `https://github.com/astral-sh/python-build-standalone/releases/download/${MANAGED_PYTHON_RELEASE}/cpython-${MANAGED_CPYTHON_VERSION}+${MANAGED_PYTHON_RELEASE}-x86_64-pc-windows-msvc-install_only_stripped.tar.gz`,
    )
  })

  it('points at python.exe inside the extracted runtime on Windows', () => {
    expect(managedPythonInterpreterPath('/data/python-runtime', 'win32')).toBe(
      path.join('/data/python-runtime', 'python', 'python.exe'),
    )
    expect(managedPythonInterpreterPath('/data/python-runtime', 'linux')).toBe(
      path.join('/data/python-runtime', 'python', 'bin', 'python3'),
    )
  })

  it('uses System32 tar.exe on Windows so a stripped Electron PATH still unpacks CPython', () => {
    expect(tarExecutable('linux')).toBe('tar')
    expect(tarExecutable('win32', 'C:\\Windows', () => true)).toBe(
      path.join('C:\\Windows', 'System32', 'tar.exe'),
    )
    expect(tarExecutable('win32', 'C:\\Windows', () => false)).toBe('tar')
  })

  it('pins the checksummed Windows x64 standalone uv archive', () => {
    expect(managedUvDownloadUrl('win32', 'x64')).toBe(
      `https://releases.astral.sh/github/uv/releases/download/${MANAGED_UV_VERSION}/uv-x86_64-pc-windows-msvc.zip`,
    )
    expect(managedUvAsset('win32', 'x64').sha256).toBe(
      'ddbfcee1ac615a0499f6aa97b5ec8ebdf3ee4a7714a48055ec2ba0030e3cf810',
    )
    expect(managedUvExecutablePath('/data/uv-runtime', 'win32', 'x64')).toBe(
      path.join('/data/uv-runtime', 'uv.exe'),
    )
  })

  it('selects native standalone uv archives for Linux and macOS', () => {
    expect(managedUvDownloadUrl('linux', 'arm64')).toContain('uv-aarch64-unknown-linux-gnu.tar.gz')
    expect(managedUvExecutablePath('/data/uv-runtime', 'darwin', 'arm64')).toBe(
      path.join('/data/uv-runtime', 'uv-aarch64-apple-darwin', 'uv'),
    )
  })
})

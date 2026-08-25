const fs = require('fs')
const path = require('path')

/**
 * AppImage mounts the application via FUSE. Chromium's setuid sandbox helper
 * (`chrome-sandbox`) cannot have the required root:4755 ownership/mode inside
 * that mount, and Electron aborts before the app's JS entrypoint can append
 * command-line switches. Wrap the packaged Linux executable so AppImage runs
 * with --no-sandbox from process start, while deb/dev keep the normal path.
 */
module.exports = async function afterPack(context) {
  if (context.electronPlatformName !== 'linux') return

  const appOutDir = context.appOutDir
  const exeName = context.packager?.executableName || 'reinforcement-studio'
  const exePath = path.join(appOutDir, exeName)
  const binPath = `${exePath}.bin`

  if (!fs.existsSync(exePath)) return

  const current = fs.readFileSync(exePath)
  if (current.slice(0, 2).toString() === '#!') return

  if (fs.existsSync(binPath)) fs.rmSync(binPath, { force: true })
  fs.renameSync(exePath, binPath)

  const script = `#!/bin/sh
set -e
APP_DIR="$(dirname "$(readlink -f "$0")")"
REAL_EXE="$APP_DIR/${exeName}.bin"

if [ -n "$APPIMAGE" ] || [ -n "$APPDIR" ]; then
  exec "$REAL_EXE" --no-sandbox "$@"
fi

exec "$REAL_EXE" "$@"
`

  fs.writeFileSync(exePath, script, { mode: 0o755 })
  fs.chmodSync(binPath, 0o755)
}

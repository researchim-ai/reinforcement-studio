"""FastAPI layer between the Electron UI and `rl_core`.

Keep `__version__` in lockstep with `package.json` — electron-builder
stamps the installer filename from the npm version, Settings reads it via
`app.getVersion()`, and `/api/system/info` returns this string so a
browser/dev session still sees the same number.
"""

__version__ = "0.1.0"

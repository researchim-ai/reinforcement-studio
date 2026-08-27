/**
 * Electron derives `app.getPath('userData')` from the app's name — on Linux
 * that's `~/.config/<name>`. This project's name changed twice while getting
 * the Linux install path space-free ("RL Studio" -> "Reinforcement Studio"
 * -> "reinforcement-studio", see electron-builder.yml/package.json history):
 * each rename points every fresh build at a brand new, empty userData
 * folder. That silently orphans everything in the *previous* one — most
 * importantly the private Python venv (`pyenv/`), which for a GPU install
 * means several gigabytes of already-downloaded PyTorch/CUDA wheels. From
 * the user's perspective, simply rebuilding the app makes it "randomly"
 * decide to reinstall all its dependencies from scratch, with no obvious
 * cause (see rl_core's `.deps-hash` check in pythonEnv.ts — it just sees an
 * empty target venv and behaves exactly as it would on a genuine first run).
 *
 * `migrateLegacyUserData()` runs once at startup, before anything else
 * reads config or touches the venv: if the *current* userData folder has no
 * fully-installed environment yet, but a sibling folder under an old name
 * does, it moves (not copies — these can be gigabytes) `pyenv/`, `rl_data/`
 * and `config.json` over instead of leaving the app to reinstall everything.
 *
 * This is safe specifically because of how this app invokes its venv: always
 * `<pyenv>/bin/python3 -m <module>` (pip, uvicorn, ...) rather than through
 * pip-installed console-script shims, which embed an absolute shebang back
 * to the venv's original path and would break on a move. The venv's own
 * `bin/python3` is a symlink to the *system* interpreter, not to itself, so
 * relocating the venv directory doesn't invalidate it.
 */
import { app } from 'electron'
import fs from 'fs'
import path from 'path'

// Every name this app (or its Electron userData folder) has been known by,
// newest first — see package.json/electron-builder.yml git history. Kept as
// a plain list rather than trying to derive it automatically since there's
// no reliable way to enumerate "past productNames" at runtime.
const LEGACY_USER_DATA_NAMES = ['reinforcement-studio', 'Reinforcement Studio', 'rl-studio', 'RL Studio']

const MIGRATED_ITEMS = ['pyenv', 'rl_data', 'config.json']

function hasInstalledEnv(userDataDir: string): boolean {
  return fs.existsSync(path.join(userDataDir, 'pyenv', '.deps-hash'))
}

export function migrateLegacyUserData(): void {
  const currentDir = app.getPath('userData')
  if (hasInstalledEnv(currentDir)) return // already set up here — nothing to migrate

  const currentName = path.basename(currentDir)
  const parentDir = path.dirname(currentDir)

  for (const legacyName of LEGACY_USER_DATA_NAMES) {
    if (legacyName === currentName) continue
    const legacyDir = path.join(parentDir, legacyName)
    if (!hasInstalledEnv(legacyDir)) continue

    console.log(
      `[migrate] Found a fully set up environment from an older build/name at "${legacyDir}" — ` +
        `moving it to "${currentDir}" instead of reinstalling everything from scratch.`,
    )
    try {
      fs.mkdirSync(currentDir, { recursive: true })
    } catch (err) {
      console.error('[migrate] Could not create the new userData directory:', err)
      return
    }
    for (const item of MIGRATED_ITEMS) {
      const src = path.join(legacyDir, item)
      const dst = path.join(currentDir, item)
      if (!fs.existsSync(src)) continue
      if (fs.existsSync(dst)) {
        // `pyenv` is special-cased: the very first launch under the new
        // name already creates an empty/half-bootstrapped venv (see
        // pythonEnv.ts) before this migration gets another chance to run —
        // that stub has to be cleared out of the way, or the real,
        // fully-installed one from `legacyDir` would just get skipped as
        // "already exists". Safe to do unconditionally here: reaching this
        // point already means the current dir has no *complete* env (the
        // early `hasInstalledEnv(currentDir)` return above would have
        // caught that). Never do this for `config.json`/`rl_data` though —
        // those might hold genuine user choices/data already written
        // under the new name.
        if (item !== 'pyenv') continue
        try {
          fs.rmSync(dst, { recursive: true, force: true })
        } catch (err) {
          console.error(`[migrate] Could not remove the incomplete "${dst}" before migrating:`, err)
          continue
        }
      }
      try {
        fs.renameSync(src, dst)
        console.log(`[migrate] Moved ${item}`)
      } catch (err) {
        console.error(`[migrate] Failed to move ${item} from "${src}" to "${dst}":`, err)
      }
    }
    return // only ever migrate from the single best (newest-named) match
  }
}

"""Standalone subprocess entry point for rendering one Environments-gallery
"tier 3" local-rollout preview GIF (see `rl_core/envs/previews.py`'s module
docstring for the three preview tiers).

Deliberately run out-of-process — `python -m rl_core.envs.preview_worker
<env_id> <output_path>` — rather than called as a plain in-process function.
Some envs' rendering backends don't fail with a catchable Python exception
when there's no display/GPU to render to: verified directly for MuJoCo's
default GLFW backend (used by the wrapped `PointMaze`/`AntMaze` envs in
`rl_core/envs/robotics_envs.py`) on a headless machine, where it segfaults
the entire interpreter instead of raising. A crash in this disposable
subprocess just means "no preview for this env, same as any other
rendering failure"; the same crash in-process would take the whole FastAPI
backend down with it.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m rl_core.envs.preview_worker <env_id> <output_path>", file=sys.stderr)
        return 2
    env_id, output_path = argv

    from rl_core.envs.previews import build_local_rollout_gif_bytes

    data = build_local_rollout_gif_bytes(env_id)
    if data is None:
        return 1
    Path(output_path).write_bytes(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

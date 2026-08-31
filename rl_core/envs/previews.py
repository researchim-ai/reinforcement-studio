"""Preview media (GIF/SVG thumbnails) for the Environments gallery.

Three tiers, tried in order for any given env id:
1. Official Farama Foundation preview GIFs (`PREVIEW_REL`) — the classic/
   toy/box2d/mujoco/atari cards, proxied + cached from the Gymnasium repo so
   the Electron file:// renderer does not depend on GitHub being reachable
   from <img> tags (and so ALE ids with slashes have a stable URL).
2. Hand-drawn bundled SVGs (`BUNDLED_SVG_PREVIEWS`) — board games and a few
   from-scratch POMDP tasks with no matching Farama asset.
3. A locally rendered rollout GIF (see `_ensure_local_rollout_gif` and
   `rl_core/envs/preview_worker.py` below) —
   every other env registered in this app (MiniGrid/Highway-env/NetHack/
   MiniHack/gymnasium-robotics wrappers, and every from-scratch env: JobShop/
   BinPacking/Trading/FinRL-*) has neither a Farama asset nor a hand-drawn
   SVG, but *does* implement `render_mode="rgb_array"` (that's what feeds
   the live GIF preview during training — see
   `rl_core/algorithms/metrics_callback.py`'s `render_episode`) — so instead
   of shipping those gallery cards with a permanent placeholder icon, a
   short random-policy rollout is recorded once and cached to disk exactly
   like the other two tiers. Not as informative as a trained agent's
   playthrough, but it's what every one of those envs actually *looks*
   like, and costs nothing to keep in sync as new envs get added (no per-
   env id to remember to list here).
"""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from rl_core.paths import PACKAGE_DIR, ROOT

GYMNASIUM_VIDEOS = (
    "https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/docs/_static/videos"
)
GYMNASIUM_VIDEOS_CDN = (
    "https://cdn.jsdelivr.net/gh/Farama-Foundation/Gymnasium@main/docs/_static/videos"
)

# Relative path under docs/_static/videos, without .gif
PREVIEW_REL: dict[str, str] = {
    "CartPole-v1": "classic_control/cart_pole",
    "MountainCar-v0": "classic_control/mountain_car",
    "MountainCarContinuous-v0": "classic_control/mountain_car_continuous",
    "Acrobot-v1": "classic_control/acrobot",
    "Pendulum-v1": "classic_control/pendulum",
    "FrozenLake-v1": "toy_text/frozen_lake",
    "Taxi-v4": "toy_text/taxi",
    "Blackjack-v1": "toy_text/blackjack",
    "CliffWalking-v1": "toy_text/cliff_walking",
    "LunarLander-v3": "box2d/lunar_lander",
    "LunarLanderContinuous-v3": "box2d/lunar_lander",
    "BipedalWalker-v3": "box2d/bipedal_walker",
    "BipedalWalkerHardcore-v3": "box2d/bipedal_walker",
    "CarRacing-v3": "box2d/car_racing",
    "InvertedPendulum-v5": "mujoco/inverted_pendulum",
    "InvertedDoublePendulum-v5": "mujoco/inverted_double_pendulum",
    "Reacher-v5": "mujoco/reacher",
    "Pusher-v5": "mujoco/pusher",
    "HalfCheetah-v5": "mujoco/half_cheetah",
    "Hopper-v5": "mujoco/hopper",
    "Walker2d-v5": "mujoco/walker2d",
    "Swimmer-v5": "mujoco/swimmer",
    "Ant-v5": "mujoco/ant",
    "Humanoid-v5": "mujoco/humanoid",
    "HumanoidStandup-v5": "mujoco/humanoid_standup",
    "ALE/Pong-v5": "atari/pong",
    "ALE/Breakout-v5": "atari/breakout",
    "ALE/SpaceInvaders-v5": "atari/space_invaders",
    "ALE/MsPacman-v5": "atari/ms_pacman",
    "ALE/Qbert-v5": "atari/qbert",
    "ALE/Seaquest-v5": "atari/seaquest",
    "ALE/Enduro-v5": "atari/enduro",
    "ALE/BeamRider-v5": "atari/beam_rider",
    "ALE/Asteroids-v5": "atari/asteroids",
    "ALE/Boxing-v5": "atari/boxing",
    "ALE/Freeway-v5": "atari/freeway",
    "ALE/Frostbite-v5": "atari/frostbite",
    "ALE/Riverraid-v5": "atari/riverraid",
    "ALE/RoadRunner-v5": "atari/road_runner",
    "ALE/Assault-v5": "atari/assault",
    "ALE/Atlantis-v5": "atari/atlantis",
    "ALE/BattleZone-v5": "atari/battle_zone",
    "ALE/Centipede-v5": "atari/centipede",
    "ALE/ChopperCommand-v5": "atari/chopper_command",
    "ALE/CrazyClimber-v5": "atari/crazy_climber",
    "ALE/DemonAttack-v5": "atari/demon_attack",
    "ALE/FishingDerby-v5": "atari/fishing_derby",
    "ALE/Gopher-v5": "atari/gopher",
    "ALE/Hero-v5": "atari/hero",
    "ALE/IceHockey-v5": "atari/ice_hockey",
    "ALE/Kangaroo-v5": "atari/kangaroo",
    "ALE/KungFuMaster-v5": "atari/kung_fu_master",
    "ALE/MontezumaRevenge-v5": "atari/montezuma_revenge",
    "ALE/Pitfall-v5": "atari/pitfall",
    "ALE/PrivateEye-v5": "atari/private_eye",
    "ALE/Skiing-v5": "atari/skiing",
    "ALE/Tennis-v5": "atari/tennis",
    "ALE/Asterix-v5": "atari/asterix",
    "ALE/Phoenix-v5": "atari/phoenix",
    "ALE/TimePilot-v5": "atari/time_pilot",
    "ALE/UpNDown-v5": "atari/up_n_down",
    "ALE/VideoPinball-v5": "atari/video_pinball",
    "ALE/WizardOfWor-v5": "atari/wizard_of_wor",
    "ALE/Zaxxon-v5": "atari/zaxxon",
    # POMDP variants reuse the base env's official GIF — the physics and
    # on-screen animation are identical, only what the *agent* observes
    # differs (see rl_core/envs/pomdp.py), so the base env's video is still
    # an accurate preview.
    "POCartPole-v0": "classic_control/cart_pole",
    "POPendulum-v0": "classic_control/pendulum",
    "POMountainCar-v0": "classic_control/mountain_car",
    "POAcrobot-v0": "classic_control/acrobot",
    "POLunarLander-v0": "box2d/lunar_lander",
    "FlickeringCartPole-v0": "classic_control/cart_pole",
    "FlickeringPong-v0": "atari/pong",
}

# Bundled (locally-shipped) SVG previews for envs with no matching Farama
# Foundation asset — board games and the from-scratch Memory Corridor task.
BUNDLED_SVG_PREVIEWS: dict[str, Path] = {
    "tic_tac_toe": PACKAGE_DIR / "envs" / "previews" / "tic_tac_toe.svg",
    "connect_four": PACKAGE_DIR / "envs" / "previews" / "connect_four.svg",
    "gomoku": PACKAGE_DIR / "envs" / "previews" / "gomoku.svg",
    "MemoryCorridor-v0": PACKAGE_DIR / "envs" / "previews" / "memory_corridor.svg",
    # Same task, just a longer delay — the static preview card doesn't need
    # to look different to be an accurate illustration.
    "MemoryCorridorLong-v0": PACKAGE_DIR / "envs" / "previews" / "memory_corridor.svg",
    "RepeatPrevious-v0": PACKAGE_DIR / "envs" / "previews" / "repeat_previous.svg",
    "RockSample-v0": PACKAGE_DIR / "envs" / "previews" / "rock_sample.svg",
    "VisualMemoryMaze-v0": PACKAGE_DIR / "envs" / "previews" / "visual_memory_maze.svg",
    "Tiger-v0": PACKAGE_DIR / "envs" / "previews" / "tiger.svg",
    "HeavenHell-v0": PACKAGE_DIR / "envs" / "previews" / "heaven_hell.svg",
    "Hallway-v0": PACKAGE_DIR / "envs" / "previews" / "hallway.svg",
    "Battleship-v0": PACKAGE_DIR / "envs" / "previews" / "battleship.svg",
    "MinesweeperPOMDP-v0": PACKAGE_DIR / "envs" / "previews" / "minesweeper_pomdp.svg",
    "Concentration-v0": PACKAGE_DIR / "envs" / "previews" / "concentration.svg",
    "LaserTag-v0": PACKAGE_DIR / "envs" / "previews" / "laser_tag.svg",
    "ActiveTMaze-v0": PACKAGE_DIR / "envs" / "previews" / "active_tmaze.svg",
    # Same task, just a longer corridor — the static preview doesn't need to
    # look different to be an accurate illustration.
    "ActiveTMazeLong-v0": PACKAGE_DIR / "envs" / "previews" / "active_tmaze.svg",
}

BUNDLED_DIR = PACKAGE_DIR / "envs" / "previews"
PREVIEW_CACHE_DIR = ROOT / "previews"
THUMB_MAX_WIDTH = 480
MEDIA_TYPES = {
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


# Env ids known to have neither a Farama asset nor be a real registered
# gym env we can `render()` (board-game/scene ids that aren't `gym.make()`-
# able, plus anything that's already failed a local-rollout attempt this
# process) — cached negatively so a broken/unrenderable id doesn't retry
# (and re-raise/re-log) on every single gallery request.
_local_render_unsupported: set[str] = set()


def has_preview(env_id: str) -> bool:
    if env_id in PREVIEW_REL or env_id in BUNDLED_SVG_PREVIEWS:
        return True
    # Every other env id that reaches `preview_api_path()` (see
    # `list_environments()` in registry.py) is a real, registered
    # `gym.make()`-able id — board games and Scene Builder scenes are
    # either already covered above (tier 2) or never call this at all — so
    # assume tier 3 applies; a bad guess here just costs one wasted 404
    # from `resolve_preview_file()`, not a crash.
    return env_id not in _local_render_unsupported


def _stem(env_id: str) -> str:
    return env_id.replace("/", "_")


def _gif_cache_path(env_id: str) -> Path:
    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PREVIEW_CACHE_DIR / f"{_stem(env_id)}.gif"


def _thumb_cache_path(env_id: str) -> Path:
    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PREVIEW_CACHE_DIR / f"{_stem(env_id)}.jpg"


def _bundled_thumb(env_id: str) -> Path | None:
    path = BUNDLED_DIR / f"{_stem(env_id)}.jpg"
    return path if path.is_file() else None


def _download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rl-studio/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            if getattr(resp, "status", 200) != 200:
                return False
            data = resp.read()
        if len(data) < 32:
            return False
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _gif_to_thumb(gif_path: Path, dest: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(gif_path) as img:
            # A third of the way into the animation, not frame 0 — several
            # of this app's own envs render an empty/just-reset first frame
            # (an unscheduled Gantt chart, a flat price history, an empty
            # bin-packing floor, ...), which makes for a far less
            # recognizable static thumbnail than a frame with the episode
            # actually underway. Harmless for Farama's own GIFs too (their
            # first frame is already mid-episode-ish).
            n_frames = getattr(img, "n_frames", 1)
            img.seek(min(n_frames - 1, n_frames // 3))
            frame = img.convert("RGB")
            frame.thumbnail((THUMB_MAX_WIDTH, THUMB_MAX_WIDTH), Image.Resampling.LANCZOS)
            dest.parent.mkdir(parents=True, exist_ok=True)
            frame.save(dest, format="JPEG", quality=72, optimize=True)
        return dest.is_file() and dest.stat().st_size > 32
    except Exception:
        return False


def _ensure_gif(env_id: str) -> Path | None:
    cached = _gif_cache_path(env_id)
    if cached.is_file() and cached.stat().st_size > 32:
        return cached
    rel = PREVIEW_REL.get(env_id)
    if not rel:
        return None
    for base in (GYMNASIUM_VIDEOS, GYMNASIUM_VIDEOS_CDN):
        if _download(f"{base}/{rel}.gif", cached):
            return cached
    return None


# Hard caps on a generated local-rollout preview — independent of the
# cache-once-forever nature of the result, so one slow-to-render env
# (MuJoCo/NLE/MiniHack) can't turn a lazy background thumbnail request into
# a many-second stall the first time its gallery card scrolls into view.
_LOCAL_ROLLOUT_MAX_STEPS = 150
_LOCAL_ROLLOUT_MAX_FRAMES = 60
_LOCAL_ROLLOUT_MAX_SIDE = 320
_LOCAL_ROLLOUT_SUBPROCESS_TIMEOUT_S = 60


def _rollout_frames(env_id: str) -> tuple[list[np.ndarray], float] | None:
    """A short random-policy rollout, restarting the episode (with a fresh
    seed) whenever it ends early — several of this app's own from-scratch
    envs (`JobShop-*-v0`, `BinPacking-v0`, ...) run for well under
    `_LOCAL_ROLLOUT_MAX_STEPS` steps on a random policy, and a 2-frame GIF
    is a worse preview than a longer one stitched from a few short
    episodes back to back.

    Only ever called from `rl_core/envs/preview_worker.py`'s subprocess,
    never in-process — see `_ensure_local_rollout_gif()`'s docstring for
    why."""
    try:
        from rl_core.envs import registry  # noqa: F401  (import side-effect: registers every custom env id)
        import gymnasium as gym

        env = gym.make(env_id, render_mode="rgb_array")
    except Exception:
        return None

    frames: list[np.ndarray] = []
    fps = float(getattr(env, "metadata", {}).get("render_fps", 8) or 8)
    try:
        env.reset(seed=0)
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        episode_seed = 1
        for _ in range(_LOCAL_ROLLOUT_MAX_STEPS):
            action = env.action_space.sample()
            _obs, _reward, terminated, truncated, _info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
            if terminated or truncated:
                env.reset(seed=episode_seed)
                episode_seed += 1
    except Exception:
        pass  # keep whatever frames were already collected — a short preview beats none
    finally:
        env.close()
    return (frames, fps) if frames else None


def build_local_rollout_gif_bytes(env_id: str) -> bytes | None:
    """`_rollout_frames()` + GIF encoding — called exclusively from
    `rl_core/envs/preview_worker.py`'s `__main__`, never in-process (public/
    unprefixed so that subprocess entry point can import it cleanly)."""
    result = _rollout_frames(env_id)
    if result is None:
        return None
    frames, fps = result
    # Lazy import: keeps stable-baselines3 off the import path of every
    # other function in this module (used by the always-hot /environments
    # list endpoint) — only the preview_worker subprocess ever needs it.
    from rl_core.algorithms.metrics_callback import _build_gif_bytes

    return _build_gif_bytes(frames, fps, _LOCAL_ROLLOUT_MAX_FRAMES, _LOCAL_ROLLOUT_MAX_SIDE)


def _render_rollout_gif_in_subprocess(env_id: str, dest: Path) -> bool:
    """Runs the actual rollout+render+encode (`build_local_rollout_gif_bytes`)
    in a throwaway `python -m rl_core.envs.preview_worker` subprocess rather
    than in-process.

    Why: some envs' OpenGL rendering backends don't fail with a catchable
    Python exception when there's no display to render to — verified
    directly for MuJoCo's default GLFW backend (used by the wrapped
    `PointMaze`/`AntMaze` envs in `rl_core/envs/robotics_envs.py`) on a
    headless/no-X-server machine: it segfaults the *entire interpreter*
    (`Fatal Python error: pygame_parachute: Segmentation Fault`), no
    `except Exception` anywhere could ever catch that. In-process, one
    gallery card scrolling into view would be enough to crash the whole
    FastAPI backend; out-of-process, the exact same crash just means "this
    subprocess exited non-zero", handled below identically to any other
    rendering failure."""
    import subprocess
    import sys

    from rl_core.paths import PROJECT_ROOT

    try:
        result = subprocess.run(
            [sys.executable, "-m", "rl_core.envs.preview_worker", env_id, str(dest)],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=_LOCAL_ROLLOUT_SUBPROCESS_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0 and dest.is_file() and dest.stat().st_size > 32


def _ensure_local_rollout_gif(env_id: str) -> Path | None:
    cached = _gif_cache_path(env_id)
    if cached.is_file() and cached.stat().st_size > 32:
        return cached
    if env_id in _local_render_unsupported:
        return None
    if _render_rollout_gif_in_subprocess(env_id, cached):
        return cached
    _local_render_unsupported.add(env_id)
    return None


def resolve_preview_file(env_id: str, *, thumb: bool = False) -> Path | None:
    """Return a local preview file. `thumb=True` is a small first-frame JPEG."""
    if env_id in BUNDLED_SVG_PREVIEWS:
        path = BUNDLED_SVG_PREVIEWS[env_id]
        return path if path.is_file() else None

    # Tier 1 (Farama) if we have one, else tier 3 (local rollout) — see
    # module docstring.
    ensure_gif = _ensure_gif if env_id in PREVIEW_REL else _ensure_local_rollout_gif

    if thumb:
        bundled = _bundled_thumb(env_id)
        if bundled:
            return bundled
        thumb_path = _thumb_cache_path(env_id)
        if thumb_path.is_file() and thumb_path.stat().st_size > 32:
            return thumb_path
        gif = ensure_gif(env_id)
        if gif and _gif_to_thumb(gif, thumb_path):
            return thumb_path
        return gif

    return ensure_gif(env_id)


def preview_media_type(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def preview_api_path(env_id: str, *, thumb: bool = False) -> str | None:
    if not has_preview(env_id):
        return None
    from urllib.parse import quote

    q = f"/environments/preview?id={quote(env_id, safe='')}"
    return f"{q}&thumb=1" if thumb else q

"""Official Farama Foundation preview GIFs for the Environments gallery.

The first ten classic/toy/box2d cards already used these files from the
Gymnasium repo. New envs use the same source; we proxy + cache them so the
Electron file:// renderer does not depend on GitHub being reachable from
<img> tags (and so ALE ids with slashes have a stable URL).
"""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

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
}

BOARD_PREVIEW_FILES: dict[str, Path] = {
    "tic_tac_toe": PACKAGE_DIR / "envs" / "previews" / "tic_tac_toe.svg",
    "connect_four": PACKAGE_DIR / "envs" / "previews" / "connect_four.svg",
    "gomoku": PACKAGE_DIR / "envs" / "previews" / "gomoku.svg",
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


def has_preview(env_id: str) -> bool:
    return env_id in PREVIEW_REL or env_id in BOARD_PREVIEW_FILES


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


def resolve_preview_file(env_id: str, *, thumb: bool = False) -> Path | None:
    """Return a local preview file. `thumb=True` is a small first-frame JPEG."""
    if env_id in BOARD_PREVIEW_FILES:
        path = BOARD_PREVIEW_FILES[env_id]
        return path if path.is_file() else None

    if env_id not in PREVIEW_REL:
        return None

    if thumb:
        bundled = _bundled_thumb(env_id)
        if bundled:
            return bundled
        thumb_path = _thumb_cache_path(env_id)
        if thumb_path.is_file() and thumb_path.stat().st_size > 32:
            return thumb_path
        gif = _ensure_gif(env_id)
        if gif and _gif_to_thumb(gif, thumb_path):
            return thumb_path
        return gif

    return _ensure_gif(env_id)


def preview_media_type(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def preview_api_path(env_id: str, *, thumb: bool = False) -> str | None:
    if not has_preview(env_id):
        return None
    from urllib.parse import quote

    q = f"/environments/preview?id={quote(env_id, safe='')}"
    return f"{q}&thumb=1" if thumb else q

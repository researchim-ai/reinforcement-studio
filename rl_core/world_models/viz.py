"""Visualizations for World Models — pure Pillow, no matplotlib/sklearn (an
`environments.py`/`previews.py`-wide deliberate choice in this app: avoid
another heavy/fragile dependency for what's fundamentally just "draw some
shapes"). Two artifacts, both served the same way any other run's live
preview already is (`GET /training/runs/{run_id}/preview.gif` /
a new `.../latent_space.png`):

- **Imagined-vs-real GIF** (`render_imagined_vs_real_gif`) — the single
  most direct way to *see* whether a world model is any good: the same
  action sequence played out in the real env side-by-side with what the
  model predicted would happen, frame by frame. Works identically for
  image and vector observations (the latter drawn as a small bar chart per
  frame instead of literal pixels — see `_obs_to_frame`), so every world
  model family/environment combination in this app gets the same kind of
  preview without any env-specific code here.
- **Latent space scatter** (`render_latent_scatter`) — a 2D PCA projection
  (plain `numpy.linalg.svd`, no sklearn) of a rollout's latent states,
  colored by time — a cheap way to see whether the model's latent space
  has learned any obviously meaningful structure (e.g. distinct clusters
  per "phase" of an episode) versus collapsed to a single point.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from rl_core.algorithms.native.preprocessing import is_image_space

_FRAME_SIZE = (160, 160)
_LABEL_HEIGHT = 18
_PAD = 8


def _obs_to_frame(obs_arr, observation_space, size: tuple[int, int] = _FRAME_SIZE) -> Image.Image:
    """One observation -> a fixed-size RGB frame for display. Image
    observations get resized as-is (nearest-neighbor, so small crops stay
    crisp instead of blurring); vector observations get drawn as a simple
    per-dimension bar chart (positive bars up, negative down from a
    midline) — capped at the first 24 dimensions so a large observation
    vector doesn't turn into an illegibly thin sliver per bar."""
    if is_image_space(observation_space):
        arr = np.asarray(obs_arr, dtype=np.float32)
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]
        return Image.fromarray(arr).convert("RGB").resize(size, Image.NEAREST)

    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    vec = np.asarray(obs_arr, dtype=np.float32).reshape(-1)[:24]
    n = max(1, len(vec))
    v_max = max(1e-6, float(np.max(np.abs(vec))))
    bar_w = size[0] / n
    mid = size[1] // 2
    for i, v in enumerate(vec):
        height = int(abs(v) / v_max * (size[1] / 2 - 4))
        x0, x1 = int(i * bar_w) + 1, int((i + 1) * bar_w) - 1
        color = (60, 120, 220) if v >= 0 else (220, 90, 60)
        if v >= 0:
            draw.rectangle([x0, mid - height, x1, mid], fill=color)
        else:
            draw.rectangle([x0, mid, x1, mid + height], fill=color)
    draw.line([(0, mid), (size[0], mid)], fill=(190, 190, 190))
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(210, 210, 210))
    return img


def render_imagined_vs_real_gif(
    real_obs_seq: list, imagined_obs_seq: list, observation_space, fps: int = 6,
) -> bytes:
    """`real_obs_seq`/`imagined_obs_seq`: equal-length lists of raw
    observations (same convention `_obs_to_frame` expects) — one real
    episode and the world model's own imagined rollout under the *same*
    action sequence, frame-aligned. Returns GIF bytes ready to write
    straight to disk or embed as base64, matching every other GIF this app
    already produces (see `rl_core/algorithms/metrics_callback.py`)."""
    n = min(len(real_obs_seq), len(imagined_obs_seq))
    width = _FRAME_SIZE[0] * 2 + _PAD * 3
    height = _FRAME_SIZE[1] + _LABEL_HEIGHT + _PAD * 2
    frames = []
    for i in range(n):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((_PAD, 4), f"Реальность (шаг {i})", fill="black")
        draw.text((_PAD * 2 + _FRAME_SIZE[0], 4), "Воображение модели", fill=(120, 40, 160))
        canvas.paste(_obs_to_frame(real_obs_seq[i], observation_space), (_PAD, _LABEL_HEIGHT + _PAD))
        canvas.paste(_obs_to_frame(imagined_obs_seq[i], observation_space), (_PAD * 2 + _FRAME_SIZE[0], _LABEL_HEIGHT + _PAD))
        frames.append(canvas)
    if not frames:
        frames = [Image.new("RGB", (width, height), "white")]
    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=max(1, int(1000 / fps)), loop=0,
    )
    return buf.getvalue()


def render_latent_scatter(latents: np.ndarray, size: tuple[int, int] = (360, 360)) -> bytes:
    """`latents`: `(N, D)` — one rollout's latent state at every step
    (`D` >= 1; `D` > 2 gets projected to its top-2 principal components via
    a plain SVD, no sklearn dependency needed for just two components).
    Points are colored on a blue -> magenta gradient by their position in
    time, so a trajectory that visibly "moves" through latent space (vs.
    one collapsed to a single blob) is immediately visible."""
    latents = np.asarray(latents, dtype=np.float64)
    if latents.ndim == 1:
        latents = latents.reshape(-1, 1)
    centered = latents - latents.mean(axis=0, keepdims=True)
    if centered.shape[1] >= 2:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vt[:2].T
    else:
        proj = np.concatenate([centered, np.zeros_like(centered)], axis=1)

    pad = 24
    mins, maxs = proj.min(axis=0), proj.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    n = len(proj)
    prev_xy = None
    for i, (x, y) in enumerate(proj):
        px = pad + (x - mins[0]) / span[0] * (size[0] - 2 * pad)
        py = size[1] - pad - (y - mins[1]) / span[1] * (size[1] - 2 * pad)
        t = i / max(1, n - 1)
        color = (int(60 + 140 * t), int(70 + 20 * (1 - t)), int(220 - 60 * t))
        if prev_xy is not None:
            draw.line([prev_xy, (px, py)], fill=(220, 220, 220), width=1)
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=color)
        prev_xy = (px, py)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(210, 210, 210))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

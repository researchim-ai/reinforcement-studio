"""Generates the RL Studio app icon (a simple agent<->environment loop mark).

Run once: python3 build/generate_icon.py
Produces build/icon.png (1024x1024) and build/icons/<size>x<size>.png for
Linux packaging, plus build/icon.ico for Windows. electron-builder derives
the macOS .icns automatically from build/icon.png at build time.
"""
from __future__ import annotations

import io
import math
import os
import struct

from PIL import Image, ImageDraw

SIZE = 1024
BG = (18, 18, 22, 255)
BG_INNER = (26, 27, 33, 255)
ACCENT = (110, 142, 246, 255)
ACCENT_SOFT = (110, 142, 246, 90)
WHITE = (240, 241, 245, 255)


def rounded_square(draw: ImageDraw.ImageDraw, size: int) -> None:
    radius = int(size * 0.22)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=BG)
    margin = int(size * 0.06)
    draw.rounded_rectangle(
        [margin, margin, size - 1 - margin, size - 1 - margin],
        radius=int(radius * 0.85),
        fill=BG_INNER,
    )


def agent_env_loop(img: Image.Image, size: int) -> None:
    """Draws two arcs (agent -> action -> environment -> reward/state -> agent)
    forming a circular feedback loop, which is the canonical RL diagram."""
    draw = ImageDraw.Draw(img, 'RGBA')
    cx, cy = size / 2, size / 2
    r = size * 0.27
    width = max(2, int(size * 0.045))

    bbox = [cx - r, cy - r, cx + r, cy + r]
    # Two arcs with gaps, like a refresh/cycle icon.
    draw.arc(bbox, start=-200, end=20, fill=ACCENT, width=width)
    draw.arc(bbox, start=-20, end=160, fill=WHITE, width=width)

    def point_on_circle(angle_deg: float) -> tuple[float, float]:
        a = math.radians(angle_deg)
        return cx + r * math.cos(a), cy + r * math.sin(a)

    def arrow_head(angle_deg: float, color: tuple[int, int, int, int]) -> None:
        tip = point_on_circle(angle_deg)
        back = math.radians(angle_deg + 90)
        s = size * 0.055
        p1 = (tip[0] + s * math.cos(back + math.pi * 0.8), tip[1] + s * math.sin(back + math.pi * 0.8))
        p2 = (tip[0] + s * math.cos(back - math.pi * 0.8), tip[1] + s * math.sin(back - math.pi * 0.8))
        draw.polygon([tip, p1, p2], fill=color)

    arrow_head(20, ACCENT)
    arrow_head(160, WHITE)

    # Agent node (circle) and Environment node (rounded square) on the loop.
    node_r = size * 0.1
    ax, ay = point_on_circle(-90)
    ex, ey = point_on_circle(90)

    draw.ellipse([ax - node_r, ay - node_r, ax + node_r, ay + node_r], fill=ACCENT)
    draw.ellipse(
        [ax - node_r * 0.5, ay - node_r * 0.5, ax + node_r * 0.5, ay + node_r * 0.5],
        fill=BG_INNER,
    )

    sq = node_r * 0.95
    draw.rounded_rectangle([ex - sq, ey - sq, ex + sq, ey + sq], radius=sq * 0.35, fill=WHITE)
    inner = sq * 0.45
    draw.rounded_rectangle(
        [ex - inner, ey - inner, ex + inner, ey + inner], radius=inner * 0.35, fill=BG_INNER
    )


ICNS_TYPES = {
    16: b'icp4',
    32: b'icp5',
    64: b'icp6',
    128: b'ic07',
    256: b'ic08',
    512: b'ic09',
    1024: b'ic10',
}


def write_icns(icon: Image.Image, path: str) -> None:
    """Minimal ICNS writer: each entry is a PNG blob tagged with its OSType."""
    entries = b''
    for size, tag in ICNS_TYPES.items():
        buf = io.BytesIO()
        icon.resize((size, size), Image.LANCZOS).save(buf, format='PNG')
        data = buf.getvalue()
        entries += tag + struct.pack('>I', 8 + len(data)) + data

    header = b'icns' + struct.pack('>I', 8 + len(entries))
    with open(path, 'wb') as f:
        f.write(header + entries)


def build_icon() -> Image.Image:
    img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, 'RGBA')
    rounded_square(draw, SIZE)
    agent_env_loop(img, SIZE)
    return img


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    icon = build_icon()
    icon.save(os.path.join(here, 'icon.png'))

    icons_dir = os.path.join(here, 'icons')
    os.makedirs(icons_dir, exist_ok=True)
    for s in (16, 24, 32, 48, 64, 128, 256, 512, 1024):
        resized = icon.resize((s, s), Image.LANCZOS)
        resized.save(os.path.join(icons_dir, f'{s}x{s}.png'))

    ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icon.save(os.path.join(here, 'icon.ico'), sizes=ico_sizes)

    write_icns(icon, os.path.join(here, 'icon.icns'))

    print('Icons written to', here)


if __name__ == '__main__':
    main()

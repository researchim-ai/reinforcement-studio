import { useMemo } from 'react'
import * as THREE from 'three'
import type { SceneMaterial, SceneMaterialPattern, SceneShape } from '@/api/types'

export const SHAPE_OPTIONS: { id: SceneShape; label: string }[] = [
  { id: 'box', label: 'Куб' },
  { id: 'sphere', label: 'Сфера' },
  { id: 'cylinder', label: 'Цилиндр' },
  { id: 'cone', label: 'Конус' },
  { id: 'capsule', label: 'Капсула' },
  { id: 'pyramid', label: 'Пирамида' },
  { id: 'crystal', label: 'Кристалл' },
]

export const PATTERN_OPTIONS: { id: SceneMaterialPattern; label: string }[] = [
  { id: 'solid', label: 'Сплошной' },
  { id: 'checker', label: 'Шахматка' },
  { id: 'stripes', label: 'Полоски' },
  { id: 'noise', label: 'Шум' },
  { id: 'brick', label: 'Кирпич' },
  { id: 'dots', label: 'Точки' },
]

const _cache = new Map<string, THREE.CanvasTexture>()

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16)
  if (Number.isNaN(n)) return [128, 128, 128]
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

function fill(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, color: string) {
  ctx.fillStyle = color
  ctx.fillRect(x, y, w, h)
}

/** Build (and cache) a procedural CanvasTexture — no image assets needed. */
export function makeProceduralTexture(
  pattern: SceneMaterialPattern,
  color: string,
  color2: string,
  size = 128,
): THREE.CanvasTexture | null {
  if (pattern === 'solid') return null
  const key = `${pattern}|${color}|${color2}|${size}`
  const hit = _cache.get(key)
  if (hit) return hit

  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  const [r1, g1, b1] = hexToRgb(color)
  const [r2, g2, b2] = hexToRgb(color2)
  const c1 = `rgb(${r1},${g1},${b1})`
  const c2 = `rgb(${r2},${g2},${b2})`

  if (pattern === 'checker') {
    const cell = size / 8
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        fill(ctx, x * cell, y * cell, cell, cell, (x + y) % 2 === 0 ? c1 : c2)
      }
    }
  } else if (pattern === 'stripes') {
    const band = size / 10
    for (let i = 0; i < 10; i++) {
      fill(ctx, 0, i * band, size, band, i % 2 === 0 ? c1 : c2)
    }
  } else if (pattern === 'noise') {
    fill(ctx, 0, 0, size, size, c1)
    const img = ctx.getImageData(0, 0, size, size)
    for (let i = 0; i < img.data.length; i += 4) {
      const t = Math.random()
      img.data[i] = Math.round(r1 * (1 - t) + r2 * t)
      img.data[i + 1] = Math.round(g1 * (1 - t) + g2 * t)
      img.data[i + 2] = Math.round(b1 * (1 - t) + b2 * t)
    }
    ctx.putImageData(img, 0, 0)
  } else if (pattern === 'brick') {
    fill(ctx, 0, 0, size, size, c2)
    const rows = 6
    const cols = 4
    const bh = size / rows
    const bw = size / cols
    const gap = 2
    for (let row = 0; row < rows; row++) {
      const offset = row % 2 === 0 ? 0 : bw / 2
      for (let col = -1; col <= cols; col++) {
        fill(ctx, col * bw + offset + gap, row * bh + gap, bw - gap * 2, bh - gap * 2, c1)
      }
    }
  } else if (pattern === 'dots') {
    fill(ctx, 0, 0, size, size, c1)
    ctx.fillStyle = c2
    const step = size / 6
    for (let y = 0; y < 6; y++) {
      for (let x = 0; x < 6; x++) {
        ctx.beginPath()
        ctx.arc((x + 0.5) * step, (y + 0.5) * step, step * 0.22, 0, Math.PI * 2)
        ctx.fill()
      }
    }
  }

  const tex = new THREE.CanvasTexture(canvas)
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping
  tex.colorSpace = THREE.SRGBColorSpace
  tex.needsUpdate = true
  _cache.set(key, tex)
  return tex
}

export function useSceneMaterialProps(material: SceneMaterial | undefined, fallbackColor: string) {
  return useMemo(() => {
    const pattern = material?.pattern ?? 'solid'
    const color = material?.color ?? fallbackColor
    const color2 = material?.color2 ?? '#1f2937'
    const map = makeProceduralTexture(pattern, color, color2)
    return {
      color: pattern === 'solid' ? color : '#ffffff',
      map: map ?? undefined,
      roughness: material?.roughness ?? 0.65,
      metalness: material?.metalness ?? 0.05,
      emissive: material?.emissive ? color : '#000000',
      emissiveIntensity: material?.emissive ? 0.35 : 0,
    }
  }, [material?.pattern, material?.color, material?.color2, material?.roughness, material?.metalness, material?.emissive, fallbackColor])
}

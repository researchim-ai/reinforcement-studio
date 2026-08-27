import { useRef } from 'react'
import { TransformControls } from '@react-three/drei'
import type { ThreeEvent } from '@react-three/fiber'
import type { Object3D } from 'three'
import type { SceneMaterial, SceneShape } from '@/api/types'
import { useSceneMaterialProps } from '@/lib/sceneMaterials'

function ShapeGeometry({ shape, size, radius }: { shape: SceneShape; size?: [number, number, number]; radius?: number }) {
  const sx = size?.[0] ?? (radius ?? 0.5) * 2
  const sy = size?.[1] ?? (radius ?? 0.5) * 2
  const sz = size?.[2] ?? (radius ?? 0.5) * 2
  const r = radius ?? Math.min(sx, sy, sz) / 2

  switch (shape) {
    case 'sphere':
      return <sphereGeometry args={[r, 20, 20]} />
    case 'cylinder':
      return <cylinderGeometry args={[Math.min(sx, sz) / 2, Math.min(sx, sz) / 2, sy, 20]} />
    case 'cone':
      return <coneGeometry args={[Math.min(sx, sz) / 2, sy, 20]} />
    case 'capsule':
      return <capsuleGeometry args={[Math.min(sx, sz) / 2 * 0.85, Math.max(0.01, sy - Math.min(sx, sz) * 0.85), 8, 16]} />
    case 'pyramid':
      return <coneGeometry args={[Math.min(sx, sz) / 2, sy, 4]} />
    case 'crystal':
      return <octahedronGeometry args={[r, 0]} />
    case 'box':
    default:
      return <boxGeometry args={[sx, sy, sz]} />
  }
}

export function SelectableMesh({
  position,
  size,
  radius,
  shape = 'box',
  material,
  fallbackColor,
  selected,
  onSelect,
  onTransform,
  yLift = 0,
}: {
  position: [number, number, number]
  size?: [number, number, number]
  radius?: number
  shape?: SceneShape
  material?: SceneMaterial
  fallbackColor: string
  selected: boolean
  onSelect: () => void
  onTransform: (pos: [number, number, number], size?: [number, number, number]) => void
  yLift?: number
}) {
  const meshRef = useRef<Object3D>(null)
  const mat = useSceneMaterialProps(material, fallbackColor)
  const pos: [number, number, number] = [position[0], position[1] + yLift, position[2]]

  return (
    <>
      {selected && meshRef.current && (
        <TransformControls
          object={meshRef.current}
          mode="translate"
          onMouseUp={() => {
            if (!meshRef.current) return
            const p = meshRef.current.position
            onTransform([p.x, p.y - yLift, p.z], size)
          }}
        />
      )}
      <mesh
        ref={meshRef as never}
        position={pos}
        onClick={(e: ThreeEvent<MouseEvent>) => { e.stopPropagation(); onSelect() }}
        castShadow
        receiveShadow
      >
        <ShapeGeometry shape={shape} size={size} radius={radius} />
        <meshStandardMaterial
          color={selected ? '#60a5fa' : mat.color}
          map={selected ? undefined : mat.map}
          roughness={mat.roughness}
          metalness={mat.metalness}
          emissive={selected ? '#1e3a8a' : mat.emissive}
          emissiveIntensity={selected ? 0.2 : mat.emissiveIntensity}
        />
      </mesh>
    </>
  )
}

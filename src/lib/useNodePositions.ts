import { useCallback, useRef, useState } from 'react'
import type { Node, NodeChange } from '@xyflow/react'

interface CacheEntry {
  node: Node
  base: Node
  x: number
  y: number
  width: number | undefined
  height: number | undefined
}

/** Lets any of our ReactFlow canvases (Experiment Designer, Network
 * Builder) be dragged around while everything else about the graph keeps
 * being recomputed from app state on every change (hyperparam edit, new
 * layer, ...). Without this, nodes snap back to their auto-layout position
 * on the next render because that render rebuilds the node list from
 * scratch with hardcoded positions — dragging only ever mutated ReactFlow's
 * own internal/uncontrolled state, which gets thrown away as soon as the
 * parent passes a fresh `nodes` prop.
 *
 * Usage: build your nodes with the usual auto-layout `position`, then call
 * `applyPositions(nodes)` right before handing them to <ReactFlow>, and
 * wire `onNodesChange` straight through.
 *
 * Why this also tracks `measured` dimensions (and not just position): React
 * Flow decides whether a node object is "the same as before" by strict
 * reference equality (`userNode === internalNode.internals.userNode`). Any
 * time we hand it a *new* object for a given node id — which naively
 * happens on every render once that node has a position override, since
 * `{ ...n, position: pos }` creates a fresh object every call — it treats
 * the node as freshly (re)initialized and resets its internal `measured`
 * size to `undefined` until the node is re-measured a moment later. While
 * unmeasured, React Flow renders the node with `visibility: hidden`. That
 * one-frame hide/show cycle, repeated on every render that happens to touch
 * this hook's state (e.g. while dragging *any other* node, since that also
 * flows through `setPositions`), is exactly what shows up as flicker. The
 * cache below hands back the *same* object reference — carrying its last
 * known `measured` size along with it — whenever a node's own position and
 * size genuinely haven't changed, so React Flow never has a reason to treat
 * it as new. */
export function useNodePositions() {
  const [positions, setPositions] = useState<Record<string, { x: number; y: number }>>({})
  const [measured, setMeasured] = useState<Record<string, { width?: number; height?: number }>>({})
  const cacheRef = useRef(new Map<string, CacheEntry>())

  const onNodesChange = useCallback((changes: NodeChange[]) => {
    setPositions((prev) => {
      let next = prev
      for (const change of changes) {
        if (change.type === 'position' && change.position) {
          if (next === prev) next = { ...prev }
          next[change.id] = change.position
        }
      }
      return next
    })
    setMeasured((prev) => {
      let next = prev
      for (const change of changes) {
        if (change.type === 'dimensions' && change.dimensions) {
          const { width, height } = change.dimensions
          const existing = prev[change.id]
          if (existing?.width === width && existing?.height === height) continue
          if (next === prev) next = { ...prev }
          next[change.id] = change.dimensions
        }
      }
      return next
    })
  }, [])

  const applyPositions = useCallback(
    <T extends Node>(nodes: T[]): T[] =>
      nodes.map((n) => {
        const pos = positions[n.id]
        const dims = measured[n.id]
        if (!pos && !dims) return n

        const x = pos?.x ?? n.position.x
        const y = pos?.y ?? n.position.y
        const width = dims?.width
        const height = dims?.height

        const cached = cacheRef.current.get(n.id)
        if (cached && cached.base === n && cached.x === x && cached.y === y && cached.width === width && cached.height === height) {
          return cached.node as T
        }

        const node = { ...n, position: { x, y }, ...(dims ? { measured: dims } : {}) } as T
        cacheRef.current.set(n.id, { node, base: n, x, y, width, height })
        return node
      }),
    [positions, measured],
  )

  return { applyPositions, onNodesChange, positions }
}

import { useEffect } from "react"
import {
  AnimatePresence,
  animate,
  motion,
  useMotionValue,
  useReducedMotion,
  useTransform,
} from "motion/react"
import type { Detection } from "@/api/types"
import { instrumentIconFor } from "@/components/instruments/InstrumentIcons"
import { cn } from "@/lib/utils"

// Shared choreography: how tiles enter/leave and how the whole ring glides to
// redistribute when a detection joins or drops out during setup.
const EASE = [0.22, 1, 0.36, 1] as const
const MOVE_S = 0.55 // ring re-spread (make space) — tiles AND their connectors

/**
 * The setup "detection hub": a floating count pill with up to eight recognised
 * catalog instruments arranged around it as tiles, each tethered to the hub by
 * a thin connector string (the sketch's hub-and-spoke). It is the brand moment on
 * the operator's idle screen — calm, not a rainbow dashboard: the spectrum is
 * reserved for the logo and the Track glow; here the accent is teal.
 *
 * Tiles show the backend's real per-detection cutouts (`detections[].thumbnail`,
 * from `/status`). A representative icon is the graceful fallback whenever a
 * preview is unavailable — no `detections` yet, or an individual `thumbnail: null`
 * (api-contract.md §/status). The count badge always comes from `detectedCount`;
 * it intentionally includes resolving/unknown detections that are excluded from
 * the constellation tiles.
 */

const MAX_TILES = 8
const RX = 37 // ellipse radii in the 0–100 hub coordinate space
const RY = 39

type Mode = "connecting" | "empty" | "detecting" | "ready" | "stalled"

interface ConstellationProps {
  detectedCount: number
  /** Per-detection crops from `/status`. Absent/empty → representative icons. */
  detections?: Detection[]
  /** Gate open: count ≥ 1 ∧ stable ≥ 2 s ∧ capture healthy (from LiveScreen). */
  ready: boolean
  stalled: boolean
  /** No status polled yet. */
  connecting: boolean
}

interface Tile {
  key: string | number
  thumbnail: string | null
  label: string
  colour: string
  iconIndex: number
}

/**
 * The tiles to render: at most the eight recognised catalog instruments.
 * Resolving/unknown detections still contribute to the backend count and blocking
 * reason, and remain visible on the video overlay, but are deliberately not part
 * of this catalog constellation. Each recognised tile falls back to an icon when
 * it has no usable thumbnail.
 */
function tilesFor(detectedCount: number, detections?: Detection[]): Tile[] {
  if (detections !== undefined) {
    return detections
      .filter((d) => d.state === "recognised")
      .slice(0, MAX_TILES)
      .map((d) => ({
        key: d.tracker_id,
        thumbnail: d.thumbnail,
        label: d.label,
        colour: d.colour,
        // Key the fallback icon to the instrument, not its slot — so a surviving
        // tile keeps its glyph when a lower id drops out between polls.
        iconIndex: d.tracker_id,
      }))
  }
  const count = Math.min(Math.max(0, detectedCount), MAX_TILES)
  return Array.from({ length: count }, (_, i) => ({
    key: i,
    thumbnail: null,
    label: `Instrument ${i + 1}`,
    colour: "var(--primary)",
    iconIndex: i,
  }))
}

function positionsFor(n: number) {
  return Array.from({ length: n }, (_, i) => {
    const angle = ((-90 + (360 / n) * i) * Math.PI) / 180
    return { x: 50 + RX * Math.cos(angle), y: 50 + RY * Math.sin(angle) }
  })
}

/**
 * A gently bowed connector from a tile to the hub centre. The container is
 * `aspect-square`, so the 0–100 space maps uniformly and the curve renders
 * true. All curves bow the same rotational way for a calm inward swirl.
 */
function curvePath(x: number, y: number) {
  const mx = (x + 50) / 2
  const my = (y + 50) / 2
  const dx = 50 - x
  const dy = 50 - y
  const len = Math.hypot(dx, dy) || 1
  const k = 7 // bow strength
  const cx = mx + (-dy / len) * k
  const cy = my + (dx / len) * k
  return `M ${x} ${y} Q ${cx} ${cy} 50 50`
}

export function DetectionConstellation({
  detectedCount,
  detections,
  ready,
  stalled,
  connecting,
}: ConstellationProps) {
  const reduce = useReducedMotion() ?? false
  const count = Math.max(0, detectedCount)
  const mode: Mode = connecting
    ? "connecting"
    : stalled
      ? "stalled"
      : count === 0
        ? "empty"
        : ready
          ? "ready"
          : "detecting"

  // No tiles while idle/connecting — otherwise a one-frame count/snapshot skew
  // could ring the "waiting for the tray" hub with live tiles.
  const tiles =
    mode === "empty" || mode === "connecting"
      ? []
      : tilesFor(detectedCount, detections)
  // Position is by slot: the ring re-spreads for the CURRENT tile count, so a
  // join/leave makes every surviving tile glide to a new angle (the "make
  // space" motion). Tile identity is the detection id, so React + Motion move
  // the same tile rather than swapping content under it.
  const positions = positionsFor(tiles.length)
  const placed = tiles.map((tile, i) => ({ tile, pos: positions[i] }))
  const flowing = mode === "detecting"
  const dimmed = mode === "stalled"
  const connectorStroke =
    mode === "ready" ? "var(--primary)" : "var(--muted-foreground)"
  const connectorOpacity = dimmed ? 0.15 : mode === "ready" ? 0.4 : 0.28

  return (
    <div
      className="relative mx-auto grid aspect-square w-full max-w-[26rem] place-items-center"
      role="group"
      aria-label={`${count} instrument${count === 1 ? "" : "s"} detected`}
    >
      {/* Connector strings — behind tiles and hub. A non-uniform stretch maps the
          0–100 space onto the container; non-scaling-stroke keeps lines crisp.
          Each string shares its tile's animated position so it stays attached
          while the ring re-spreads. */}
      <svg
        className="pointer-events-none absolute inset-0 h-full w-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <AnimatePresence>
          {placed.map(({ tile, pos }) => (
            <Connector
              key={tile.key}
              x={pos.x}
              y={pos.y}
              stroke={connectorStroke}
              opacity={connectorOpacity}
              flowing={flowing}
              reduce={reduce}
            />
          ))}
        </AnimatePresence>
      </svg>

      {/* Instrument tiles */}
      <AnimatePresence>
        {placed.map(({ tile, pos }) => (
          <Tile
            key={tile.key}
            tile={tile}
            pos={pos}
            dimmed={dimmed}
            ready={mode === "ready"}
            reduce={reduce}
          />
        ))}
      </AnimatePresence>

      {/* Hub */}
      <Hub mode={mode} count={count} reduce={reduce} />
    </div>
  )
}

/** One detection tile: eases in on join, eases out on leave, and glides to its
 *  new slot when the ring re-spreads. Idle bob lives on an inner element so it
 *  never fights Motion's transform. */
function Tile({
  tile,
  pos,
  dimmed,
  ready,
  reduce,
}: {
  tile: Tile
  pos: { x: number; y: number }
  dimmed: boolean
  ready: boolean
  reduce: boolean
}) {
  const Icon = instrumentIconFor(tile.iconIndex)
  const seed = typeof tile.key === "number" ? tile.key : 0
  return (
    <motion.div
      className="absolute"
      style={{ x: "-50%", y: "-50%" }}
      initial={
        reduce
          ? { left: `${pos.x}%`, top: `${pos.y}%`, opacity: 1, scale: 1 }
          : { left: `${pos.x}%`, top: `${pos.y}%`, opacity: 0, scale: 0.5 }
      }
      animate={{
        left: `${pos.x}%`,
        top: `${pos.y}%`,
        opacity: dimmed ? 0.4 : 1,
        scale: 1,
      }}
      exit={reduce ? { opacity: 0 } : { opacity: 0, scale: 0.5 }}
      transition={{
        left: { duration: reduce ? 0 : MOVE_S, ease: EASE },
        top: { duration: reduce ? 0 : MOVE_S, ease: EASE },
        opacity: { duration: reduce ? 0 : 0.3 },
        scale: { duration: reduce ? 0 : 0.42, ease: EASE },
      }}
    >
      <div
        data-testid="constellation-tile"
        className={cn(
          "grid size-[3.25rem] place-items-center overflow-hidden rounded-xl border-2 bg-card text-foreground/80 transition-[box-shadow] duration-300",
          "shadow-[0_6px_20px_-10px_oklch(0.22_0.025_205_/_0.35)]",
          ready && "shadow-[0_0_18px_-8px_var(--primary)]"
        )}
        style={{ borderColor: tile.colour }}
      >
        <div
          className="grid size-full place-items-center"
          style={{
            animation:
              dimmed || reduce
                ? undefined
                : `float-y ${4 + (seed % 3) * 0.6}s ease-in-out ${(seed % 4) * 0.2}s infinite`,
          }}
        >
          {tile.thumbnail ? (
            <img
              src={tile.thumbnail}
              alt={tile.label}
              className="size-full object-contain"
              draggable={false}
            />
          ) : (
            <Icon className="size-6" />
          )}
        </div>
      </div>
    </motion.div>
  )
}

/** The bowed string from a tile to the hub. Its endpoint tracks the tile via a
 *  shared motion value, so the curve morphs in lockstep with the ring re-spread
 *  instead of snapping to the tile's final spot. */
function Connector({
  x,
  y,
  stroke,
  opacity,
  flowing,
  reduce,
}: {
  x: number
  y: number
  stroke: string
  opacity: number
  flowing: boolean
  reduce: boolean
}) {
  const mx = useMotionValue(x)
  const my = useMotionValue(y)
  useEffect(() => {
    if (reduce) {
      mx.set(x)
      my.set(y)
      return
    }
    const ax = animate(mx, x, { duration: MOVE_S, ease: EASE })
    const ay = animate(my, y, { duration: MOVE_S, ease: EASE })
    return () => {
      ax.stop()
      ay.stop()
    }
  }, [x, y, reduce, mx, my])
  const d = useTransform(() => curvePath(mx.get(), my.get()))

  return (
    <motion.path
      d={d}
      fill="none"
      vectorEffect="non-scaling-stroke"
      stroke={stroke}
      strokeWidth={1}
      strokeLinecap="round"
      initial={reduce ? false : { opacity: 0 }}
      animate={{ opacity }}
      exit={{ opacity: 0 }}
      transition={{ duration: reduce ? 0 : 0.4 }}
      style={{
        transition: "stroke 300ms ease",
        strokeDasharray: flowing ? "1.5 4" : undefined,
        animation:
          flowing && !reduce ? "connector-flow 1.4s linear infinite" : undefined,
      }}
    />
  )
}

function Hub({
  mode,
  count,
  reduce,
}: {
  mode: Mode
  count: number
  reduce: boolean
}) {
  if (mode === "empty" || mode === "connecting") {
    return (
      <div className="relative z-10 grid size-[8.5rem] place-items-center rounded-full">
        {/* A connected camera gets one quiet scan cue; waiting remains still. */}
        <motion.span
          aria-hidden
          className="absolute inset-0 rounded-full border border-dashed border-border"
          initial={false}
          animate={{ rotate: mode === "connecting" && !reduce ? 360 : 0 }}
          transition={
            mode === "connecting" && !reduce
              ? { duration: 18, ease: "linear", repeat: Infinity }
              : { duration: 0 }
          }
        />
        <div className="relative flex flex-col items-center gap-1 px-4 text-center">
          <span className="text-sm font-medium text-foreground">
            {mode === "connecting" ? "Connecting…" : "Waiting for the tray"}
          </span>
          <span className="max-w-[10rem] text-xs leading-snug text-muted-foreground">
            {mode === "connecting"
              ? "Reaching the camera"
              : "Point the camera at the instrument tray"}
          </span>
        </div>
      </div>
    )
  }

  return (
    <motion.div
      className={cn(
        "relative z-10 rounded-2xl bg-card px-5 py-2.5 text-center ring-1 ring-border",
        mode === "stalled" && "opacity-70"
      )}
      initial={false}
      animate={{ opacity: mode === "stalled" ? 0.7 : 1 }}
      transition={{ duration: reduce ? 0 : 0.22, ease: EASE }}
    >
      <div className="flex flex-col items-center gap-1">
        <div className="flex items-baseline gap-1.5">
          <span className="text-[1.9rem] font-semibold leading-none tabular-nums text-foreground">
            {count}
          </span>
          <span className="text-sm text-muted-foreground">
            instrument{count === 1 ? "" : "s"}
          </span>
        </div>
        <StateLine mode={mode} />
      </div>
    </motion.div>
  )
}

function StateLine({ mode }: { mode: Mode }) {
  if (mode === "stalled") {
    return (
      <span className="flex items-center gap-1.5 text-xs font-medium text-destructive">
        <Dot className="bg-destructive" />
        Stalled
      </span>
    )
  }
  if (mode === "ready") {
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-[color:var(--success)]">
        <CheckDot />
        Ready
      </span>
    )
  }
  // detecting
  return (
    <span className="flex items-center gap-1.5 text-xs font-medium text-[color:var(--primary)]">
      <Dot className="animate-pulse bg-[color:var(--primary)]" />
      Stabilizing
    </span>
  )
}

function Dot({ className }: { className?: string }) {
  return <span className={cn("size-1.5 rounded-full", className)} />
}

function CheckDot() {
  return (
    <svg viewBox="0 0 24 24" className="size-3" fill="none" aria-hidden="true">
      <path
        d="M20 6 9 17l-5-5"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

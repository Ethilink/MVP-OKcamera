import type { ReactElement, SVGProps } from "react"

/**
 * A small, consistent family of surgical-instrument line icons used as the
 * detection fallback during setup. The API normally supplies real detection
 * crops; these icons cover missing or failed thumbnails without changing the
 * layout.
 *
 * One drawing convention: 24×24 box, currentColor stroke, 1.5 width, round
 * joins — so the set reads as one clinical icon family, per BRANDING.md.
 */

type IconProps = SVGProps<SVGSVGElement>

const base: IconProps = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round",
  strokeLinejoin: "round",
}

export function ScissorsIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="6" cy="6" r="3" />
      <circle cx="6" cy="18" r="3" />
      <path d="M8.12 8.12 12 12" />
      <path d="M20 4 8.12 15.88" />
      <path d="M14.8 14.8 20 20" />
    </svg>
  )
}

export function ScalpelIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      {/* handle */}
      <path d="M3.5 20.5 12 12" />
      {/* blade */}
      <path d="M12 12 19.4 4.6a1.8 1.8 0 0 1 2.5 2.5l-6 6-2.5-2.5" />
      <path d="M13.4 13.4 15.9 15.9" />
    </svg>
  )
}

export function ForcepsIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      {/* two arms splayed at the top, meeting at a fine tip */}
      <path d="M8.5 3C10 9 11.4 14 12 21" />
      <path d="M15.5 3C14 9 12.6 14 12 21" />
      <path d="M9.3 6.2 14.7 6.2" opacity="0.5" />
    </svg>
  )
}

export function ClampIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      {/* finger rings */}
      <circle cx="7" cy="18.5" r="2.4" />
      <circle cx="13.4" cy="18.5" r="2.4" />
      {/* crossing shafts up to serrated jaws */}
      <path d="M8.7 16.8C11 12 12 8 16.5 3.5" />
      <path d="M11.7 16.8C10 12 11 8 8.5 3.6" />
    </svg>
  )
}

export function NeedleHolderIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      {/* suture needle arc + thread */}
      <path d="M4 20c3-1 6-5 8.5-9.5S17 3 20 3" />
      <path d="M20 3 17.6 3.2 M20 3 19.8 5.4" />
      <circle cx="5" cy="19.5" r="1" />
    </svg>
  )
}

export type InstrumentIcon = (props: IconProps) => ReactElement

/** Deterministic rotation used to give each detection tile a distinct glyph. */
export const INSTRUMENT_ICONS: InstrumentIcon[] = [
  ScissorsIcon,
  ScalpelIcon,
  ForcepsIcon,
  ClampIcon,
  NeedleHolderIcon,
]

export function instrumentIconFor(index: number): InstrumentIcon {
  return INSTRUMENT_ICONS[index % INSTRUMENT_ICONS.length]
}

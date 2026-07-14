import { cn } from "@/lib/utils"
import haloIcon from "@/assets/logo_icon.svg"

/**
 * The supplied Halo identity asset. Its source SVG has a transparent canvas,
 * so the mark sits directly on the clinical surface rather than in a white
 * badge.
 */
export function HaloMark({
  className,
  size = 32,
}: {
  className?: string
  size?: number
}) {
  return (
    <img
      src={haloIcon}
      alt="Halo"
      width={size}
      height={size}
      className={cn("shrink-0", className)}
    />
  )
}

/**
 * The "halo" wordmark — a custom script logotype drawn as a single path.
 * Rendered as inline SVG (not a bitmap) with fill="currentColor" so it inherits the
 * ink colour from the surrounding text context rather than being locked to a
 * fixed colour. Decorative here: the sibling HaloMark already carries the
 * accessible name, so this is aria-hidden to avoid a duplicate announcement.
 */
export function HaloWordmark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="15 64.9 87.9 34"
      fill="currentColor"
      aria-hidden="true"
      className={cn("w-auto", className)}
    >
      <path d="m101.9 81c-0.6-0.3-1.4 0-1.6 0.6-0.1 0.1-1.8 3.7-5.2 3.7-1 0-1.8-0.3-2.8-0.7-0.9-0.3-1.8-0.8-3.6-0.8-4.5 0-7.9 3.8-7.9 7.7v0.1c-1.3 2-3.6 4.2-6.3 4.2-2.2 0-3.8-1.3-4.9-4.3 2.3-3.2 7.2-10.8 8.4-17.7 0.5-2.7 0.6-5.5-0.5-7.1-0.7-1.2-1.8-1.8-3.2-1.8-5 0-7 7.1-8 12.1-0.6 3.3-1.2 8.8 0.2 14-1.1 1.4-3.9 4.8-6.7 4.8-0.7 0-1.2-0.2-1.6-0.7-0.8-1-0.6-3.5-0.6-5.7 0-1.1 0-2.3-0.7-3.4-0.8-1.3-2.3-2.3-4.1-2.3-3.4-0.1-6.2 2.9-6.2 6.6 0 1.7 0.5 3.3 1.4 4.8-1.8 0.6-2.9 0.9-4.2 1-1.4 0-2.8-0.4-3.4-1.2-0.8-0.9-0.7-2.3-0.3-3.9 1.2-4.4-0.3-6.4-1.8-7.1-1.8-0.9-4.1-0.4-6.5 1.5h-0.1c2.2-2.7 8.1-10.8 8.4-15.3 0.2-3.4-1.6-5.5-3.9-5.5-5.7 0.1-7.5 11.7-8.4 21.4-2.4 3-6.8 7-11.8 9.6-0.7 0.3-0.9 1.1-0.7 1.8 0.3 0.7 1.2 0.9 1.8 0.6 3.7-1.8 7.2-4.4 10.3-7.7-0.2 2.5-0.5 4.6-0.7 6.3-0.1 0.7 0.4 1.4 1.2 1.4 0.6 0 1.2-0.5 1.2-1.1 0.5-2.5 1.9-8.9 5.5-10.3 0.8-0.3 1.9-0.6 2.5 0s0.9 2 0.3 4c-0.4 2-0.6 4 0.8 5.8 1.1 1.3 2.8 2.2 5.6 2.2 1.9 0 4.1-0.6 5.7-1.3 0.8 0.6 1.8 1 3.2 1 1.2 0 2.4-0.4 3.6-1.4 1 1.1 2.2 1.4 3.6 1.4 2.8 0 5.5-2.1 7.8-4.4 1.7 3.5 4.4 4.4 6.5 4.4 2.7 0 5-1.3 6.7-3.4 1 2.1 3.1 3.4 6.1 3.4 4.3 0 7.9-3.6 8.4-7.4 0.1-1.3 0-2.3-0.4-3.3 4-0.2 6.1-3 7.2-5.1 0.4-0.6 0.2-1.4-0.3-1.8zm-65.6-13.5c0.7 0 1.1 0.7 1.1 2 0 2.6-3 7.9-6.4 12.6 0.9-6.7 2.8-14.5 5.2-14.6h0.1zm14 26.6c-0.6-1.1-0.9-2.4-0.9-3.8s1-4.2 3.2-4.2c1 0 1.9 0.4 2.2 1.2 1.1 2.3-2.2 5.4-4.5 6.7v0.1zm1.9 1.6c0.8-0.8 1.7-1.7 2.6-2.4 0 0.4 0.1 1 0.2 1.3-0.9 0.8-1.7 1.4-2.8 1.1zm16.4-7.4c-0.4-2.7-0.3-5.9 0.4-10.3 1-4.7 2.6-10.4 5-10.5 0.4 0 0.9 0.2 1.2 0.8 0.5 1 0.4 3 0 5-1 4.9-3.6 10.4-6.4 15h-0.1zm18.8 7.4c-2.3 0-4.2-1.4-4.2-4.2 0.3-2.9 2.2-5.2 5-5.3 2.7 0 4.7 1.7 4.5 4.4-0.2 2.4-2.1 5-5.3 5v0.1z" />
    </svg>
  )
}

/**
 * Lockup used top-left across screens: the ring mark + the "halo" wordmark.
 * The wordmark inherits the ink colour (text-foreground) so it tracks the theme.
 */
export function HaloBrand({ className }: { className?: string }) {
  return (
    <span className={cn("flex items-center gap-2.5 text-foreground", className)}>
      <HaloMark size={34} />
      <HaloWordmark className="h-[26px]" />
    </span>
  )
}

# ORC visual identity

## The short version

ORC should feel **calm, precise, and clinically clear**: a near-white working
surface, dark readable type, and one confident teal for interaction. The
rainbow ring is the identity signature, not a colour system to scatter across
the product.

> **Rule of thumb:** 90–95% white and neutral, 5–10% teal and semantic status
> colour, and the full rainbow only in the logo or an occasional brand moment.

This lets the logo bring warmth and movement while the live camera and report
remain easy to read in an operating-room demo.

## Locked frontend colour decision

The frontend is **not** a rainbow interface. Its visual language is:

> **White workspace · graphite text · deep-teal actions · semantic status colour.**

The full spectrum belongs to the logo. It may appear as a very soft halo in a
rare brand moment—such as onboarding or an empty state—but never in routine
buttons, navigation, badges, tables, detection boxes, or charts.

| Role | Colour | Frontend use |
| --- | --- | --- |
| App background | `#FFFFFF` | Main workspace |
| Soft panel | `#F4F7F8` | Toolbars and inactive areas |
| Ink | `#16242A` | Text, headings, default icons |
| Primary teal | `#006B70` | Primary action and selected state |
| Teal tint | `#E3F3F3` | Hover, selection, gentle information panels |
| Focus cyan | `#00A7B0` | Keyboard focus outline only |
| Success | `#167A59` | Present and complete |
| Warning | `#9A5D00` | Needs attention |
| Danger | `#B22F52` | Missing instrument and errors |

## Logo

The source logo assets live in [`app/frontend/src/assets/`](../frontend/src/assets/):

- `logo_icon.svg` — preferred app icon; scalable and crisp.
- `logo_name.svg` — preferred wordmark; scalable and crisp.
- `logo_name_transparent.png` — use only where a PNG with transparency is
  required.

Use the complete logo on a white or near-white surface. Give it clear space at
least equal to one quarter of the icon's width. Do not recolour the ring, crop
it, put it on a saturated panel, or turn it into a button background.

## Colour strategy

The deep teal below is deliberately taken from the logo's cyan-to-green area.
It provides a single stable colour for actions while still feeling native to a
rainbow identity. It is dark enough for white text; the bright cyan in the
logo is *not* suitable as a button fill.

### Foundation

| Role | Token | Hex | Use |
| --- | --- | --- | --- |
| Canvas | `--background` | `#FFFFFF` | Main page background |
| Soft surface | `--muted` | `#F4F7F8` | Toolbars, inactive areas, skeletons |
| Raised surface | `--card` | `#FFFFFF` | Panels and dialogs |
| Border | `--border` | `#D7E0E3` | Dividers and inputs |
| Primary ink | `--foreground` | `#16242A` | Headings and normal text |
| Secondary ink | `--muted-foreground` | `#52636A` | Supporting labels and descriptions |

### Interaction and state

| Role | Token | Hex | Use |
| --- | --- | --- | --- |
| ORC teal | `--primary` | `#006B70` | Primary actions, selected state, key links |
| Teal hover | `--primary-hover` | `#005B60` | Hover / pressed primary actions |
| Teal tint | `--primary-soft` | `#E3F3F3` | Selected rows, subtle information panels |
| Focus | `--ring` | `#00A7B0` | 3 px focus ring, never the main button fill |
| Success | `--success` | `#167A59` | Present / complete / healthy |
| Warning | `--warning` | `#9A5D00` | Needs attention or an in-progress caution |
| Danger | `--destructive` | `#B22F52` | Missing, failed, irreversible actions |
| Info | `--info` | `#1E5BB8` | Neutral guidance and system information |

The ink and semantic text colours are chosen to work on white. The bright
focus cyan is an outline only, never text. Do not rely on colour alone: pair
every status with its existing label, icon, or shape.

### Rainbow accent

The ring itself is the rainbow accent. If a supporting visual needs brand
colour—such as a welcome screen, empty state, or printed hand-out—use a very
light wash sampled from the logo (roughly 5–8% opacity) behind the logo only.
Never use a rainbow gradient for text, button fills, table rows, detection
boxes, charts, or status badges. In the product, colour must carry meaning.

## Components

### Buttons

| Button | Treatment | When to use |
| --- | --- | --- |
| Primary | `#006B70` fill, white label | The one clear next action: **Start recording**, **Save**, **Continue** |
| Secondary | White fill, `#D7E0E3` border, dark label | A useful alternative action |
| Quiet / ghost | Transparent, dark or teal label | Toolbar and low-emphasis actions |
| Stop / destructive | White fill, rose border and `#B22F52` label; pale rose hover | **Stop recording**, delete, or actions that need a deliberate pause |
| Disabled | `#E3EAEC` fill, `#66777D` label | Unavailable actions; never lower opacity alone |

Use solid teal for only the primary action on a given view. On the live screen,
**Start recording** is teal; **Stop recording** should be the deliberate
rose-outline treatment, rather than competing with the start action.

Buttons should use the existing modest 10 px radius, 44 px minimum touch
target for the important live controls, and a visible teal focus ring. Do not
make buttons rainbow, cyan, yellow, or pastel: those colours have insufficient
contrast and blur action hierarchy.

### Badges, alerts, and live status

- **Present / complete:** green tint with dark green text, plus the word
  “Present” or “Complete”.
- **Missing / error:** pale rose tint with dark rose text, plus the label.
- **Checking / waiting:** teal tint and a calm text label; avoid a festive
  multicolour loader.
- **Warning:** pale amber tint with the dark amber text token.
- **Neutral metadata:** soft surface with secondary ink.

The camera detection overlay should use the functional status colours above,
not the logo spectrum. Consistent colour meaning matters more than brand
decoration when an operator is checking instruments.

### Icons and illustrations

Use one simple, rounded-outline icon family. Icons are normally primary ink;
use teal only for selected/interactive controls and semantic colours only for
their matching state. Supporting illustrations can use black/teal linework
with a small logo-colour halo, but should not introduce extra rainbow UI
controls.

## Reports and charts

Reports are an evidence surface, so default to teal and neutrals rather than a
rainbow series. Use green, amber, and rose only for their semantic states.
If individual instruments need distinct series, assign a fixed, accessible set
and repeat it consistently; never map “good” and “bad” to arbitrary logo hues.

## Implementation tokens

The frontend already uses shadcn-style CSS variables in
[`app/frontend/src/index.css`](../frontend/src/index.css). When the palette is
implemented, these are the intended light-theme values:

```css
:root {
  --background: oklch(1 0 0);             /* #FFFFFF */
  --foreground: oklch(0.22 0.025 205);    /* #16242A */
  --card: oklch(1 0 0);                   /* #FFFFFF */
  --muted: oklch(0.97 0.008 205);         /* #F4F7F8 */
  --muted-foreground: oklch(0.47 0.025 205); /* #52636A */
  --border: oklch(0.89 0.015 205);        /* #D7E0E3 */
  --primary: oklch(0.45 0.08 198);        /* #006B70 */
  --primary-foreground: oklch(1 0 0);     /* #FFFFFF */
  --ring: oklch(0.65 0.13 200);           /* #00A7B0 */
  --destructive: oklch(0.50 0.16 10);     /* #B22F52 */
}
```

Keep `--secondary` and `--accent` neutral by default. The teal should be
intentional, not a decorative wash across every card.

## Accessibility checks

- Normal text needs at least 4.5:1 contrast; large text and important icons
  need at least 3:1.
- Test primary buttons with white text, plus keyboard focus and disabled state.
- Do not use pale logo colours for text on white.
- Never communicate presence, absence, or an error only by colour.

## Not our look

- A rainbow button set, rainbow navigation, or multicolour chart by default.
- Bright cyan with white type.
- Cream, beige, or heavily tinted page backgrounds.
- Large soft gradients behind routine UI; the camera feed and data should do
  the visual work.
- Excessively rounded, floating “glass” cards.

The intended result is a mostly white clinical workspace that feels quietly
branded the moment the ring appears—not a rainbow-themed dashboard.

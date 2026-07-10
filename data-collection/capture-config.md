# Capture Configuration — OK-Camera / CSA Instrument Tracking

Canonical camera settings for the CSA rig (phone-as-webcam via **Camo**, mounted
overhead above the weighing station, capturing surgical instrument trays).

**Rule: tune once, lock, and use the identical config for both data collection
and data testing.** Any drift between the two sets corrupts the evaluation.

---

## Locked settings (current)

| Parameter | Value | Notes |
|---|---|---|
| **Shutter speed** | **1/500 s** | Freezes hand-moved instruments (tracking mode). |
| **ISO** | **54** | Near base — low noise, clean edges for matching. |
| **White balance** | **MANUAL, 5600 K** | Daylight-neutral, fixed. Never auto. |
| **Tint** | +4 | As set in Camo. |
| **Focus** | **MANUAL, locked** | Lock to mid-working-height (a bit above the tray surface) so lifted instruments stay sharp. Never autofocus. |
| **Image adjustments** | **all OFF** | Brightness, Hue, Saturation, Vibrance, Contrast, Gamma, **Sharpness**, Flash — all toggles off. Rawest neutral image; do any sharpening in the training pipeline, not the sensor. |
| **Frame rate** | 60 fps (target) | Higher fps = easier frame-to-frame tracking. Confirm phone/Camo supports it at chosen resolution. |
| **Resolution** | highest that holds 60 fps | For tracking, fps > pixels if forced to choose. |

> ISO 54 + 1/500 s only expose correctly under the **specific lighting** these
> were tuned in. If the lighting changes, re-tune and log which config each
> recording used.

---

## Why these values (tracking, not static)

The subject is **moving** (instruments picked up, moved, placed) and made of
**shiny stainless steel**. That drives every choice:

- **Fast shutter (1/500 s)** — freezes motion so the tracker sees a crisp shape
  per frame, not a smear.
- **Low ISO (54)** — minimal noise keeps the fine edges the matcher relies on.
- **Fast shutter + low ISO both demand LIGHT** → lighting is the master lever
  (see below), not a camera setting.
- **Expose slightly under** — polished steel throws hot specular highlights; if
  they clip to pure white the edge is lost. Keep the histogram off the right wall.

### Flicker caveat (important)
A 1/500 s shutter under mains-powered (50 Hz, Belgium) hospital lighting can
produce rolling brightness bands. 1/500 is only safe with **flicker-free
(DC / high-frequency LED) lighting**. If you see banding, either switch to
flicker-free lights or drop to 1/100 s (and accept more blur / higher ISO).

---

## Lighting spec (the real lever)

Glare is fixed at the light source, not in Camo. Priority order:

1. **Bright + flicker-free LED** — enables 1/500 s at ISO 54 with no banding.
2. **Two-sided diffuse light (~45° from opposite sides)** — a single overhead
   source makes one hotspot down the tray; two diffused panels even it out.
3. **Cross-polarization** (polarizing filter on lens + polarizing film on lights,
   crossed 90°) — removes specular glare almost entirely. Highest-ROI upgrade
   for matching quality; cheap.

---

## Mount & perspective

Overhead capture means the **same instrument looks different by position** (edge
of frame = oblique/foreshortened) **and height** (lifted = larger, rotated).
That parallax/scale variation confuses matching and tracking. Mitigate:

- **Mount as high and centered as practical** — more top-down across the whole
  frame (less edge distortion) and deeper depth of field. Biggest mechanical win.
- **Bake the variation into training data on purpose** — capture each instrument
  at many positions across the frame and at a few heights, so the model learns
  pose/scale invariance instead of memorizing one centered view.
- **(Optional, later)** One-time camera calibration (printed checkerboard) gives
  intrinsics to un-warp perspective — only needed for measurement, not matching.

---

## Two modes (reference)

| Mode | Shutter | FPS | ISO | Use |
|---|---|---|---|---|
| **Tracking** (current) | 1/500 s | 60 | base (~54) | Instruments in motion. |
| **Static census** | 1/100 s | 30 | base | Frozen tray snapshot; trade fps for max resolution. |

---

*Rig: CSA weighing station + overhead phone via Camo. Keep this file in sync with
the physical setup; log any per-session deviations alongside the recordings.*

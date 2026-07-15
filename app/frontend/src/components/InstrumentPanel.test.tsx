import { render, screen, within } from "@testing-library/react"
import { expect, test } from "vitest"
import { recordingAllOn } from "@/test/fixtures"
import { InstrumentPanel } from "./InstrumentPanel"

// T10: every recording row carries a swatch of the instrument's mask colour, so
// an operator can pair a row with a shape on the video at a glance.
//
// The palette itself is a backend TUNABLE, so nothing here hard-codes a hex: the
// tests assert the swatch reflects WHATEVER colour the payload carried.

/** The <li> for an instrument, found by its label (rows are keyed by id). */
function rowFor(label: string): HTMLElement {
  const row = screen
    .getAllByRole("listitem")
    .find((li) => within(li).queryByText(label))
  if (!row) throw new Error(`no row for ${label}`)
  return row
}

/** The swatch's rendered colour, as the browser normalises it. */
function swatchColour(row: HTMLElement): string {
  return within(row).getByTestId("instrument-swatch").style.backgroundColor
}

test("each row's swatch is the colour /status gave that instrument", () => {
  const recording = recordingAllOn.recording!
  render(<InstrumentPanel recording={recording} />)

  for (const inst of recording.instruments) {
    expect(within(rowFor(inst.label)).getByTestId("instrument-swatch")).toHaveStyle({
      backgroundColor: inst.colour,
    })
  }
})

test("the swatch follows the payload, not the row's position", () => {
  // Colour is keyed off the instrument's id, so re-ordering the payload must
  // carry each colour with its own row: a panel colouring by row index would
  // hand instrument 5's hue to whatever landed first.
  const recording = recordingAllOn.recording!
  render(
    <InstrumentPanel
      recording={{ ...recording, instruments: [...recording.instruments].reverse() }}
    />
  )

  for (const inst of recording.instruments) {
    expect(within(rowFor(inst.label)).getByTestId("instrument-swatch")).toHaveStyle({
      backgroundColor: inst.colour,
    })
  }
})

test("distinct instruments get distinct swatches", () => {
  const recording = recordingAllOn.recording!
  render(<InstrumentPanel recording={recording} />)

  const colours = recording.instruments.map((inst) => swatchColour(rowFor(inst.label)))
  expect(new Set(colours).size).toBe(colours.length)
})

test("renders exactly the instruments /status returned — never invents a row", () => {
  // Unknown/foreign objects are video-only: the backend keeps them out of
  // `/status.recording.instruments` entirely, so there is no unknown row to
  // build. The panel is a pure projection of the payload — given a gappy id set
  // it must not fill the gaps in from a roster of its own.
  const recording = recordingAllOn.recording!
  const present = [1, 2, 5]
  render(
    <InstrumentPanel
      recording={{
        ...recording,
        instruments: recording.instruments.filter((i) =>
          present.includes(i.tracker_id)
        ),
      }}
    />
  )

  expect(screen.getAllByRole("listitem")).toHaveLength(present.length)
  expect(screen.getAllByTestId("instrument-swatch")).toHaveLength(present.length)
  expect(
    screen.getAllByRole("listitem").map((li) => within(li).getByText(/^Instrument \d$/).textContent)
  ).toEqual(["Instrument 1", "Instrument 2", "Instrument 5"])
  // The ids the payload omitted are absent, not blank-rowed.
  expect(screen.queryByText("Instrument 3")).toBeNull()
  expect(screen.queryByText("Instrument 4")).toBeNull()
})

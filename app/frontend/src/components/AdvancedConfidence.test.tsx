import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { http, HttpResponse } from "msw"
import { expect, test, vi } from "vitest"
import { BASE } from "@/api/client"
import type { DetectorControl } from "@/api/types"
import { server } from "@/test/server"
import { AdvancedConfidence } from "./AdvancedConfidence"

const control: DetectorControl = {
  confidence: 0.5,
  default_confidence: 0.5,
  minimum: 0.3,
  maximum: 0.9,
  step: 0.05,
}

/** Confidence PATCH handler that records the requested value. */
function patchHandler(spy: (v: number) => void, status = 200) {
  return http.patch(`${BASE}/settings/detection-confidence`, async ({ request }) => {
    const body = (await request.json()) as { confidence: number }
    spy(body.confidence)
    if (status !== 200) {
      return HttpResponse.json({ detail: "confidence change failed" }, { status })
    }
    return HttpResponse.json({ ...control, confidence: body.confidence })
  })
}

function relinkHandler(spy: () => void, status = 200) {
  return http.post(`${BASE}/settings/relink`, () => {
    spy()
    if (status !== 200) {
      return HttpResponse.json({ detail: "relink failed" }, { status })
    }
    return HttpResponse.json(control)
  })
}

function openPanel() {
  fireEvent.click(screen.getByRole("button", { name: /advanced/i }))
  // The base-ui slider thumb (and its range input) is `visibility:hidden` under
  // jsdom — no layout to position it — so include hidden elements.
  return screen.getByLabelText("Detection confidence threshold", {
    selector: 'input[type="range"]',
  })
}

test("F4 reads min/max/step/value from server metadata", () => {
  render(<AdvancedConfidence control={control} />)
  const slider = openPanel()
  expect(slider).toHaveAttribute("min", "0.3")
  expect(slider).toHaveAttribute("max", "0.9")
  expect(slider).toHaveAttribute("step", "0.05")
  // Displayed with two decimals.
  expect(screen.getByText("0.50")).toBeInTheDocument()
})

test("F4 debounces: rapid slides coalesce into one PATCH with the final value", async () => {
  const spy = vi.fn()
  server.use(patchHandler(spy))
  render(<AdvancedConfidence control={control} />)
  const slider = openPanel()

  // `input` = a live drag (onValueChange → debounced); `change` = a commit.
  fireEvent.input(slider, { target: { value: "0.55" } })
  fireEvent.input(slider, { target: { value: "0.6" } })
  // Not PATCHed synchronously — the value is debounced.
  expect(spy).not.toHaveBeenCalled()

  await waitFor(() => expect(spy).toHaveBeenCalled())
  expect(spy).toHaveBeenCalledTimes(1)
  expect(spy).toHaveBeenLastCalledWith(0.6)
})

test("F4 Reset to default is disabled at the default and PATCHes the default otherwise", async () => {
  const spy = vi.fn()
  server.use(patchHandler(spy))
  // Current value differs from default → Reset is available.
  render(<AdvancedConfidence control={{ ...control, confidence: 0.65 }} />)
  fireEvent.click(screen.getByRole("button", { name: /advanced/i }))

  const reset = screen.getByRole("button", { name: /reset to default/i })
  expect(reset).toBeEnabled()
  fireEvent.click(reset)
  await waitFor(() => expect(spy).toHaveBeenCalledWith(0.5))
})

test("F4 Reset to default is disabled when already at the default", () => {
  render(<AdvancedConfidence control={control} />)
  fireEvent.click(screen.getByRole("button", { name: /advanced/i }))
  expect(screen.getByRole("button", { name: /reset to default/i })).toBeDisabled()
})

test("F4/M1 a confirmed value survives a stale control prop (no snap-back)", async () => {
  const spy = vi.fn()
  server.use(patchHandler(spy))
  const { rerender } = render(<AdvancedConfidence control={control} />)
  const slider = openPanel()

  fireEvent.input(slider, { target: { value: "0.6" } })
  await waitFor(() => expect(spy).toHaveBeenCalledWith(0.6))
  await waitFor(() => expect(screen.getByText("0.60")).toBeInTheDocument())

  // The parent's poll is still stale at 0.50 — re-render with it. The slider must
  // NOT snap back to 0.50 (M1: the just-confirmed value wins).
  rerender(<AdvancedConfidence control={control} />)
  expect(screen.getByText("0.60")).toBeInTheDocument()
  expect(screen.queryByText("0.50")).toBeNull()
})

test("F4 a successful confidence change fires onReset (enrolment restarted)", async () => {
  const spy = vi.fn()
  server.use(patchHandler(spy))
  const onReset = vi.fn()
  render(<AdvancedConfidence control={control} onReset={onReset} />)
  const slider = openPanel()

  fireEvent.input(slider, { target: { value: "0.6" } })
  await waitFor(() => expect(onReset).toHaveBeenCalledTimes(1))
})

test("F4 Reset during a pending debounce restores the display without PATCHing", async () => {
  const spy = vi.fn()
  server.use(patchHandler(spy))
  render(<AdvancedConfidence control={control} />)
  const slider = openPanel()

  fireEvent.input(slider, { target: { value: "0.7" } }) // debounced, not committed
  expect(screen.getByText("0.70")).toBeInTheDocument()

  fireEvent.click(screen.getByRole("button", { name: /reset to default/i }))
  // Default equals the confirmed value → no PATCH, and the display returns to it
  // (rather than stranding at the un-submitted 0.70).
  expect(screen.getByText("0.50")).toBeInTheDocument()
  await new Promise((r) => setTimeout(r, 300))
  expect(spy).not.toHaveBeenCalled()
})

test("F4 a failed PATCH rolls back to the last confirmed value and shows an error", async () => {
  const spy = vi.fn()
  server.use(patchHandler(spy, 503))
  render(<AdvancedConfidence control={control} />)
  const slider = openPanel()

  fireEvent.change(slider, { target: { value: "0.7" } })
  await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument())
  // Rolled back to the server-confirmed 0.50, not the attempted 0.70.
  expect(screen.getByText("0.50")).toBeInTheDocument()
})

test("F4 reports its PATCH-in-flight up (to disable Track)", async () => {
  // A hanging PATCH keeps pending true.
  server.use(
    http.patch(`${BASE}/settings/detection-confidence`, () => new Promise(() => {})),
  )
  const onPending = vi.fn()
  render(<AdvancedConfidence control={control} onPendingChange={onPending} />)
  const slider = openPanel()

  fireEvent.change(slider, { target: { value: "0.6" } })
  await waitFor(() => expect(onPending).toHaveBeenCalledWith(true))
})

test("Advanced relink requests a fresh catalogue binding and refreshes setup", async () => {
  const spy = vi.fn()
  const onReset = vi.fn()
  server.use(relinkHandler(spy))
  render(<AdvancedConfidence control={control} onReset={onReset} />)
  fireEvent.click(screen.getByRole("button", { name: /advanced/i }))

  fireEvent.click(screen.getByRole("button", { name: /relink current masks/i }))

  await waitFor(() => expect(spy).toHaveBeenCalledTimes(1))
  expect(onReset).toHaveBeenCalledTimes(1)
})

test("Advanced relink reports a failed command without claiming setup reset", async () => {
  const spy = vi.fn()
  const onReset = vi.fn()
  server.use(relinkHandler(spy, 503))
  render(<AdvancedConfidence control={control} onReset={onReset} />)
  fireEvent.click(screen.getByRole("button", { name: /advanced/i }))

  fireEvent.click(screen.getByRole("button", { name: /relink current masks/i }))

  await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/relink failed/i))
  expect(onReset).not.toHaveBeenCalled()
})

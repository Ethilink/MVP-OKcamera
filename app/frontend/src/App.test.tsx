import { render, screen } from "@testing-library/react"
import { http, HttpResponse } from "msw"
import { expect, test } from "vitest"
import App from "./App"
import { server } from "./test/server"

test("renders the shell with a shadcn Button (AC7)", () => {
  render(<App />)
  expect(screen.getByRole("button", { name: /start/i })).toBeInTheDocument()
})

test("MSW intercepts a fetch through the test server (AC8)", async () => {
  server.use(
    http.get("http://localhost:8000/status", () =>
      HttpResponse.json({ phase: "setup" }),
    ),
  )
  const res = await fetch("http://localhost:8000/status")
  expect(res.status).toBe(200)
  expect(await res.json()).toEqual({ phase: "setup" })
})

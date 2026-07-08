import {
  finishedStatus,
  recordingAllOn,
  recordingOneOff,
  scriptedHandlers,
  setupStable,
  setupUnstable,
} from "./fixtures"

// Default pre-backend dev sequence: walks the whole phase machine so the dev
// page (AC6) visibly cycles setup → recording → finished, then holds. One step
// per ~2 Hz poll.
const DEV_SEQUENCE = [
  setupUnstable,
  setupUnstable,
  setupStable,
  setupStable,
  recordingAllOn,
  recordingOneOff,
  recordingOneOff,
  finishedStatus,
]

export const handlers = scriptedHandlers(DEV_SEQUENCE)

// ORC data-collection dashboard — T06 frontend logic.
// Vanilla JS, no build step, no external assets. Talks to the frozen T05
// endpoint contract (see T05-api.md). One rule everywhere: a server error is
// shown with its `detail`, never swallowed.

"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  stream: $("stream"),
  flag: $("flag"),
  confidence: $("confidence"),
  confidenceValue: $("confidence-value"),
  settings: $("settings"),
  error: $("error"),
  toast: $("toast"),
  health: $("health"),
  healthText: $("health-text"),
  count: $("count"),
  nFlagged: $("n-flagged"),
  datasetChip: $("dataset-chip"),
  cameraIndex: $("camera_index"),
  stageFrame: $("stage-frame"),
  savedChip: $("saved-chip"),
  savedText: $("saved-text"),
  discard: $("discard"),
  // Decorative/live extras — guarded everywhere so the page still works if a
  // future markup change drops one of them.
  hudCount: $("hud-count"),
  liveBadge: $("live-badge"),
  noSignal: $("no-signal"),
  captureFlash: $("capture-flash"),
  // Recording mode (TR6) — guarded the same way.
  flagLabel: $("flag-label"),
  record: $("record"),
  recordLabel: $("record-label"),
  recordControls: $("record-controls"),
  recordDiscard: $("record-discard"),
  recBadge: $("rec-badge"),
  recBadgeText: $("rec-badge-text"),
  keyframePill: $("keyframe-pill"),
  nKeyframes: $("n-keyframes"),
  keyframeChip: $("keyframe-chip"),
  // Unified capture UI (U3): mode toggle.
  modeToggle: $("mode-toggle"),
  modeImage: $("mode-image"),
  modeVideo: $("mode-video"),
  nameFieldLabel: $("name-field-label"),
  datasetName: $("dataset_name"),
};

let currentHealth = "unknown";
let toastTimer = null;

// --- capture mode + recording state ------------------------------------------
// Foreground state is idle | recording ONLY (ADR-0002: /record/stop writes the
// keyframe annotations synchronously and returns to idle — no background drain).
// `mode` is a pure client-side UI concern: image vs video. Toggling it never
// touches the backend (AC7).
let mode = "image"; // "image" | "video"
let recState = "idle"; // "idle" | "recording"
let recordBusy = false; // guards the Record/Stop button while a request is in flight
let nKeyframes = 0;

// --- live frame loop (client-driven display, AC-freeze) ---------------------
// We poll /frame instead of using an <img src="/stream"> MJPEG feed, because the
// browser must know the *generation* of the frame it is painting to capture that
// exact frame on SPACE. `after=<gen>` makes an unchanged frame a cheap 204.

const FRAME_POLL_MS = 66; // ~15 fps display cadence
let currentGen = -1; // generation currently painted (the id SPACE captures)
let currentBlobUrl = null; // object URL backing the <img>, revoked on swap
let frozen = false; // true while a just-captured frame is held for confirmation
let freezeTimer = null; // auto-resume timer for the confirmation hold
let frameTimer = null;

async function frameLoop() {
  if (!frozen) {
    try {
      const res = await fetch(`/frame?after=${currentGen}`);
      if (res.status === 200) {
        const gen = Number(res.headers.get("X-Frame-Generation"));
        const url = URL.createObjectURL(await res.blob());
        els.stream.src = url;
        if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
        currentBlobUrl = url;
        currentGen = gen;
      }
      // 204 (unchanged) / 503 (no frame yet): leave the current frame in place.
    } catch (_) {
      /* transient fetch error — the next tick retries */
    }
  }
  frameTimer = setTimeout(frameLoop, FRAME_POLL_MS);
}

// --- error / toast plumbing -------------------------------------------------

// FastAPI puts the message in `detail`: a string for HTTPException, an array of
// {loc, msg} for 422 validation errors. Render both readably.
function detailText(body, fallback) {
  const d = body && body.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    return d
      .map((e) => `${(e.loc || []).slice(1).join(".") || "input"}: ${e.msg}`)
      .join("; ");
  }
  return fallback;
}

function showError(msg) {
  els.error.textContent = msg;
  els.error.hidden = false;
}

function clearError() {
  els.error.hidden = true;
}

function showToast(msg) {
  els.toast.textContent = msg;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    els.toast.hidden = true;
  }, 1200);
}

// Parse a fetch Response; throw an Error carrying the server's detail on !ok.
async function readJson(res) {
  let body = null;
  try {
    body = await res.json();
  } catch (_) {
    /* empty / non-JSON body */
  }
  if (!res.ok) {
    throw new Error(detailText(body, `${res.status} ${res.statusText}`));
  }
  return body || {};
}

// --- confidence slider (debounced ~150 ms, AC4) -----------------------------

let confidenceTimer = null;

function renderConfidence(v) {
  const n = Number(v);
  els.confidenceValue.textContent = n.toFixed(2);
  // Fill the slider track up to the current value (0..1 -> 0..100%).
  els.confidence.style.setProperty("--fill", `${Math.round(n * 100)}%`);
}

els.confidence.addEventListener("input", () => {
  const value = Number(els.confidence.value);
  renderConfidence(value); // text tracks the drag instantly...
  clearTimeout(confidenceTimer); // ...but requests are debounced.
  confidenceTimer = setTimeout(async () => {
    try {
      const body = await readJson(
        await fetch("/confidence", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ value }),
        })
      );
      if (typeof body.confidence === "number") renderConfidence(body.confidence);
      clearError();
    } catch (e) {
      showError(`Confidence: ${e.message}`);
    }
  }, 150);
});

// --- flag (SPACE or button, AC1/AC2/AC3) ------------------------------------

function fireCaptureAnimation() {
  els.flag.classList.remove("flash");
  void els.flag.offsetWidth; // restart the CSS animation
  els.flag.classList.add("flash");
  if (els.captureFlash) {
    els.captureFlash.classList.remove("fire", "fire-keyframe");
    void els.captureFlash.offsetWidth; // restart the flash animation
    els.captureFlash.classList.add("fire");
  }
}

// Keyframe mark feedback (recording mode): a brief tinted pulse, no freeze —
// distinct from fireCaptureAnimation's still-capture flash (AC2, spec §What
// the operator experiences).
let keyframeChipTimer = null;
function fireKeyframeAnimation() {
  if (els.captureFlash) {
    els.captureFlash.classList.remove("fire", "fire-keyframe");
    void els.captureFlash.offsetWidth;
    els.captureFlash.classList.add("fire-keyframe");
  }
  if (els.keyframeChip) {
    clearTimeout(keyframeChipTimer);
    els.keyframeChip.hidden = false;
    keyframeChipTimer = setTimeout(() => {
      els.keyframeChip.hidden = true;
    }, 700);
  }
}

// Show the frozen still + "saved · discard" confirmation, then auto-resume live.
function enterFrozen(text) {
  frozen = true; // the frame loop stops swapping, so the exact frame stays put
  if (els.stageFrame) els.stageFrame.classList.add("frozen");
  if (els.savedChip && els.savedText) {
    els.savedText.textContent = text;
    els.savedChip.hidden = false;
  }
}

function resumeLive() {
  clearTimeout(freezeTimer);
  frozen = false;
  if (els.stageFrame) els.stageFrame.classList.remove("frozen");
  if (els.savedChip) els.savedChip.hidden = true;
}

async function flag() {
  if (currentHealth !== "ok") return; // never flag a frozen/dead stream
  if (frozen) return; // already showing a capture — ignore until it resumes
  if (currentGen < 0) return; // nothing painted yet
  const capturedGen = currentGen; // the EXACT frame on screen right now
  enterFrozen("Saving…");
  try {
    const body = await readJson(
      await fetch("/flag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ generation: capturedGen }),
      })
    );
    if (typeof body.n_flagged === "number") {
      els.nFlagged.textContent = body.n_flagged; // confirm without waiting for poll
    }
    fireCaptureAnimation();
    const anns = body.n_annotations;
    enterFrozen(anns != null ? `Saved · ${anns} instruments` : "Saved");
    clearError();
    freezeTimer = setTimeout(resumeLive, 1500); // hold the still, then resume
  } catch (e) {
    showError(`Flag: ${e.message}`);
    resumeLive(); // don't strand the feed frozen on an error
  }
}

// Mark the on-screen frame as a keyframe (recording mode). Instant, no freeze
// — sends the generation of the frame currently painted (the exact frame the
// operator saw); the backend resolves it to the MP4 frame_number via the ring
// and captures the live detection on it. A frame that aged out of the ring is a
// non-fatal 409 the operator can just re-mark.
async function markKeyframe() {
  if (recState !== "recording") return;
  if (currentGen < 0) return; // nothing painted yet
  const generation = currentGen;
  try {
    const body = await readJson(
      await fetch("/keyframe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ generation }),
      })
    );
    if (typeof body.n_keyframes === "number") {
      nKeyframes = body.n_keyframes;
    }
    fireKeyframeAnimation();
    renderRecordingUI();
    clearError();
  } catch (e) {
    // 409 not-recording / aged-out — surfaced, non-fatal, never swallowed.
    showError(`Keyframe: ${e.message}`);
  }
}

// SPACE and the FLAG button dispatch on MODE now (AC2 — "SPACE marks, always"):
//   image mode           -> flag a still
//   video + recording    -> mark a keyframe
//   video + idle         -> no-op with a hint ("Press Record first")
function flagOrKeyframe() {
  if (mode === "video") {
    if (recState === "recording") {
      markKeyframe();
    } else {
      showToast("Press Record first");
    }
    return;
  }
  flag();
}

els.flag.addEventListener("click", flagOrKeyframe);

// Discard the just-saved capture (undo). Available while the confirmation is up.
if (els.discard) {
  els.discard.addEventListener("click", async (e) => {
    e.stopPropagation();
    clearTimeout(freezeTimer);
    try {
      const body = await readJson(await fetch("/discard", { method: "POST" }));
      if (typeof body.n_flagged === "number") els.nFlagged.textContent = body.n_flagged;
      showToast("Discarded last capture");
      clearError();
    } catch (err) {
      showError(`Discard: ${err.message}`);
    }
    resumeLive();
  });
}

// Ignore auto-repeat (AC1) and typing in a form field (AC2).
function typingInField(target) {
  if (!target) return false;
  const tag = target.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    target.isContentEditable
  );
}

document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" && e.key !== " ") return;
  if (e.repeat) return; // hold-to-repeat must not machine-gun
  if (typingInField(e.target)) return; // SPACE while typing is a space, not a flag
  e.preventDefault(); // stop the page from scrolling
  flagOrKeyframe();
});

// --- settings (AC3 collision/invalid handling) ------------------------------

els.settings.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    output_path: $("output_path").value.trim(),
    dataset_name: $("dataset_name").value.trim(),
  };
  const cam = $("camera_index").value.trim();
  if (cam !== "") payload.camera_index = Number(cam);

  try {
    await readJson(
      await fetch("/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
    );
    showToast("Settings applied");
    clearError();
  } catch (err) {
    showError(`Settings: ${err.message}`);
  }
});

// --- capture mode + recording UI (U3) ---------------------------------------

// FLAG/KEYFRAME disables only when the live stream is unhealthy — there is no
// live frame to act on. (Video+idle stays enabled so the button can surface the
// "Press Record first" hint per AC2.)
function updateFlagDisabled() {
  if (!els.flag) return;
  els.flag.disabled = currentHealth !== "ok";
}

function renderRecordingUI() {
  const recording = recState === "recording";
  const video = mode === "video";

  // Mode toggle reflects the active mode and greys out while recording — you
  // can't change modes mid-take.
  const lockMode = recording;
  if (els.modeImage) {
    els.modeImage.classList.toggle("active", !video);
    els.modeImage.setAttribute("aria-selected", String(!video));
    els.modeImage.disabled = lockMode;
  }
  if (els.modeVideo) {
    els.modeVideo.classList.toggle("active", video);
    els.modeVideo.setAttribute("aria-selected", String(video));
    els.modeVideo.disabled = lockMode;
  }
  if (els.modeToggle) els.modeToggle.dataset.mode = mode;

  // Settings name field relabels per mode (AC4).
  if (els.nameFieldLabel) {
    els.nameFieldLabel.textContent = video ? "Recording session name" : "Dataset name";
  }

  if (els.stageFrame) els.stageFrame.dataset.recState = recState;

  // REC badge — only while actually recording now (no processing/failed states).
  if (els.recBadge) {
    els.recBadge.hidden = !recording;
    els.recBadge.dataset.state = recState;
  }
  if (els.recBadgeText) els.recBadgeText.textContent = recording ? "REC" : "";

  // Keyframe counter — visible only while recording.
  if (els.keyframePill) els.keyframePill.hidden = !recording;
  if (els.nKeyframes) els.nKeyframes.textContent = nKeyframes;

  // Record controls exist in VIDEO mode only (AC3).
  if (els.recordControls) els.recordControls.hidden = !video;

  // Record button toggles ● Record <-> ■ Stop (AC3).
  if (els.record) {
    els.record.disabled = recordBusy;
    els.record.dataset.state = recState;
  }
  if (els.recordLabel) els.recordLabel.textContent = recording ? "Stop" : "Record";

  // Discard appears while actively recording (abort the take).
  if (els.recordDiscard) els.recordDiscard.hidden = !recording || recordBusy;

  // FLAG button relabels FLAG <-> KEYFRAME by mode, keeping its position (AC2).
  if (els.flagLabel) els.flagLabel.textContent = video ? "KEYFRAME" : "FLAG";

  updateFlagDisabled();
}

// Switch capture mode — purely client-side. Never re-fires /settings (AC7): the
// image Dataset writer persists on the backend, so returning to image mode
// resumes the same Dataset without a collision-reject.
function setMode(next) {
  if (next !== "image" && next !== "video") return;
  if (next === mode) return;
  if (recState === "recording") return; // locked while a take is live (AC5)
  mode = next;
  clearError();
  renderRecordingUI();
}
if (els.modeImage) els.modeImage.addEventListener("click", () => setMode("image"));
if (els.modeVideo) els.modeVideo.addEventListener("click", () => setMode("video"));

// Record/Stop (video mode). idle -> /record/start with the Settings name field
// as entry_base (AC4); recording -> /record/stop (writes keyframe annotations
// synchronously, returns to idle — ADR-0002, no background drain).
if (els.record) {
  els.record.addEventListener("click", async () => {
    if (recState === "idle") {
      const entryBase = els.datasetName ? els.datasetName.value.trim() : "";
      if (!entryBase) {
        showError("Record: enter a recording session name first");
        return;
      }
      recordBusy = true;
      renderRecordingUI();
      try {
        await readJson(
          await fetch("/record/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ entry_base: entryBase }),
          })
        );
        recState = "recording";
        nKeyframes = 0;
        clearError();
      } catch (e) {
        showError(`Record: ${e.message}`); // 409 already recording, 422 bad base
      } finally {
        recordBusy = false;
        renderRecordingUI();
      }
    } else if (recState === "recording") {
      recordBusy = true;
      renderRecordingUI();
      try {
        const stop = await readJson(await fetch("/record/stop", { method: "POST" }));
        recState = "idle"; // annotations written synchronously by /record/stop
        if (stop && stop.error) {
          // The encoder died mid-take: the clip was still saved (whatever frames
          // + keyframes landed before the failure) but it is TRUNCATED — don't
          // let it look like a clean recording.
          showError(`Recording incomplete — encoder failed mid-take (${stop.error}). The saved clip is truncated; re-record.`);
        } else {
          clearError();
        }
      } catch (e) {
        showError(`Record: ${e.message}`);
      } finally {
        recordBusy = false;
        renderRecordingUI();
      }
    }
  });
}

// Discard aborts the live recording and deletes its half-written Entry.
async function discardRecording(label) {
  try {
    await readJson(await fetch("/record/discard", { method: "POST" }));
    if (recState === "recording") recState = "idle";
    nKeyframes = 0;
    showToast(label || "Recording discarded");
    clearError();
  } catch (e) {
    showError(`Discard: ${e.message}`);
  } finally {
    renderRecordingUI();
  }
}
if (els.recordDiscard) {
  els.recordDiscard.addEventListener("click", () => discardRecording("Recording discarded"));
}

// Polled alongside /status (same 1 s cadence). Shape: {state: "idle" |
// "recording", error: string | null} (ADR-0002 dropped the drain block; error
// carries a mid-take encoder failure). Keeps the last known state on a transient
// failure.
async function pollRecordStatus() {
  try {
    const s = await readJson(await fetch("/record/status"));
    if (typeof s.state === "string") recState = s.state;
    // A live encoder failure while recording: surface it now so the operator can
    // Discard and restart instead of recording into a dead encoder until Stop.
    if (s.error) {
      showError(`Recording failing — encoder error (${s.error}). Discard and restart.`);
    }
  } catch (_) {
    /* transient — keep the last known state */
  } finally {
    renderRecordingUI();
  }
}

// --- status poll (1 s, AC5 health) ------------------------------------------

function applyHealth(health) {
  currentHealth = health;
  const healthy = health === "ok";
  els.health.dataset.health = health;
  els.healthText.textContent = healthy ? "live" : health; // "stale" / "dead"
  updateFlagDisabled(); // FLAG/KEYFRAME disables while the stream is unhealthy
  if (els.liveBadge) els.liveBadge.dataset.health = health;
  // Veil the video with a "no signal" message the moment the feed freezes.
  if (els.noSignal) els.noSignal.hidden = healthy || health === "unknown";
}

async function poll() {
  try {
    const s = await readJson(await fetch("/status"));
    els.count.textContent = s.count ?? "–";
    if (els.hudCount) els.hudCount.textContent = s.count ?? 0;
    els.nFlagged.textContent = s.n_flagged ?? 0;
    els.datasetChip.textContent = s.dataset_name || "no dataset";
    els.datasetChip.classList.toggle("active", !!s.dataset_name);
    if (typeof s.confidence === "number" && document.activeElement !== els.confidence) {
      els.confidence.value = s.confidence;
      renderConfidence(s.confidence);
    }
    // Show the camera the loop is actually streaming from (set at startup via
    // --camera-index, or after a /settings swap). Don't clobber the field while
    // the operator is editing it to enter a new index.
    if (
      els.cameraIndex &&
      typeof s.camera_index === "number" &&
      document.activeElement !== els.cameraIndex
    ) {
      els.cameraIndex.value = s.camera_index;
    }
    applyHealth(s.capture_health || "unknown");
    // Early/cheap signal for the top-level state (also refined by
    // pollRecordStatus's richer /record/status body, same cadence below).
    if (typeof s.recording_state === "string") recState = s.recording_state;
  } catch (_) {
    applyHealth("dead"); // status unreachable → treat as dead, block flagging
  }
  await pollRecordStatus();
}

renderConfidence(els.confidence.value);
renderRecordingUI();
poll();
setInterval(poll, 1000);
frameLoop();

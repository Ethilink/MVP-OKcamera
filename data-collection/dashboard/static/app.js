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
  validate: $("validate"),
  validateResults: $("validate-results"),
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
  recordDiscard: $("record-discard"),
  recBadge: $("rec-badge"),
  recBadgeText: $("rec-badge-text"),
  keyframePill: $("keyframe-pill"),
  nKeyframes: $("n-keyframes"),
  keyframeChip: $("keyframe-chip"),
  postpassVeil: $("postpass-veil"),
  postpassTitle: $("postpass-title"),
  progressFill: $("progress-fill"),
  postpassSub: $("postpass-sub"),
  postpassError: $("postpass-error"),
  postpassRetry: $("postpass-retry"),
  postpassDiscard: $("postpass-discard"),
};

let currentHealth = "unknown";
let toastTimer = null;

// --- recording state (TR6) ---------------------------------------------------
// idle | recording | processing | failed — mirrors the TR5 state machine.
let recState = "idle";
let recordBusy = false; // guards the Record/Stop button while a request is in flight
let nKeyframes = 0;
let postpassDone = 0;
let postpassTotal = 0;
let postError = null;

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
// The on-screen frame's number, from X-Frame-Number (present while recording
// only). SPACE-while-recording echoes THIS value to /keyframe — never the
// newest frame — because that is the frame the operator was looking at.
let currentFrameNumber = -1;

async function frameLoop() {
  if (!frozen) {
    try {
      const res = await fetch(`/frame?after=${currentGen}`);
      if (res.status === 200) {
        const gen = Number(res.headers.get("X-Frame-Generation"));
        const fn = res.headers.get("X-Frame-Number");
        currentFrameNumber = fn !== null ? Number(fn) : -1;
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
// — sends the X-Frame-Number captured from the latest /frame response, never
// the newest frame (AC2/AC3 of RECORDING.md — the frame the operator saw).
async function markKeyframe() {
  if (recState !== "recording") return;
  if (currentFrameNumber < 0) return; // nothing painted yet
  const frameNumber = currentFrameNumber;
  try {
    const body = await readJson(
      await fetch("/keyframe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame_number: frameNumber }),
      })
    );
    if (typeof body.n_keyframes === "number") {
      nKeyframes = body.n_keyframes;
    }
    fireKeyframeAnimation();
    renderRecordingUI();
    clearError();
  } catch (e) {
    showError(`Keyframe: ${e.message}`); // 409 (not recording) / 422 (out of range) — never swallowed
  }
}

// SPACE and the FLAG button branch on recording_state (TR6 UI contract):
// idle -> still-capture flag (unchanged), recording -> keyframe mark.
// processing/failed: no-op — there is no live frame to act on.
function flagOrKeyframe() {
  if (recState === "recording") {
    markKeyframe();
  } else if (recState === "idle") {
    flag();
  }
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

// --- validate (AC6) ---------------------------------------------------------

function renderList(title, items, cls) {
  if (!items || items.length === 0) return "";
  const lis = items.map((i) => `<li>${escapeHtml(String(i))}</li>`).join("");
  return `<h3>${title}</h3><ul class="${cls}">${lis}</ul>`;
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

els.validate.addEventListener("click", async () => {
  els.validateResults.hidden = true;
  try {
    const body = await readJson(await fetch("/validate", { method: "POST" }));
    const errors = body.errors || [];
    const warnings = body.warnings || [];
    let html = "";
    if (errors.length === 0 && warnings.length === 0) {
      html = '<p class="clean">✓ import-ready — no errors or warnings</p>';
    } else {
      html =
        renderList("Errors", errors, "errors") +
        renderList("Warnings", warnings, "warnings");
    }
    els.validateResults.innerHTML = html;
    els.validateResults.hidden = false;
    clearError();
  } catch (e) {
    showError(`Validate: ${e.message}`);
  }
});

// --- recording mode (TR6): Record/Stop, progress, retry/discard -------------

// FLAG disables when the stream is unhealthy (unchanged) OR while a post-pass
// owns the detector (processing/failed — there is nothing live to act on).
function updateFlagDisabled() {
  if (!els.flag) return;
  els.flag.disabled =
    currentHealth !== "ok" || recState === "processing" || recState === "failed";
}

function renderRecordingUI() {
  const recording = recState === "recording";
  const processing = recState === "processing";
  const failed = recState === "failed";

  if (els.stageFrame) els.stageFrame.dataset.recState = recState;

  if (els.recBadge) {
    els.recBadge.hidden = !(recording || processing || failed);
    els.recBadge.dataset.state = recState;
  }
  if (els.recBadgeText) {
    els.recBadgeText.textContent = recording
      ? "REC"
      : processing
      ? "PROCESSING"
      : failed
      ? "FAILED"
      : "";
  }

  // Keyframe counter — visible only while recording (spec: "Visible while recording").
  if (els.keyframePill) els.keyframePill.hidden = !recording;
  if (els.nKeyframes) els.nKeyframes.textContent = nKeyframes;

  // Record/Stop button.
  if (els.record) {
    els.record.disabled = processing || failed || recordBusy;
    els.record.dataset.state = recState;
  }
  if (els.recordLabel) {
    els.recordLabel.textContent = recording
      ? "Stop"
      : processing
      ? "Processing…"
      : failed
      ? "Failed"
      : "Record";
  }

  // Discard while actively recording (abort) — lives next to Record/Stop.
  if (els.recordDiscard) els.recordDiscard.hidden = !recording || recordBusy;

  // Processing/failed veil over the stage, with the progress bar + error/retry.
  if (els.postpassVeil) {
    els.postpassVeil.hidden = !(processing || failed);
    els.postpassVeil.dataset.state = recState;
  }
  if (els.postpassTitle) {
    els.postpassTitle.textContent = failed ? "Post-pass failed" : "Processing…";
  }
  const pct = postpassTotal > 0 ? Math.round((postpassDone / postpassTotal) * 100) : 0;
  if (els.progressFill) els.progressFill.style.width = `${pct}%`;
  if (els.postpassSub) {
    els.postpassSub.textContent = `${postpassDone} / ${postpassTotal} frames`;
  }
  if (els.postpassError) {
    els.postpassError.hidden = !failed;
    els.postpassError.textContent = postError || "Unknown error";
  }
  if (els.postpassRetry) els.postpassRetry.hidden = !failed;
  // Discard is available from recording/processing/failed (AC7); the veil-side
  // control covers processing/failed, els.recordDiscard covers recording.
  if (els.postpassDiscard) els.postpassDiscard.hidden = !(processing || failed);

  if (els.flagLabel) els.flagLabel.textContent = recording ? "KEYFRAME" : "FLAG";
  updateFlagDisabled();
}

// Record/Stop: idle -> prompt for entry_name -> /record/start; recording -> /record/stop.
if (els.record) {
  els.record.addEventListener("click", async () => {
    if (recState === "idle") {
      const name = window.prompt("Entry name:");
      if (name === null) return; // cancelled
      const entryName = name.trim();
      if (!entryName) return;
      recordBusy = true;
      renderRecordingUI();
      try {
        await readJson(
          await fetch("/record/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ entry_name: entryName }),
          })
        );
        recState = "recording";
        nKeyframes = 0;
        clearError();
      } catch (e) {
        showError(`Record: ${e.message}`); // 409 collision/wrong-state, 422 bad name
      } finally {
        recordBusy = false;
        renderRecordingUI();
      }
    } else if (recState === "recording") {
      recordBusy = true;
      renderRecordingUI();
      try {
        await readJson(await fetch("/record/stop", { method: "POST" }));
        recState = "processing";
        clearError();
      } catch (e) {
        showError(`Record: ${e.message}`);
      } finally {
        recordBusy = false;
        renderRecordingUI();
      }
    }
  });
}

async function discardRecording() {
  try {
    await readJson(await fetch("/record/discard", { method: "POST" }));
    recState = "idle";
    nKeyframes = 0;
    postpassDone = 0;
    postpassTotal = 0;
    postError = null;
    showToast("Recording discarded");
    clearError();
  } catch (e) {
    showError(`Discard recording: ${e.message}`);
  } finally {
    renderRecordingUI();
  }
}
if (els.recordDiscard) els.recordDiscard.addEventListener("click", discardRecording);
if (els.postpassDiscard) els.postpassDiscard.addEventListener("click", discardRecording);

if (els.postpassRetry) {
  els.postpassRetry.addEventListener("click", async () => {
    try {
      await readJson(await fetch("/record/retry", { method: "POST" }));
      recState = "processing";
      postError = null;
      clearError();
    } catch (e) {
      showError(`Retry: ${e.message}`);
    } finally {
      renderRecordingUI();
    }
  });
}

// Polled alongside /status (same 1 s cadence, see poll() below). Silently
// keeps the last known recording state on failure — this 404s until TR5's
// backend ships, same as any other not-yet-implemented endpoint.
async function pollRecordStatus() {
  try {
    const s = await readJson(await fetch("/record/status"));
    if (typeof s.state === "string") recState = s.state;
    if (typeof s.n_keyframes === "number") nKeyframes = s.n_keyframes;
    if (s.postpass) {
      if (typeof s.postpass.done === "number") postpassDone = s.postpass.done;
      if (typeof s.postpass.total === "number") postpassTotal = s.postpass.total;
    }
    postError = s.error || null;
  } catch (_) {
    /* TR5 backend not deployed yet, or transient — keep the last known state */
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
  updateFlagDisabled(); // FLAG disables while unhealthy or mid-postpass
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

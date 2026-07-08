// ORB data-collection dashboard — T06 frontend logic.
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
};

let currentHealth = "unknown";
let toastTimer = null;

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
    els.captureFlash.classList.remove("fire");
    void els.captureFlash.offsetWidth; // restart the flash animation
    els.captureFlash.classList.add("fire");
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

els.flag.addEventListener("click", flag);

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
  flag();
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

// --- status poll (1 s, AC5 health) ------------------------------------------

function applyHealth(health) {
  currentHealth = health;
  const healthy = health === "ok";
  els.health.dataset.health = health;
  els.healthText.textContent = healthy ? "live" : health; // "stale" / "dead"
  els.flag.disabled = !healthy; // FLAG disables while unhealthy
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
  } catch (_) {
    applyHealth("dead"); // status unreachable → treat as dead, block flagging
  }
}

renderConfidence(els.confidence.value);
poll();
setInterval(poll, 1000);
frameLoop();

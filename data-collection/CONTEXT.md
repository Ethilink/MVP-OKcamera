# Data-collection dashboard

The capture tool: stream the camera, run the detector live, and let the operator
save training material — either individual still frames or full recordings that a
later post-pass detects over. Feeds the separate annotation dashboard; never
edits anything itself.

## Language

**Output path**:
The operator-configured drop-zone directory under which every capture target is
created. A staging area only — finished folders are later copied by hand into the
annotation tool. Not itself a project.
_Avoid_: dataset root, save dir.

**Dataset**:
An image capture target — one folder that accumulates many still frames across a
session (1 folder : N stills). Persistent name, set once in Settings.
_Avoid_: using "dataset" for a recording.

**Entry**:
A video capture target — one folder holding exactly one recording plus its
derived keyframes (1 folder : 1 recording). Multiple recordings ⇒ multiple
entries.
_Avoid_: dataset (for video), clip, project.

**Take**:
One act of recording, start to stop — produces exactly one Entry.

**Post-pass**:
The offline job that reopens a finished recording and runs the detector over
every frame, producing the all-frames sidecar and the keyframe annotations. The
authoritative detections (live overlay is UX only).

**Base name**:
The single name in Settings. In image mode it *is* the Dataset name; in video
mode it seeds Entry names, auto-suffixed per take (`<base>_001`, `<base>_002`…).

## Relationships

- An **Output path** contains many **Datasets** and **Entries** as sibling folders.
- A **Dataset** and an **Entry** are always **disjoint** — never the same folder.
  A non-empty `video/` forces the whole folder to open as a video project,
  orphaning any stills; the two `annotations.json` schemas are incompatible.
- A **Base name** maps to at most one **Dataset** (`images/<base>/`) and any number
  of **Entries** (`videos/<base>_NNN/`) — the same name in the operator's head,
  split cleanly on disk.
- A **Take** produces one **Entry**; its **Post-pass** is enqueued and drains when
  the tool is idle.

## Example dialogue

> **Bram:** "One tray, I grab a few stills and record a couple of clips — same
> folder, right?"
> **Contract:** "Same **base name**, not the same folder. Stills accumulate in
> `images/<base>/` (a **Dataset**); each recording is its own `videos/<base>_NNN/`
> (an **Entry**). If they shared a folder the annotation tool would see `video/`,
> open it as a video project, and your stills would vanish from review."

## Flagged ambiguities

- "Dataset name" was being used for both image and video targets — resolved: the
  Settings field is a **Base name** that means the **Dataset** name in image mode
  and an **Entry** base (auto-suffixed) in video mode; the UI relabels it per mode.

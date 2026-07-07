# Import format — pre-annotated image datasets

How an **external pipeline** (outside this repo) must lay out an already-annotated
image dataset so the annotation dashboard can open it, let a human correct the
annotations, and re-export it for training.

This is the "boxes **and** masks, independent images, single class" case. The
dashboard has no importer for pre-annotated folders — it simply **discovers** any
folder that already contains a valid `annotations.json`. So the contract is: emit
the file in the shape below, drop the folder under `data/processed/`, done.

> Validate before you drop it in: `python annotation_tool/scripts/validate_import.py <folder>`

---

## 1. Folder layout

```
data/processed/<dataset_name>/
├── images/
│   ├── frame_0001.jpg          # flat directory — every image the JSON references
│   ├── frame_0002.jpg
│   └── ...
└── annotations/
    └── annotations.json        # the entire dataset (COCO-VID), one file
```

- `<dataset_name>` — a single path component (no `/`, no leading `.`). Becomes the
  project name in the browser.
- `images/` must be **flat** — no subfolders. Filenames must be unique.
- Only `annotations/annotations.json` is required beyond the images. Everything
  else (`debug/`, `annotations/metadata/`, backups) is created by the dashboard.

**Discovery rule** (`src/models/project.py` → `Project.from_directory`): a folder
opens as an *image project* when `annotations/annotations.json` **and** an
`images/` dir both exist. Missing JSON → shows as a raw folder "needs detection";
JSON without `images/` (or `video/`) → ignored.

---

## 2. `annotations.json` — COCO-VID schema

```jsonc
{
  "info": {
    "description": "My external pipeline v3 — tray 0421",
    "date_created": "2026-07-07T10:00:00"
    // free-form; any extra keys are preserved round-trip
  },

  "categories": [
    { "id": 1, "name": "surgical_instrument", "supercategory": "" }
  ],

  "images": [
    {
      "id": 1,                       // unique int, referenced by annotations[].image_id
      "file_name": "frame_0001.jpg", // basename inside images/ (no path)
      "width": 1920,                 // pixels, must match the actual JPEG
      "height": 1080,
      "review_status": "pending"     // see §4 — start machine output as "pending"
    }
  ],

  "annotations": [
    {
      "id": 1,                       // unique int across the whole file
      "image_id": 1,                 // FK → images[].id
      "category_id": 1,              // FK → categories[].id
      "track_id": 1,                 // unique per instance (see §3)
      "bbox": [x, y, w, h],          // absolute pixels, top-left origin — REQUIRED
      "area": 17527,                 // int; mask pixel-area (or bbox area if box-only)
      "iscrowd": 0,
      "segmentation": {              // COCO RLE — the mask (see §3)
        "size": [1080, 1920],        //   [height, width] — must match the image
        "counts": "TemX1<YQ17J4K..." //   compressed RLE string
      }
    }
  ]
}
```

### Field reference

| Field | Required | Notes |
|---|---|---|
| `info` | dict | Free-form. Unknown keys survive load→save. Use it for provenance. |
| `categories` | yes | Single entry `{id:1, name:"surgical_instrument"}` for this case. |
| `images[].id` | yes | Unique int. |
| `images[].file_name` | yes | Basename only; the file must exist in `images/`. |
| `images[].width` / `height` | yes | Must equal the real JPEG dimensions (see §5, EXIF). |
| `images[].review_status` | no (default `"pending"`) | See §4. |
| `annotations[].id` | yes | Unique across the file. |
| `annotations[].image_id` | yes | Must match an `images[].id`. |
| `annotations[].category_id` | yes | Must match a `categories[].id`. |
| `annotations[].track_id` | recommended | Unique per instance. Omit → editor assigns on edit. |
| `annotations[].bbox` | **yes** | `[x, y, width, height]`, absolute pixels. |
| `annotations[].segmentation` | for this case, yes | COCO RLE dict, see §3. |
| `annotations[].area` | recommended | Int. Recomputed on edit; fine to approximate. |
| `annotations[].iscrowd` | recommended | `0`. |

> The top-level `video` block used by the *video* pipeline is **not needed** for an
> independent-image dataset. `video_id` and `frame_number` on images default to
> `None` and can be omitted.

Any extra top-level keys on an image or annotation record (your own confidence
scores, source IDs, model version, etc.) are **preserved** across a load→edit→save
cycle — the loader stashes unknown keys in an `extras` bucket. Stash whatever
provenance you want.

---

## 3. Masks — COCO RLE encoding

The dashboard stores masks as **COCO RLE** (`{ "size": [h, w], "counts": "..." }`).
Encode your binary masks with `pycocotools`:

```python
import numpy as np
from pycocotools import mask as mask_utils

# mask: HxW uint8/bool array for one instance, in Fortran order
rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
rle["counts"] = rle["counts"].decode("utf-8")   # bytes -> str for JSON
area = int(mask_utils.area(rle))
x, y, w, h = [float(v) for v in mask_utils.toBbox(rle)]

annotation["segmentation"] = {"size": rle["size"], "counts": rle["counts"]}
annotation["area"] = area
annotation["bbox"] = [x, y, w, h]
```

- `size` **must** equal `[image_height, image_width]` of the referenced image.
- Polygon segmentation (`[[x1,y1,x2,y2,...]]`) is also accepted — the loader
  converts it to RLE via pycocotools. RLE is preferred (exact, no re-conversion).
- **Box-only fallback:** if a given instance has no mask, omit `segmentation`
  entirely. The dashboard's *Detect & Segment* can fill masks with SAM3 later.

---

## 4. `review_status` — the correction loop

This is the field that makes your workflow work:

- Set every machine-produced image to **`"pending"`** (the default). It then shows
  as *unreviewed* / *needs review* in the browser and is **excluded** from the
  send-to-training export until a human confirms it.
- The dashboard flips an image to `"reviewed"` when the human accepts it.
- `send.py` (export to `data/sent/`) can drop non-`reviewed` rows — so wrong
  machine annotations never reach training unless a human passed over them.

Do **not** pre-mark machine output as `"reviewed"` — that would skip the whole
point of the review pass.

---

## 5. Gotchas

- **EXIF rotation.** The dashboard's own importer bakes EXIF orientation into
  pixels. An external pipeline does not go through that path, so: **strip EXIF
  rotation and write upright pixels**, and make `width`/`height` match those
  upright pixels. A portrait phone JPEG with an EXIF-rotate tag will otherwise
  render rotated in the canvas while masks sit on the un-rotated pixels.
- **ID consistency.** Every `image_id` must resolve to an `images[].id`; every
  `category_id` to a `categories[].id`. Duplicate `annotations[].id` or
  `images[].id` will corrupt editing.
- **`size` mismatch.** RLE `size` that disagrees with the image dimensions
  produces masks that don't align. The validator checks this against the JPEG.
- **Filenames.** `file_name` is a basename, not a path. Files live flat in
  `images/`. Unique names only.

---

## 6. Minimal example

`data/processed/demo_tray/annotations/annotations.json`:

```json
{
  "info": { "description": "demo", "date_created": "2026-07-07T10:00:00" },
  "categories": [{ "id": 1, "name": "surgical_instrument", "supercategory": "" }],
  "images": [
    { "id": 1, "file_name": "frame_0001.jpg", "width": 1920, "height": 1080, "review_status": "pending" }
  ],
  "annotations": [
    {
      "id": 1, "image_id": 1, "category_id": 1, "track_id": 1,
      "bbox": [1241.0, 500.0, 332.0, 359.0], "area": 17527, "iscrowd": 0,
      "segmentation": { "size": [1080, 1920], "counts": "TemX1<YQ17J4K..." }
    }
  ]
}
```

with `data/processed/demo_tray/images/frame_0001.jpg` present.

# PaddleOCR Labeler Service Design

**Date:** 2026-07-31
**Status:** Approved in conversation; awaiting review of this written specification

## 1. Objective

Build a lightweight local browser service that:

- opens a local directory of document images;
- runs PP-OCRv6 text detection and the fine-tuned Vietnamese recognition model;
- returns recognized text, confidence, reading order, and four-point bounding
  polygons;
- displays the page and editable regions in a three-column annotation UI;
- allows full text and geometry correction;
- autosaves one JSON sidecar per image; and
- exports all annotations as a portable JSONL manifest.

The service is an inference and correction tool. It does not modify the source
images and does not change the fine-tuning pipeline.

## 2. Scope

### Included

- Local directory selection and direct-child image discovery.
- PNG, JPEG, WebP, BMP, and TIFF input.
- OCR for the current image.
- Sequential batch OCR for all eligible images in the selected directory.
- Batch progress, cancellation, and safe continuation.
- Text editing.
- Adding, selecting, moving, resizing, and deleting four-point polygons.
- Manual reading-order changes.
- Undo and redo for editor mutations.
- Atomic per-image autosave.
- Revision conflict detection.
- JSONL export.
- GPU inference on `gpu:0` by default, with explicit CPU mode available.

### Excluded

- User accounts, authentication, and multi-user collaboration.
- Remote image URLs or cloud storage.
- Recursive directory scanning.
- Automatic document-region classification.
- Table structure reconstruction.
- Rich-text/HTML editing.
- Fine-tuning from the browser.
- Concurrent GPU inference.

Every block has one implicit type: `Text`.

## 3. Architecture

The implementation lives in a new top-level `ocr_labeler/` package and remains
separate from `finetune.py`.

```text
Browser UI
   |
   | JSON API + image responses
   v
FastAPI application
   |-- directory/image catalog
   |-- annotation store
   |-- batch job manager
   `-- singleton OCR engine
          |-- PP-OCRv6_medium_det
          `-- fine-tuned PP-OCRv6_medium_rec
```

The FastAPI process serves both the API and static HTML/CSS/JavaScript. The
frontend has no build step and no runtime Node dependency.

The OCR engine is initialized once at application startup. A single worker
queue serializes all GPU calls so current-image OCR and batch OCR cannot race or
duplicate model memory.

## 4. OCR Pipeline

### Detection

Use `PP-OCRv6_medium_det`. The launcher accepts `--det-model-dir`. If no
directory is passed, it uses the official model name and PaddleOCR's local
model cache.

### Recognition

The default recognition directory is:

```text
runs/vi_rec_3datasets_v1/inference/best_accuracy
```

The launcher accepts `--rec-model-dir` to override it.

### Runtime configuration

The OCR pipeline is created with:

- device: `gpu:0`;
- recognition input shape: `3,48,1600`;
- document orientation classification: disabled;
- document unwarping: disabled;
- text-line orientation classification: disabled;
- recognition score threshold: `0.0`, preserving low-confidence candidates for
  correction;
- low-confidence visual warning threshold: `0.60`.

The service does not silently fall back to CPU. `--device cpu` is an explicit
operator choice.

### Result normalization

Each PaddleOCR result is normalized into:

- recognized UTF-8 text;
- four absolute pixel coordinates in clockwise order;
- recognition confidence;
- stable block ID;
- reading-order index;
- source, either `ocr` or `manual`.

The service preserves PaddleOCR's result order for the initial annotation.
Users can reorder blocks manually.

## 5. Backend Components

### Application

Creates the FastAPI app, owns startup/shutdown, serves static assets, and maps
domain errors to consistent JSON responses.

### Settings

Holds validated launch configuration:

- host and port;
- device;
- detection model name or directory;
- recognition model directory;
- recognition input shape;
- confidence warning threshold;
- autosave delay.

### Directory catalog

Opens one active root directory, discovers supported direct-child images using
case-insensitive extensions, and sorts them naturally. It returns opaque image
IDs rather than accepting arbitrary paths from browser requests.

All resolved paths must remain inside the selected root. The catalog excludes
`.paddleocr-labeler`.

### OCR engine

Wraps PaddleOCR construction and result normalization behind a small
interface. It rejects unreadable images and model initialization failures with
actionable messages.

### Annotation store

Loads, validates, and atomically saves JSON documents. A save writes a
temporary file in the target directory, flushes it, and replaces the final
file. The store recomputes aggregate text from block order.

Each save supplies the last known revision. A mismatched revision returns HTTP
409 and never overwrites the newer document.

### Batch manager

Maintains one in-memory job with these states:

```text
queued -> running -> completed
                  -> cancelling -> cancelled
                  -> failed
```

Cancellation takes effect after the current image finishes. A batch skips
images with a valid saved annotation. Re-running a batch after restart safely
continues because completed images are discovered from sidecars.

Job state itself is not persisted; persisted annotations are the resume
checkpoint.

### Exporter

Validates all saved annotations and atomically writes:

```text
.paddleocr-labeler/manifest.jsonl
```

One invalid sidecar makes export fail with the affected image and validation
reason instead of producing a partially trustworthy manifest.

## 6. API Contract

The initial API surface is:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Model, device, and service readiness |
| `POST` | `/api/workspace/open` | Open and validate a local image directory |
| `GET` | `/api/images` | List images and annotation/OCR states |
| `GET` | `/api/images/{image_id}/content` | Serve an image from the active root |
| `GET` | `/api/images/{image_id}/annotation` | Load annotation or an empty draft |
| `PUT` | `/api/images/{image_id}/annotation` | Validate and save one revision |
| `POST` | `/api/images/{image_id}/ocr` | OCR the current image |
| `POST` | `/api/batch` | Start sequential OCR for eligible images |
| `GET` | `/api/batch` | Read batch status and progress |
| `POST` | `/api/batch/cancel` | Request cancellation |
| `POST` | `/api/export` | Write and return manifest metadata |

OCR on an image that already has non-empty saved blocks requires
`replace_existing=true`. The frontend displays a confirmation before sending
that flag.

## 7. Annotation Schema

Per-image annotations are stored under:

```text
<image-root>/.paddleocr-labeler/annotations/<image-stem>.json
```

Duplicate stems with different extensions are rejected when the workspace is
opened so two images cannot map to the same sidecar.

Example:

```json
{
  "version": 1,
  "image": {
    "path": "page-001.png",
    "width": 1275,
    "height": 2100,
    "sha256": "hex-digest"
  },
  "revision": 3,
  "status": "edited",
  "text": "First line\nSecond line",
  "blocks": [
    {
      "id": "2fe903f0-66e2-4ced-bbb5-5ff9d8893214",
      "order": 0,
      "text": "First line",
      "polygon": [[120, 80], [860, 80], [860, 132], [120, 132]],
      "score": 0.96,
      "source": "ocr"
    },
    {
      "id": "9c1b2a80-a24f-4874-9036-24781174016c",
      "order": 1,
      "text": "Second line",
      "polygon": [[118, 148], [900, 148], [900, 201], [118, 201]],
      "score": 0.58,
      "source": "ocr"
    }
  ],
  "ocr": {
    "det_model": "PP-OCRv6_medium_det",
    "rec_model": "vi_rec_3datasets_v1",
    "duration_ms": 914
  },
  "updated_at": "2026-07-31T12:00:00+07:00"
}
```

Rules:

- coordinates are absolute image pixels;
- every polygon has exactly four finite points;
- points are clamped to image bounds when edited;
- block IDs are UUIDs and remain stable across text/geometry edits;
- orders are unique, contiguous integers starting at zero;
- aggregate `text` is block text joined with newline in reading order;
- `score` is a number from zero to one for OCR blocks and `null` for manual
  blocks;
- `status` is `ocr`, `edited`, or `completed`;
- empty block text is valid while editing but prevents `completed` status;
- the image hash and dimensions detect source-image replacement.

JSONL repeats portable fields only:

```json
{"image":"page-001.png","width":1275,"height":2100,"text":"First line\nSecond line","blocks":[{"id":"2fe903f0-66e2-4ced-bbb5-5ff9d8893214","order":0,"text":"First line","polygon":[[120,80],[860,80],[860,132],[120,132]],"score":0.96,"source":"ocr"},{"id":"9c1b2a80-a24f-4874-9036-24781174016c","order":1,"text":"Second line","polygon":[[118,148],[900,148],[900,201],[118,201]],"score":0.58,"source":"ocr"}]}
```

## 8. User Interface

### Header

- Product name.
- Active folder selector/path.
- Current batch progress.
- Model/GPU readiness indicator.
- `OCR ảnh này`.
- `OCR toàn folder`.
- `Dừng`.
- `Đánh dấu hoàn tất` or `Mở lại để sửa`.
- `Xuất JSONL`.

### Left sidebar

- Image count.
- Search by filename.
- Filters: all, not OCRed, OCRed, edited, completed, error.
- Naturally sorted image list.
- Status icon and compact dimensions for each image.

### Central workspace

- Dark canvas-style background.
- Image scaled without changing annotation coordinates.
- SVG overlay for polygons, labels, handles, and pointer events.
- Green normal boxes.
- Blue selected box.
- Orange low-confidence boxes.
- Zoom using the wheel.
- Pan using middle mouse or Space plus drag.
- Fit-to-page action and zoom percentage.

### Right inspector

- Reading-order block list with drag handles.
- Selected-region crop preview.
- Confidence and source.
- Plain textarea preserving Vietnamese Unicode and line breaks.
- Four point coordinate editor.
- Delete action.

### Editor behavior

- Selecting a polygon selects and scrolls its inspector entry.
- Selecting an inspector entry centers and highlights its polygon.
- Dragging inside a polygon moves all points.
- Dragging a corner moves that point.
- Add mode creates an axis-aligned rectangle that can later be adjusted by
  corner.
- Delete removes the selected block after confirmation when it contains text.
- Reordering rewrites all contiguous order values.
- Undo and redo cover text, geometry, add, delete, and reorder changes.
- Autosave begins 500 ms after the last mutation.
- The header shows `Đang lưu`, `Đã lưu`, `Xung đột`, or `Lỗi lưu`.

The UI is desktop-first and optimized for screens at least 1280 pixels wide.

## 9. Error Handling

- Invalid or inaccessible root: reject workspace opening with the resolved
  reason.
- Unsupported or corrupted image: mark the image as error and allow the batch
  to continue.
- Missing models or unusable device: fail service readiness with an actionable
  startup error.
- OCR failure: preserve existing annotations, record the error in batch state,
  and continue with the next image.
- Save validation failure: return field-level errors without modifying disk.
- Revision conflict: return HTTP 409 and require reload or explicit user
  reconciliation.
- Changed source image: block annotation save/export until the annotation is
  reset or the user re-runs OCR.
- Export validation failure: report every affected image and do not replace the
  previous valid manifest.

## 10. Resource Strategy

The RTX 5060 Ti has enough VRAM for one PP-OCRv6 medium detection and
recognition pipeline. The service avoids avoidable pressure by:

- creating one pipeline;
- serializing inference;
- batching recognition inside PaddleOCR rather than running page jobs in
  parallel;
- releasing per-image arrays after normalization;
- serving image bytes without keeping the entire directory in RAM; and
- limiting UI thumbnails to browser-scaled previews.

The service uses one Uvicorn worker because multiple workers would load
duplicate GPU models.

## 11. Verification Strategy

### Python unit tests

- schema validation and aggregate text;
- polygon clamping and malformed coordinates;
- root confinement and opaque image IDs;
- duplicate-stem rejection;
- image hash/dimension change detection;
- atomic save behavior;
- revision conflict behavior;
- JSONL validation and atomic export;
- OCR result normalization, including empty and low-confidence results;
- batch skip, progress, failure continuation, and cancellation.

### API integration tests

- service health;
- workspace open and image listing;
- protected image serving;
- annotation create/load/update;
- HTTP 409 revision conflict;
- current-image OCR replacement protection;
- batch lifecycle;
- export response and manifest content.

### Frontend tests

Use Node's built-in test runner for dependency-free editor state functions:

- selection;
- polygon movement and corner editing;
- adding/deleting blocks;
- reordering;
- undo/redo;
- autosave state transitions.

### Live verification

- Start the real service with the project virtual environment.
- Load the fine-tuned recognition model and PP-OCRv6 medium detector on
  `gpu:0`.
- OCR at least one full-page image.
- Confirm text, confidence, and polygons render.
- Edit text and geometry, reload, and confirm persistence.
- Run a small batch and cancel it.
- Resume batch and confirm completed images are skipped.
- Export JSONL and parse every line.

## 12. Acceptance Criteria

The feature is complete when:

1. One command launches the local service with the existing fine-tuned model.
2. A local image folder opens without exposing files outside that root.
3. Current-image OCR returns editable text and four-point polygons on GPU.
4. Full text, geometry, add/delete, reorder, undo, and redo work in the browser.
5. Autosaved annotations survive reload without corruption.
6. Revision conflicts cannot silently overwrite newer work.
7. Sequential batch OCR reports progress, continues past bad images, cancels
   safely, and resumes by skipping completed annotations.
8. Source images remain unchanged.
9. Export produces valid, portable JSONL with one record per saved image.
10. Automated tests and a real GPU/browser smoke flow pass.

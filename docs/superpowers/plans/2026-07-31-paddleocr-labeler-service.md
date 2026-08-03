# PaddleOCR Labeler Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight local browser service that runs PP-OCRv6 detection with the fine-tuned Vietnamese recognition model and provides full text, polygon, reading-order, autosave, batch, and JSONL editing workflows.

**Architecture:** A single-process FastAPI application serves a build-free HTML/CSS/JavaScript frontend and owns one PaddleOCR pipeline. Focused Python modules implement schemas, workspace confinement, atomic storage, OCR normalization, and a sequential batch worker; pure JavaScript modules implement immutable editor state and SVG coordinate transforms.

**Tech Stack:** Python 3.12, PaddlePaddle 3.3.0, PaddleOCR 3.7.0, FastAPI, Uvicorn, Pydantic 2, Pillow, Python `unittest`, vanilla ES modules, SVG, Node 24 built-in test runner.

## Global Constraints

- Keep all application code under a new top-level `ocr_labeler/` package; do not modify `finetune.py` or PaddleOCR source.
- Use `PP-OCRv6_medium_det` and default recognition directory `runs/vi_rec_3datasets_v1/inference/best_accuracy`.
- Use recognition input shape `3,48,1600`, recognition score threshold `0.0`, and visual warning threshold `0.60`.
- Disable document orientation classification, document unwarping, and text-line orientation classification.
- Default to `gpu:0`; never silently fall back to CPU.
- Create exactly one PaddleOCR pipeline and serialize all inference.
- Scan only direct-child PNG, JPEG, WebP, BMP, and TIFF files.
- Every block has the implicit type `Text`; do not add layout classification or rich HTML.
- Store absolute pixel polygons with exactly four finite points.
- Save under `<image-root>/.paddleocr-labeler/` using atomic replacement and revision checks.
- Preserve source images unchanged.
- Runtime frontend must have no Node dependency or build step.
- Run Uvicorn with one worker to avoid duplicate GPU models.
- The workspace root is not a Git repository. Do not initialize Git or claim commits; record passing verification commands at each checkpoint.

---

## File Map

```text
ocr_labeler/
├── __init__.py              package metadata
├── settings.py              validated launch/runtime settings
├── models.py                Pydantic API and annotation contracts
├── geometry.py              polygon normalization and ordering helpers
├── catalog.py               confined workspace and image discovery
├── storage.py               sidecars, revisions, atomic writes, export
├── ocr_engine.py            PaddleOCR creation and result normalization
├── batch.py                 serialized inference and batch lifecycle
├── app.py                   FastAPI construction and routes
├── cli.py                   command-line parsing and Uvicorn launch
└── static/
    ├── index.html           three-column application shell
    ├── styles.css           responsive desktop-first styling
    ├── state.mjs            immutable editor state and history
    ├── geometry.mjs         SVG/image coordinate transforms
    └── app.mjs              API, rendering, events, autosave, batch polling
run_labeler.py               minimal executable entry point
requirements-labeler.txt     web/test-only Python dependencies
tests/
├── test_labeler_models.py
├── test_labeler_catalog_storage.py
├── test_labeler_ocr_engine.py
├── test_labeler_batch.py
├── test_labeler_app.py
├── test_labeler_cli.py
├── test_labeler_static.py
└── frontend/
    ├── state.test.mjs
    └── geometry.test.mjs
```

---

### Task 1: Annotation Models and Polygon Rules

**Files:**
- Create: `ocr_labeler/__init__.py`
- Create: `ocr_labeler/models.py`
- Create: `ocr_labeler/geometry.py`
- Create: `tests/test_labeler_models.py`

**Interfaces:**
- Produces: `Point`, `Block`, `ImageInfo`, `OCRMetadata`, `Annotation`, `AnnotationStatus`.
- Produces: `clamp_polygon(polygon, width, height)`, `normalize_annotation(annotation)`, and `aggregate_text(blocks)`.
- Consumes: only Pydantic 2 and Python standard library.

- [ ] **Step 1: Write failing model and geometry tests**

```python
# tests/test_labeler_models.py
import math
import unittest
from uuid import UUID

from pydantic import ValidationError

from ocr_labeler.geometry import clamp_polygon, normalize_annotation
from ocr_labeler.models import Annotation, Block, ImageInfo


class LabelerModelTests(unittest.TestCase):
    def test_annotation_orders_blocks_and_rebuilds_text(self):
        annotation = Annotation(
            image=ImageInfo(
                path="page-001.png", width=100, height=80, sha256="a" * 64
            ),
            blocks=[
                Block(
                    order=7,
                    text="Second",
                    polygon=[(2, 20), (80, 20), (80, 30), (2, 30)],
                    score=0.5,
                    source="ocr",
                ),
                Block(
                    order=2,
                    text="First",
                    polygon=[(1, 1), (90, 1), (90, 10), (1, 10)],
                    score=0.9,
                    source="ocr",
                ),
            ],
        )

        normalized = normalize_annotation(annotation)

        self.assertEqual([block.order for block in normalized.blocks], [0, 1])
        self.assertEqual(normalized.text, "First\nSecond")
        self.assertTrue(all(isinstance(block.id, UUID) for block in normalized.blocks))

    def test_polygon_is_clamped_to_image_bounds(self):
        polygon = [(-2, -3), (120, 0), (101, 91), (0, 90)]
        self.assertEqual(
            clamp_polygon(polygon, width=100, height=80),
            [(0.0, 0.0), (99.0, 0.0), (99.0, 79.0), (0.0, 79.0)],
        )

    def test_polygon_rejects_non_finite_or_wrong_point_count(self):
        with self.assertRaises(ValidationError):
            Block(
                order=0,
                text="bad",
                polygon=[(0, 0), (1, math.inf), (1, 1)],
                score=None,
                source="manual",
            )

    def test_completed_annotation_rejects_empty_block_text(self):
        with self.assertRaises(ValidationError):
            Annotation(
                status="completed",
                image=ImageInfo(
                    path="page.png", width=10, height=10, sha256="b" * 64
                ),
                blocks=[
                    Block(
                        order=0,
                        text=" ",
                        polygon=[(0, 0), (9, 0), (9, 9), (0, 9)],
                        score=None,
                        source="manual",
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the missing-package failure**

Run:

```bash
.venv/bin/python -m unittest tests.test_labeler_models -v
```

Expected: import failure for `ocr_labeler`.

- [ ] **Step 3: Implement the annotation contracts**

```python
# ocr_labeler/models.py
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

Point = tuple[float, float]
AnnotationStatus = Literal["ocr", "edited", "completed"]
BlockSource = Literal["ocr", "manual"]


class ImageInfo(BaseModel):
    path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Block(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    order: int = Field(ge=0)
    text: str
    polygon: list[Point]
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    source: BlockSource

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value: list[Point]) -> list[Point]:
        import math

        if len(value) != 4:
            raise ValueError("polygon must contain exactly four points")
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in value):
            raise ValueError("polygon coordinates must be finite")
        return value


class OCRMetadata(BaseModel):
    det_model: str
    rec_model: str
    duration_ms: int = Field(ge=0)


class Annotation(BaseModel):
    version: Literal[1] = 1
    image: ImageInfo
    revision: int = Field(default=0, ge=0)
    status: AnnotationStatus = "edited"
    text: str = ""
    blocks: list[Block] = Field(default_factory=list)
    ocr: OCRMetadata | None = None
    updated_at: datetime | None = None

    @model_validator(mode="after")
    def validate_completed_text(self):
        if self.status == "completed" and any(not block.text.strip() for block in self.blocks):
            raise ValueError("completed annotations cannot contain empty block text")
        return self
```

```python
# ocr_labeler/geometry.py
from .models import Annotation, Block, Point


def clamp_polygon(
    polygon: list[Point], width: int, height: int
) -> list[Point]:
    max_x = float(width - 1)
    max_y = float(height - 1)
    return [
        (min(max(float(x), 0.0), max_x), min(max(float(y), 0.0), max_y))
        for x, y in polygon
    ]


def aggregate_text(blocks: list[Block]) -> str:
    return "\n".join(block.text for block in sorted(blocks, key=lambda item: item.order))


def normalize_annotation(annotation: Annotation) -> Annotation:
    ordered = sorted(annotation.blocks, key=lambda item: item.order)
    blocks = [
        block.model_copy(
            update={
                "order": order,
                "polygon": clamp_polygon(
                    block.polygon, annotation.image.width, annotation.image.height
                ),
            }
        )
        for order, block in enumerate(ordered)
    ]
    return annotation.model_copy(
        update={"blocks": blocks, "text": aggregate_text(blocks)}
    )
```

Create `ocr_labeler/__init__.py` with `__version__ = "0.1.0"`.

- [ ] **Step 4: Run the focused and existing suites**

```bash
.venv/bin/python -m unittest tests.test_labeler_models -v
.venv/bin/python -m unittest discover -s tests -v
```

Expected: the new tests pass and existing fine-tune tests remain green.

- [ ] **Step 5: Record the checkpoint**

Record the two passing commands in the execution notes. Do not initialize Git.

---

### Task 2: Confined Workspace, Atomic Sidecars, and JSONL Export

**Files:**
- Create: `ocr_labeler/catalog.py`
- Create: `ocr_labeler/storage.py`
- Create: `tests/test_labeler_catalog_storage.py`

**Interfaces:**
- Consumes: `Annotation`, `ImageInfo`, and `normalize_annotation`.
- Produces: immutable `ImageRecord`.
- Produces: `WorkspaceCatalog.open(root)`, `list_images()`, `get(image_id)`.
- Produces: `AnnotationStore.load(record)`, `create_draft(record)`, `save(record, annotation)`, and `export_manifest(catalog)`.
- Produces: `DuplicateStemError`, `UnknownImageError`, `RevisionConflict`, and `SourceImageChanged`.

- [ ] **Step 1: Write failing catalog and storage tests**

```python
# tests/test_labeler_catalog_storage.py
import json
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from ocr_labeler.catalog import DuplicateStemError, WorkspaceCatalog
from ocr_labeler.models import Block
from ocr_labeler.storage import (
    AnnotationStore,
    RevisionConflict,
    SourceImageChanged,
)


def make_image(path: Path, size=(64, 32), color="white"):
    Image.new("RGB", size, color).save(path)


class CatalogStorageTests(unittest.TestCase):
    def test_catalog_scans_direct_images_only_and_uses_opaque_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page-2.PNG")
            make_image(root / "page-10.jpg")
            nested = root / "nested"
            nested.mkdir()
            make_image(nested / "ignored.png")

            catalog = WorkspaceCatalog.open(root)
            records = catalog.list_images()

        self.assertEqual([record.name for record in records], ["page-2.PNG", "page-10.jpg"])
        self.assertTrue(all("/" not in record.image_id for record in records))

    def test_duplicate_stems_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            make_image(root / "page.jpg")
            with self.assertRaises(DuplicateStemError):
                WorkspaceCatalog.open(root)

    def test_symlink_cannot_escape_workspace_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "images"
            root.mkdir()
            outside = base / "outside.png"
            make_image(outside)
            (root / "leak.png").symlink_to(outside)
            with self.assertRaises(ValueError):
                WorkspaceCatalog.open(root)

    def test_corrupt_image_is_listed_as_error_without_blocking_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.png").write_bytes(b"not an image")
            records = WorkspaceCatalog.open(root).list_images()
        self.assertEqual(len(records), 1)
        self.assertIsNotNone(records[0].error)

    def test_save_is_revision_checked_and_export_is_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            catalog = WorkspaceCatalog.open(root)
            record = catalog.list_images()[0]
            store = AnnotationStore(root)
            annotation = store.create_draft(record)
            annotation.blocks = [
                Block(
                    order=0,
                    text="Việt Nam",
                    polygon=[(1, 1), (50, 1), (50, 20), (1, 20)],
                    score=None,
                    source="manual",
                )
            ]

            saved = store.save(record, annotation)
            with self.assertRaises(RevisionConflict):
                store.save(record, annotation)
            manifest = store.export_manifest(catalog)
            row = json.loads(manifest.read_text(encoding="utf-8").strip())

        self.assertEqual(saved.revision, 1)
        self.assertEqual(row["image"], "page.png")
        self.assertEqual(row["text"], "Việt Nam")
        self.assertEqual(len(row["blocks"]), 1)

    def test_changed_source_image_blocks_load_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            catalog = WorkspaceCatalog.open(root)
            record = catalog.list_images()[0]
            store = AnnotationStore(root)
            store.save(record, store.create_draft(record))
            make_image(root / "page.png", size=(80, 40), color="black")
            with self.assertRaises(SourceImageChanged):
                store.load(record)
            with self.assertRaises(SourceImageChanged):
                store.export_manifest(catalog)

    def test_atomic_save_leaves_no_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            record = WorkspaceCatalog.open(root).list_images()[0]
            store = AnnotationStore(root)
            store.save(record, store.create_draft(record))
            leftovers = list(store.annotation_dir.glob(".page.json.*"))
        self.assertEqual(leftovers, [])
```

- [ ] **Step 2: Verify the tests fail because catalog/storage are absent**

```bash
.venv/bin/python -m unittest tests.test_labeler_catalog_storage -v
```

Expected: import failure for `ocr_labeler.catalog`.

- [ ] **Step 3: Implement confined image discovery**

```python
# ocr_labeler/catalog.py
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class DuplicateStemError(ValueError):
    pass


class UnknownImageError(KeyError):
    pass


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    name: str
    path: Path
    relative_path: str
    width: int | None
    height: int | None
    sha256: str
    size_bytes: int
    mtime_ns: int
    error: str | None = None


def _natural_key(value: str):
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value)]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WorkspaceCatalog:
    def __init__(self, root: Path, records: list[ImageRecord]):
        self.root = root
        self._records = records
        self._by_id = {record.image_id: record for record in records}

    @classmethod
    def open(cls, root: Path) -> "WorkspaceCatalog":
        resolved = root.expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("workspace root must be a directory")
        paths = sorted(
            (
                path
                for path in resolved.iterdir()
                if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
            ),
            key=lambda path: _natural_key(path.name),
        )
        stems = [path.stem.casefold() for path in paths]
        if len(stems) != len(set(stems)):
            raise DuplicateStemError("image filenames must have unique stems")
        records = []
        for path in paths:
            resolved_path = path.resolve(strict=True)
            if resolved_path.parent != resolved:
                raise ValueError(f"image symlink escapes workspace: {path.name}")
            relative = path.relative_to(resolved).as_posix()
            image_id = sha256(relative.encode("utf-8")).hexdigest()[:24]
            stat = path.stat()
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                error = None
            except Exception as exc:
                width, height = None, None
                error = f"{type(exc).__name__}: {exc}"
            records.append(
                ImageRecord(
                    image_id=image_id,
                    name=path.name,
                    path=path,
                    relative_path=relative,
                    width=width,
                    height=height,
                    sha256=_file_sha256(path),
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    error=error,
                )
            )
        return cls(resolved, records)

    def list_images(self) -> list[ImageRecord]:
        return list(self._records)

    def get(self, image_id: str) -> ImageRecord:
        try:
            record = self._by_id[image_id]
        except KeyError as exc:
            raise UnknownImageError(image_id) from exc
        if record.path.resolve(strict=True).parent != self.root:
            raise UnknownImageError(image_id)
        return record
```

- [ ] **Step 4: Implement revisions, atomic writes, stale-image checks, and export**

```python
# ocr_labeler/storage.py
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from .catalog import ImageRecord, WorkspaceCatalog
from .geometry import normalize_annotation
from .models import Annotation, ImageInfo


class RevisionConflict(RuntimeError):
    pass


class SourceImageChanged(RuntimeError):
    pass


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


class AnnotationStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.data_dir = self.root / ".paddleocr-labeler"
        self.annotation_dir = self.data_dir / "annotations"

    def _path(self, record: ImageRecord) -> Path:
        return self.annotation_dir / f"{Path(record.relative_path).stem}.json"

    def create_draft(self, record: ImageRecord) -> Annotation:
        if record.error or record.width is None or record.height is None:
            raise ValueError(f"invalid source image: {record.name}: {record.error}")
        return Annotation(
            image=ImageInfo(
                path=record.relative_path,
                width=record.width,
                height=record.height,
                sha256=record.sha256,
            )
        )

    def load(self, record: ImageRecord) -> Annotation:
        path = self._path(record)
        if not path.exists():
            return self.create_draft(record)
        annotation = Annotation.model_validate_json(path.read_text(encoding="utf-8"))
        self._assert_source(record, annotation)
        return annotation

    def _assert_source(self, record: ImageRecord, annotation: Annotation) -> None:
        stat = record.path.stat()
        if stat.st_size != record.size_bytes or stat.st_mtime_ns != record.mtime_ns:
            raise SourceImageChanged(record.relative_path)
        expected = (record.sha256, record.width, record.height, record.relative_path)
        actual = (
            annotation.image.sha256,
            annotation.image.width,
            annotation.image.height,
            annotation.image.path,
        )
        if actual != expected:
            raise SourceImageChanged(record.relative_path)

    def save(self, record: ImageRecord, annotation: Annotation) -> Annotation:
        self._assert_source(record, annotation)
        path = self._path(record)
        current_revision = 0
        if path.exists():
            current_revision = Annotation.model_validate_json(
                path.read_text(encoding="utf-8")
            ).revision
        if annotation.revision != current_revision:
            raise RevisionConflict(
                f"expected revision {current_revision}, got {annotation.revision}"
            )
        saved = normalize_annotation(annotation).model_copy(
            update={
                "revision": current_revision + 1,
                "updated_at": datetime.now().astimezone(),
            }
        )
        _atomic_text(path, saved.model_dump_json(indent=2))
        return saved

    def export_manifest(self, catalog: WorkspaceCatalog) -> Path:
        rows = []
        for record in catalog.list_images():
            path = self._path(record)
            if not path.exists():
                continue
            annotation = self.load(record)
            rows.append(
                json.dumps(
                    {
                        "image": annotation.image.path,
                        "width": annotation.image.width,
                        "height": annotation.image.height,
                        "text": annotation.text,
                        "blocks": [
                            block.model_dump(mode="json") for block in annotation.blocks
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        manifest = self.data_dir / "manifest.jsonl"
        _atomic_text(manifest, "".join(f"{row}\n" for row in rows))
        return manifest
```

- [ ] **Step 5: Run focused tests and full regression**

```bash
.venv/bin/python -m unittest tests.test_labeler_catalog_storage -v
.venv/bin/python -m unittest discover -s tests -v
```

Expected: catalog/storage tests pass; all previous tests stay green.

---

### Task 3: PaddleOCR Engine Adapter

**Files:**
- Create: `ocr_labeler/settings.py`
- Create: `ocr_labeler/ocr_engine.py`
- Create: `tests/test_labeler_ocr_engine.py`

**Interfaces:**
- Consumes: `ImageRecord`, `Annotation`, `Block`, and `OCRMetadata`.
- Produces: frozen `LabelerSettings`.
- Produces: `PaddleOCREngine.create(settings)` and `recognize(record) -> Annotation`.
- Produces: `normalize_ocr_result(result, record, settings, duration_ms)`.

- [ ] **Step 1: Write failing result-normalization and configuration tests**

```python
# tests/test_labeler_ocr_engine.py
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ocr_labeler.catalog import WorkspaceCatalog
from ocr_labeler.ocr_engine import PaddleOCREngine
from ocr_labeler.settings import LabelerSettings


class FakeResult(dict):
    pass


class FakePipeline:
    def predict(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        return [
            FakeResult(
                rec_texts=["Xin chào", "Việt Nam"],
                rec_scores=[0.98, 0.42],
                rec_polys=[
                    [[1, 2], [40, 2], [40, 12], [1, 12]],
                    [[2, 15], [55, 15], [55, 25], [2, 25]],
                ],
            )
        ]


class OCREngineTests(unittest.TestCase):
    def test_fake_pipeline_is_normalized_without_filtering_low_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (60, 30), "white").save(root / "page.png")
            record = WorkspaceCatalog.open(root).list_images()[0]
            settings = LabelerSettings(rec_model_dir=Path("runs/model"))
            pipeline = FakePipeline()
            engine = PaddleOCREngine(settings=settings, pipeline=pipeline)

            annotation = engine.recognize(record)

        self.assertEqual(annotation.status, "ocr")
        self.assertEqual(annotation.text, "Xin chào\nViệt Nam")
        self.assertEqual([block.score for block in annotation.blocks], [0.98, 0.42])
        self.assertEqual(pipeline.kwargs["text_rec_score_thresh"], 0.0)

    def test_settings_use_quality_defaults(self):
        settings = LabelerSettings(rec_model_dir=Path("runs/model"))
        self.assertEqual(settings.text_rec_input_shape, (3, 48, 1600))
        self.assertEqual(settings.device, "gpu:0")
        self.assertEqual(settings.confidence_warning_threshold, 0.60)
```

- [ ] **Step 2: Verify the adapter tests fail**

```bash
.venv/bin/python -m unittest tests.test_labeler_ocr_engine -v
```

Expected: import failure for `ocr_labeler.ocr_engine`.

- [ ] **Step 3: Implement immutable settings**

```python
# ocr_labeler/settings.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabelerSettings:
    rec_model_dir: Path
    det_model_dir: Path | None = None
    det_model_name: str = "PP-OCRv6_medium_det"
    device: str = "gpu:0"
    text_rec_input_shape: tuple[int, int, int] = (3, 48, 1600)
    text_rec_score_thresh: float = 0.0
    confidence_warning_threshold: float = 0.60
    autosave_delay_ms: int = 500
    host: str = "127.0.0.1"
    port: int = 8010

    def validate(self) -> "LabelerSettings":
        rec_dir = self.rec_model_dir.expanduser().resolve()
        required = {
            "inference.json",
            "inference.pdiparams",
            "inference.yml",
            "ppocr_keys.txt",
        }
        missing = sorted(name for name in required if not (rec_dir / name).is_file())
        if missing:
            raise ValueError(f"recognition model is missing: {', '.join(missing)}")
        if self.det_model_dir is not None:
            det_dir = self.det_model_dir.expanduser().resolve()
            det_required = {"inference.json", "inference.pdiparams", "inference.yml"}
            det_missing = sorted(
                name for name in det_required if not (det_dir / name).is_file()
            )
            if det_missing:
                raise ValueError(
                    f"detection model is missing: {', '.join(det_missing)}"
                )
        if self.device != "cpu" and not self.device.startswith("gpu:"):
            raise ValueError("device must be cpu or gpu:<index>")
        return self
```

- [ ] **Step 4: Implement PaddleOCR creation and direct result access**

```python
# ocr_labeler/ocr_engine.py
from time import perf_counter

from .catalog import ImageRecord
from .geometry import normalize_annotation
from .models import Annotation, Block, ImageInfo, OCRMetadata
from .settings import LabelerSettings


class PaddleOCREngine:
    def __init__(self, settings: LabelerSettings, pipeline):
        self.settings = settings
        self.pipeline = pipeline

    @classmethod
    def create(cls, settings: LabelerSettings) -> "PaddleOCREngine":
        from paddleocr import PaddleOCR

        settings.validate()
        model_args = (
            {"text_detection_model_dir": str(settings.det_model_dir)}
            if settings.det_model_dir
            else {"text_detection_model_name": settings.det_model_name}
        )
        pipeline = PaddleOCR(
            **model_args,
            text_recognition_model_dir=str(settings.rec_model_dir),
            text_rec_input_shape=settings.text_rec_input_shape,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=settings.device,
        )
        return cls(settings=settings, pipeline=pipeline)

    def recognize(self, record: ImageRecord) -> Annotation:
        started = perf_counter()
        results = self.pipeline.predict(
            str(record.path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=self.settings.text_rec_score_thresh,
        )
        if len(results) != 1:
            raise RuntimeError(f"expected one OCR result, got {len(results)}")
        duration_ms = round((perf_counter() - started) * 1000)
        return normalize_ocr_result(results[0], record, self.settings, duration_ms)

    def close(self) -> None:
        close = getattr(self.pipeline, "close", None)
        if close is not None:
            close()


def normalize_ocr_result(result, record, settings, duration_ms: int) -> Annotation:
    texts = list(result["rec_texts"])
    scores = list(result["rec_scores"])
    polygons = list(result["rec_polys"])
    if not (len(texts) == len(scores) == len(polygons)):
        raise RuntimeError("PaddleOCR result arrays have different lengths")
    blocks = [
        Block(
            order=order,
            text=str(text),
            polygon=[(float(x), float(y)) for x, y in polygon],
            score=float(score),
            source="ocr",
        )
        for order, (text, score, polygon) in enumerate(zip(texts, scores, polygons))
    ]
    annotation = Annotation(
        image=ImageInfo(
            path=record.relative_path,
            width=record.width,
            height=record.height,
            sha256=record.sha256,
        ),
        status="ocr",
        blocks=blocks,
        ocr=OCRMetadata(
            det_model=(
                str(settings.det_model_dir)
                if settings.det_model_dir
                else settings.det_model_name
            ),
            rec_model=settings.rec_model_dir.name,
            duration_ms=duration_ms,
        ),
    )
    return normalize_annotation(annotation)
```

- [ ] **Step 5: Run focused adapter tests**

```bash
.venv/bin/python -m unittest tests.test_labeler_ocr_engine -v
```

Expected: both tests pass without loading PaddleOCR or allocating GPU memory.

---

### Task 4: Serialized Inference and Batch Lifecycle

**Files:**
- Create: `ocr_labeler/batch.py`
- Create: `tests/test_labeler_batch.py`

**Interfaces:**
- Consumes: engine `recognize(record)`, `WorkspaceCatalog`, and `AnnotationStore`.
- Produces: `InferenceCoordinator.recognize(record)`.
- Produces: `BatchManager.start(catalog, store)`, `cancel()`, `snapshot()`.
- Produces: `BatchSnapshot` with state, total, processed, skipped, failed, current image, and error records.

- [ ] **Step 1: Write failing serialization, skip, failure, and cancellation tests**

```python
# tests/test_labeler_batch.py
import tempfile
import threading
import time
import unittest
from pathlib import Path

from PIL import Image

from ocr_labeler.batch import BatchManager, InferenceCoordinator
from ocr_labeler.catalog import WorkspaceCatalog
from ocr_labeler.storage import AnnotationStore


class FakeEngine:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = []

    def recognize(self, record):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append(record.name)
        time.sleep(0.01)
        annotation = AnnotationStore(record.path.parent).create_draft(record)
        annotation.status = "ocr"
        self.active -= 1
        return annotation


class BatchTests(unittest.TestCase):
    def test_coordinator_serializes_concurrent_requests(self):
        engine = FakeEngine()
        coordinator = InferenceCoordinator(engine)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.png", "b.png"):
                Image.new("RGB", (20, 10), "white").save(root / name)
            records = WorkspaceCatalog.open(root).list_images()
            threads = [
                threading.Thread(target=coordinator.recognize, args=(record,))
                for record in records
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(engine.max_active, 1)

    def test_batch_saves_each_result_and_completes(self):
        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.png", "b.png"):
                Image.new("RGB", (20, 10), "white").save(root / name)
            catalog = WorkspaceCatalog.open(root)
            store = AnnotationStore(root)
            manager = BatchManager(InferenceCoordinator(engine))
            manager.start(catalog, store)
            manager.wait(timeout=2)
            snapshot = manager.snapshot()
        self.assertEqual(snapshot.state, "completed")
        self.assertEqual(snapshot.processed, 2)
        self.assertEqual(snapshot.failed, 0)

    def test_batch_skips_saved_annotations_and_continues_past_bad_image(self):
        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 10), "white").save(root / "saved.png")
            Image.new("RGB", (20, 10), "white").save(root / "fresh.png")
            (root / "broken.png").write_bytes(b"not an image")
            catalog = WorkspaceCatalog.open(root)
            store = AnnotationStore(root)
            saved_record = next(
                record for record in catalog.list_images() if record.name == "saved.png"
            )
            store.save(saved_record, store.create_draft(saved_record))
            manager = BatchManager(InferenceCoordinator(engine))
            manager.start(catalog, store)
            manager.wait(timeout=2)
            snapshot = manager.snapshot()
        self.assertEqual(snapshot.state, "completed")
        self.assertEqual(snapshot.skipped, 1)
        self.assertEqual(snapshot.processed, 1)
        self.assertEqual(snapshot.failed, 1)

    def test_cancel_finishes_current_image_then_stops(self):
        class BlockingEngine(FakeEngine):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.release = threading.Event()

            def recognize(self, record):
                self.started.set()
                self.release.wait(timeout=1)
                return super().recognize(record)

        engine = BlockingEngine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.png", "b.png", "c.png"):
                Image.new("RGB", (20, 10), "white").save(root / name)
            catalog = WorkspaceCatalog.open(root)
            manager = BatchManager(InferenceCoordinator(engine))
            manager.start(catalog, AnnotationStore(root))
            self.assertTrue(engine.started.wait(timeout=1))
            manager.cancel()
            engine.release.set()
            manager.wait(timeout=2)
            snapshot = manager.snapshot()
        self.assertEqual(snapshot.state, "cancelled")
        self.assertEqual(snapshot.processed, 1)
```

Import `AnnotationStore` inside `FakeEngine` as shown so the test creates valid
draft annotations without coupling the fake to application internals.

- [ ] **Step 2: Verify tests fail because the batch module is absent**

```bash
.venv/bin/python -m unittest tests.test_labeler_batch -v
```

Expected: import failure for `ocr_labeler.batch`.

- [ ] **Step 3: Implement the inference lock and snapshot contract**

```python
# ocr_labeler/batch.py
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from threading import Event, Lock, Thread


@dataclass(frozen=True)
class BatchError:
    image: str
    message: str


@dataclass
class BatchSnapshot:
    state: str = "idle"
    total: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    current_image: str | None = None
    errors: list[BatchError] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class InferenceCoordinator:
    def __init__(self, engine):
        self.engine = engine
        self._lock = Lock()

    def recognize(self, record):
        with self._lock:
            return self.engine.recognize(record)
```

- [ ] **Step 4: Implement sequential batch execution**

```python
class BatchManager:
    def __init__(self, coordinator: InferenceCoordinator):
        self.coordinator = coordinator
        self._state_lock = Lock()
        self._cancel = Event()
        self._thread: Thread | None = None
        self._snapshot = BatchSnapshot()

    def start(self, catalog, store) -> BatchSnapshot:
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("a batch job is already running")
            self._cancel.clear()
            self._snapshot = BatchSnapshot(
                state="queued", total=len(catalog.list_images())
            )
            self._thread = Thread(
                target=self._run, args=(catalog, store), daemon=True
            )
            self._thread.start()
        return self.snapshot()

    def _run(self, catalog, store):
        self._set(state="running")
        for record in catalog.list_images():
            if self._cancel.is_set():
                self._set(state="cancelled", current_image=None)
                return
            self._set(current_image=record.name)
            if record.error:
                self._add_error(record.name, record.error)
                continue
            if store.has_annotation(record):
                try:
                    store.load(record)
                    self._increment("skipped")
                except Exception as exc:
                    self._add_error(record.name, str(exc))
                continue
            try:
                annotation = self.coordinator.recognize(record)
                store.save(record, annotation)
                self._increment("processed")
            except Exception as exc:
                self._add_error(record.name, str(exc))
        self._set(state="completed", current_image=None)

    def cancel(self) -> BatchSnapshot:
        if self.snapshot().state in {"queued", "running"}:
            self._set(state="cancelling")
            self._cancel.set()
        return self.snapshot()

    def snapshot(self) -> BatchSnapshot:
        with self._state_lock:
            return deepcopy(self._snapshot)

    def wait(self, timeout=None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _set(self, **values) -> None:
        with self._state_lock:
            for name, value in values.items():
                setattr(self._snapshot, name, value)

    def _increment(self, name: str) -> None:
        with self._state_lock:
            setattr(self._snapshot, name, getattr(self._snapshot, name) + 1)

    def _add_error(self, image: str, message: str) -> None:
        with self._state_lock:
            self._snapshot.failed += 1
            self._snapshot.errors.append(BatchError(image=image, message=message))
```

Add this exact helper to `AnnotationStore`:

```python
def has_annotation(self, record: ImageRecord) -> bool:
    return self._path(record).is_file()
```

- [ ] **Step 5: Run batch and full suites**

```bash
.venv/bin/python -m unittest tests.test_labeler_batch -v
.venv/bin/python -m unittest discover -s tests -v
```

Expected: no overlap in fake inference; batch progress and saved sidecars pass.

---

### Task 5: FastAPI Application and HTTP Contract

**Files:**
- Create: `requirements-labeler.txt`
- Create: `ocr_labeler/app.py`
- Create: `tests/test_labeler_app.py`

**Interfaces:**
- Consumes: settings, catalog, store, coordinator, batch manager, and OCR engine.
- Produces: `create_app(settings, engine=None) -> FastAPI`.
- Produces the approved `/api/health`, workspace, image, annotation, OCR, batch,
  cancel, and export routes.

- [ ] **Step 1: Add the isolated web/test dependency file and install it**

```text
# requirements-labeler.txt
fastapi>=0.116,<1
uvicorn>=0.35,<1
httpx>=0.28,<1
```

Run:

```bash
.venv/bin/python -m pip install -r requirements-labeler.txt
```

Expected: FastAPI, Uvicorn, and HTTPX install into the existing project venv;
PaddlePaddle remains 3.3.0.

- [ ] **Step 2: Write failing API tests using an injected fake engine**

```python
# tests/test_labeler_app.py
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from ocr_labeler.app import create_app
from ocr_labeler.models import Block
from ocr_labeler.settings import LabelerSettings
from ocr_labeler.storage import AnnotationStore


class FakeEngine:
    def recognize(self, record):
        annotation = AnnotationStore(record.path.parent).create_draft(record)
        annotation.status = "ocr"
        annotation.blocks = [
            Block(
                order=0,
                text="Việt Nam",
                polygon=[(1, 1), (30, 1), (30, 10), (1, 10)],
                score=0.95,
                source="ocr",
            )
        ]
        return annotation

    def close(self):
        pass


class LabelerAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.new("RGB", (40, 20), "white").save(self.root / "page.png")
        settings = LabelerSettings(rec_model_dir=Path("unused-in-injected-test"))
        self.client = TestClient(create_app(settings=settings, engine=FakeEngine()))

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def test_open_list_ocr_save_reload_and_export(self):
        opened = self.client.post(
            "/api/workspace/open", json={"path": str(self.root)}
        )
        self.assertEqual(opened.status_code, 200)
        images = self.client.get("/api/images").json()["images"]
        image_id = images[0]["image_id"]

        ocr = self.client.post(
            f"/api/images/{image_id}/ocr", json={"replace_existing": False}
        )
        self.assertEqual(ocr.status_code, 200)
        saved = self.client.put(
            f"/api/images/{image_id}/annotation", json=ocr.json()
        )
        self.assertEqual(saved.json()["revision"], 1)
        loaded = self.client.get(
            f"/api/images/{image_id}/annotation"
        )
        self.assertEqual(loaded.json()["revision"], 1)
        exported = self.client.post("/api/export")
        self.assertEqual(exported.status_code, 200)
        self.assertTrue(Path(exported.json()["path"]).is_file())

    def test_unknown_image_does_not_accept_a_path(self):
        self.client.post("/api/workspace/open", json={"path": str(self.root)})
        response = self.client.get("/api/images/../../etc/passwd/content")
        self.assertIn(response.status_code, {404, 422})

    def test_revision_conflict_and_ocr_replacement_protection(self):
        self.client.post("/api/workspace/open", json={"path": str(self.root)})
        image_id = self.client.get("/api/images").json()["images"][0]["image_id"]
        draft = self.client.post(
            f"/api/images/{image_id}/ocr", json={"replace_existing": False}
        ).json()
        saved = self.client.put(
            f"/api/images/{image_id}/annotation", json=draft
        )
        self.assertEqual(saved.status_code, 200)
        conflict = self.client.put(
            f"/api/images/{image_id}/annotation", json=draft
        )
        self.assertEqual(conflict.status_code, 409)
        protected = self.client.post(
            f"/api/images/{image_id}/ocr", json={"replace_existing": False}
        )
        self.assertEqual(protected.status_code, 409)
        replacement = self.client.post(
            f"/api/images/{image_id}/ocr", json={"replace_existing": True}
        )
        self.assertEqual(replacement.status_code, 200)
        self.assertEqual(replacement.json()["revision"], 1)

    def test_batch_endpoint_reaches_terminal_state(self):
        self.client.post("/api/workspace/open", json={"path": str(self.root)})
        started = self.client.post("/api/batch")
        self.assertEqual(started.status_code, 200)
        for _ in range(50):
            snapshot = self.client.get("/api/batch").json()
            if snapshot["state"] in {"completed", "cancelled", "failed"}:
                break
            time.sleep(0.01)
        self.assertEqual(snapshot["state"], "completed")
        self.assertEqual(snapshot["processed"], 1)
```

- [ ] **Step 3: Verify API tests fail**

```bash
.venv/bin/python -m unittest tests.test_labeler_app -v
```

Expected: import failure for `ocr_labeler.app`.

- [ ] **Step 4: Implement application state, errors, and workspace routes**

```python
# ocr_labeler/app.py
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .batch import BatchManager, InferenceCoordinator
from .catalog import UnknownImageError, WorkspaceCatalog
from .models import Annotation
from .ocr_engine import PaddleOCREngine
from .storage import AnnotationStore, RevisionConflict, SourceImageChanged


class OpenWorkspaceRequest(BaseModel):
    path: str


class OCRRequest(BaseModel):
    replace_existing: bool = False


class AppState:
    def __init__(self, settings, engine):
        self.settings = settings
        self.engine = engine
        self.coordinator = InferenceCoordinator(engine)
        self.batch = BatchManager(self.coordinator)
        self.catalog = None
        self.store = None
        self.workspace_lock = Lock()

    def require_workspace(self):
        if self.catalog is None or self.store is None:
            raise HTTPException(status_code=409, detail="open a workspace first")
        return self.catalog, self.store
```

Implement application creation and exception mapping exactly once:

```python
from fastapi.responses import JSONResponse


def create_app(settings, engine=None, initial_workspace=None) -> FastAPI:
    owns_engine = engine is None
    active_engine = engine or PaddleOCREngine.create(settings)
    state = AppState(settings, active_engine)

    @asynccontextmanager
    async def lifespan(app):
        if initial_workspace is not None:
            catalog = WorkspaceCatalog.open(Path(initial_workspace))
            state.catalog = catalog
            state.store = AnnotationStore(catalog.root)
        yield
        if owns_engine:
            active_engine.close()

    app = FastAPI(title="PaddleOCR Labeler", lifespan=lifespan)
    app.state.labeler = state

    @app.exception_handler(UnknownImageError)
    async def unknown_image_handler(request, exc):
        return JSONResponse(status_code=404, content={"detail": "unknown image"})

    @app.exception_handler(RevisionConflict)
    async def revision_handler(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(SourceImageChanged)
    async def source_changed_handler(request, exc):
        return JSONResponse(
            status_code=409,
            content={"detail": f"source image changed: {exc}"},
        )

    @app.get("/api/health")
    def health():
        return {
            "ready": True,
            "device": settings.device,
            "det_model": (
                str(settings.det_model_dir)
                if settings.det_model_dir
                else settings.det_model_name
            ),
            "rec_model": str(settings.rec_model_dir),
        }

    @app.post("/api/workspace/open")
    def open_workspace(request: OpenWorkspaceRequest):
        if state.batch.snapshot().state in {"queued", "running", "cancelling"}:
            raise HTTPException(status_code=409, detail="cancel the batch first")
        try:
            catalog = WorkspaceCatalog.open(Path(request.path))
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with state.workspace_lock:
            state.catalog = catalog
            state.store = AnnotationStore(catalog.root)
        return {"root": str(catalog.root), "images": len(catalog.list_images())}
```

Return `app` after all routes and static mounts have been registered.

- [ ] **Step 5: Implement annotation, OCR, batch, and export routes**

Use Pydantic response serialization directly from `Annotation.model_dump`.
Implement the route bodies with these contracts:

```python
@app.get("/api/images")
def list_images():
    catalog, store = state.require_workspace()
    images = []
    for record in catalog.list_images():
        saved = store.has_annotation(record)
        status = "error" if record.error else "not_ocr"
        if saved and not record.error:
            status = store.load(record).status
        images.append(
            {
                "image_id": record.image_id,
                "name": record.name,
                "width": record.width,
                "height": record.height,
                "status": status,
                "error": record.error,
            }
        )
    return {"images": images}


@app.get("/api/images/{image_id}/content")
def image_content(image_id: str):
    catalog, _ = state.require_workspace()
    record = catalog.get(image_id)
    return FileResponse(record.path)


@app.get("/api/images/{image_id}/annotation")
def get_annotation(image_id: str):
    catalog, store = state.require_workspace()
    return store.load(catalog.get(image_id))


@app.put("/api/images/{image_id}/annotation")
def save_annotation(image_id: str, annotation: Annotation):
    catalog, store = state.require_workspace()
    return store.save(catalog.get(image_id), annotation)


@app.post("/api/images/{image_id}/ocr")
def ocr_image(image_id: str, request: OCRRequest):
    catalog, store = state.require_workspace()
    record = catalog.get(image_id)
    if record.error:
        raise HTTPException(status_code=422, detail=record.error)
    existing = store.load(record) if store.has_annotation(record) else None
    if existing is not None and not request.replace_existing:
        if existing.blocks:
            raise HTTPException(
                status_code=409,
                detail="annotation already contains blocks; confirm replacement",
            )
    annotation = state.coordinator.recognize(record)
    if existing is not None:
        annotation = annotation.model_copy(update={"revision": existing.revision})
    return annotation


@app.post("/api/batch")
def start_batch():
    catalog, store = state.require_workspace()
    return state.batch.start(catalog, store).to_dict()


@app.get("/api/batch")
def get_batch():
    return state.batch.snapshot().to_dict()


@app.post("/api/batch/cancel")
def cancel_batch():
    return state.batch.cancel().to_dict()


@app.post("/api/export")
def export():
    catalog, store = state.require_workspace()
    manifest = store.export_manifest(catalog)
    records = sum(store.has_annotation(record) for record in catalog.list_images())
    return {"path": str(manifest), "records": records}
```

Batch routes delegate only to `BatchManager`; they never call PaddleOCR
directly.

- [ ] **Step 6: Run API and regression suites**

```bash
.venv/bin/python -m unittest tests.test_labeler_app -v
.venv/bin/python -m unittest discover -s tests -v
```

Expected: API flow passes and all previous tests remain green.

---

### Task 6: Pure Frontend Editor State and History

**Files:**
- Create: `ocr_labeler/static/state.mjs`
- Create: `tests/frontend/state.test.mjs`

**Interfaces:**
- Produces: `createEditorState(annotation)`.
- Produces: `selectBlock`, `updateText`, `moveBlock`, `moveCorner`,
  `reorderBlock`, `setStatus`, `undo`, and `redo`.
- Every mutating function returns a new state and records one undo snapshot.

- [ ] **Step 1: Write failing Node tests**

```javascript
// tests/frontend/state.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import {
  createEditorState,
  moveCorner,
  reorderBlock,
  undo,
  updateText,
} from "../../ocr_labeler/static/state.mjs";

const annotation = {
  revision: 1,
  status: "ocr",
  text: "A\nB",
  image: { width: 100, height: 80 },
  blocks: [
    { id: "a", order: 0, text: "A", polygon: [[0,0],[20,0],[20,10],[0,10]] },
    { id: "b", order: 1, text: "B", polygon: [[0,20],[20,20],[20,30],[0,30]] },
  ],
};

test("text mutation rebuilds aggregate text and undo restores it", () => {
  const initial = createEditorState(annotation);
  const changed = updateText(initial, "a", "Xin");
  assert.equal(changed.annotation.text, "Xin\nB");
  assert.equal(changed.annotation.status, "edited");
  assert.equal(undo(changed).annotation.text, "A\nB");
});

test("corner movement clamps to image bounds", () => {
  const state = createEditorState(annotation);
  const changed = moveCorner(state, "a", 0, -5, 100);
  assert.deepEqual(changed.annotation.blocks[0].polygon[0], [0, 79]);
});

test("reorder writes contiguous order values", () => {
  const state = reorderBlock(createEditorState(annotation), "b", 0);
  assert.deepEqual(
    state.annotation.blocks.map((block) => [block.id, block.order]),
    [["b", 0], ["a", 1]],
  );
});
```

- [ ] **Step 2: Verify the frontend test fails**

```bash
node --test tests/frontend/state.test.mjs
```

Expected: module-not-found failure for `state.mjs`.

- [ ] **Step 3: Implement immutable state mutations**

```javascript
// ocr_labeler/static/state.mjs
const clone = (value) => structuredClone(value);

function normalized(annotation) {
  const next = clone(annotation);
  next.blocks.sort((a, b) => a.order - b.order);
  next.blocks.forEach((block, index) => { block.order = index; });
  next.text = next.blocks.map((block) => block.text).join("\n");
  return next;
}

export function createEditorState(annotation) {
  return {
    annotation: normalized(annotation),
    selectedId: annotation.blocks[0]?.id ?? null,
    undoStack: [],
    redoStack: [],
    dirty: false,
  };
}

function mutate(state, change) {
  const before = clone(state.annotation);
  const annotation = normalized(change(clone(state.annotation)));
  annotation.status = annotation.status === "completed" ? "completed" : "edited";
  return {
    ...state,
    annotation,
    undoStack: [...state.undoStack.slice(-49), before],
    redoStack: [],
    dirty: true,
  };
}

export function selectBlock(state, id) {
  return { ...state, selectedId: id };
}

export function updateText(state, id, text) {
  return mutate(state, (annotation) => {
    annotation.blocks.find((block) => block.id === id).text = text;
    return annotation;
  });
}

export function moveCorner(state, id, corner, x, y) {
  return mutate(state, (annotation) => {
    const block = annotation.blocks.find((item) => item.id === id);
    block.polygon[corner] = [
      Math.min(Math.max(x, 0), annotation.image.width - 1),
      Math.min(Math.max(y, 0), annotation.image.height - 1),
    ];
    return annotation;
  });
}

export function moveBlock(state, id, dx, dy) {
  return mutate(state, (annotation) => {
    const block = annotation.blocks.find((item) => item.id === id);
    const xs = block.polygon.map(([x]) => x);
    const ys = block.polygon.map(([, y]) => y);
    const safeDx = Math.min(
      Math.max(dx, -Math.min(...xs)),
      annotation.image.width - 1 - Math.max(...xs),
    );
    const safeDy = Math.min(
      Math.max(dy, -Math.min(...ys)),
      annotation.image.height - 1 - Math.max(...ys),
    );
    block.polygon = block.polygon.map(([x, y]) => [x + safeDx, y + safeDy]);
    return annotation;
  });
}

export function reorderBlock(state, id, targetIndex) {
  return mutate(state, (annotation) => {
    const sourceIndex = annotation.blocks.findIndex((block) => block.id === id);
    const [block] = annotation.blocks.splice(sourceIndex, 1);
    annotation.blocks.splice(targetIndex, 0, block);
    annotation.blocks.forEach((item, index) => { item.order = index; });
    return annotation;
  });
}

export function setStatus(state, status) {
  if (
    status === "completed"
    && state.annotation.blocks.some((block) => !block.text.trim())
  ) {
    throw new Error("Không thể hoàn tất khi còn block rỗng");
  }
  return mutate(state, (annotation) => {
    annotation.status = status;
    return annotation;
  });
}

export function undo(state) {
  if (state.undoStack.length === 0) return state;
  const annotation = state.undoStack.at(-1);
  return {
    ...state,
    annotation,
    undoStack: state.undoStack.slice(0, -1),
    redoStack: [...state.redoStack, clone(state.annotation)],
    dirty: true,
  };
}

export function redo(state) {
  if (state.redoStack.length === 0) return state;
  const annotation = state.redoStack.at(-1);
  return {
    ...state,
    annotation,
    undoStack: [...state.undoStack, clone(state.annotation)],
    redoStack: state.redoStack.slice(0, -1),
    dirty: true,
  };
}
```

All geometry mutations clamp x to `[0, image.width - 1]` and y to
`[0, image.height - 1]`. `undo` and `redo` preserve `selectedId`.

- [ ] **Step 4: Run frontend state tests**

```bash
node --test tests/frontend/state.test.mjs
```

Expected: all state/history tests pass.

---

### Task 7: SVG Geometry and Three-Column Static Shell

**Files:**
- Create: `ocr_labeler/static/geometry.mjs`
- Create: `ocr_labeler/static/index.html`
- Create: `ocr_labeler/static/styles.css`
- Create: `tests/frontend/geometry.test.mjs`
- Create: `tests/test_labeler_static.py`
- Modify: `ocr_labeler/app.py`

**Interfaces:**
- Produces: `screenToImage`, `imageToScreen`, `translatePolygon`, and
  `rectanglePolygon`.
- Produces stable DOM IDs used by `app.mjs`.
- FastAPI serves `/` and `/static/*`.

- [ ] **Step 1: Write failing coordinate-transform tests**

```javascript
// tests/frontend/geometry.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import {
  imageToScreen,
  rectanglePolygon,
  screenToImage,
  translatePolygon,
} from "../../ocr_labeler/static/geometry.mjs";

test("screen and image transforms are inverse", () => {
  const viewport = { scale: 0.5, offsetX: 20, offsetY: 30 };
  const screen = imageToScreen([100, 80], viewport);
  assert.deepEqual(screen, [70, 70]);
  assert.deepEqual(screenToImage(screen, viewport), [100, 80]);
});

test("translation clamps all four points", () => {
  const polygon = [[0,0],[90,0],[90,20],[0,20]];
  assert.deepEqual(
    translatePolygon(polygon, 20, 70, 100, 80),
    [[9,59],[99,59],[99,79],[9,79]],
  );
});

test("rectangle normalizes reverse drag direction", () => {
  assert.deepEqual(
    rectanglePolygon([20, 30], [5, 10]),
    [[5,10],[20,10],[20,30],[5,30]],
  );
});
```

- [ ] **Step 2: Write the failing static-contract test**

```python
# tests/test_labeler_static.py
import unittest
from pathlib import Path


class StaticContractTests(unittest.TestCase):
    def test_application_shell_has_required_regions_and_module_entry(self):
        html = Path("ocr_labeler/static/index.html").read_text(encoding="utf-8")
        for element_id in (
            "folder-path",
            "image-list",
            "page-stage",
            "overlay",
            "block-list",
            "text-editor",
            "batch-progress",
            "save-status",
        ):
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn('src="/static/app.mjs"', html)
        self.assertIn('type="module"', html)
```

- [ ] **Step 3: Verify both test groups fail**

```bash
node --test tests/frontend/geometry.test.mjs
.venv/bin/python -m unittest tests.test_labeler_static -v
```

Expected: missing geometry module and missing HTML file.

- [ ] **Step 4: Implement coordinate helpers**

```javascript
// ocr_labeler/static/geometry.mjs
export const imageToScreen = ([x, y], view) => [
  x * view.scale + view.offsetX,
  y * view.scale + view.offsetY,
];

export const screenToImage = ([x, y], view) => [
  (x - view.offsetX) / view.scale,
  (y - view.offsetY) / view.scale,
];

export function rectanglePolygon(start, end) {
  const left = Math.min(start[0], end[0]);
  const right = Math.max(start[0], end[0]);
  const top = Math.min(start[1], end[1]);
  const bottom = Math.max(start[1], end[1]);
  return [[left, top], [right, top], [right, bottom], [left, bottom]];
}

export function translatePolygon(polygon, dx, dy, width, height) {
  const xs = polygon.map(([x]) => x);
  const ys = polygon.map(([, y]) => y);
  const safeDx = Math.min(Math.max(dx, -Math.min(...xs)), width - 1 - Math.max(...xs));
  const safeDy = Math.min(Math.max(dy, -Math.min(...ys)), height - 1 - Math.max(...ys));
  return polygon.map(([x, y]) => [x + safeDx, y + safeDy]);
}
```

- [ ] **Step 5: Build the semantic shell and styling**

`index.html` must contain:

```html
<header class="topbar">
  <strong>PaddleOCR Labeler</strong>
  <input id="folder-path" aria-label="Folder ảnh" />
  <button id="open-folder">Mở folder</button>
  <progress id="batch-progress" value="0" max="1"></progress>
  <span id="model-status">Đang kiểm tra model</span>
  <button id="ocr-current">OCR ảnh này</button>
  <button id="ocr-batch">OCR toàn folder</button>
  <button id="cancel-batch">Dừng</button>
  <button id="toggle-completed">Đánh dấu hoàn tất</button>
  <button id="export-jsonl">Xuất JSONL</button>
  <span id="save-status">Chưa lưu</span>
</header>
<main class="workspace">
  <aside class="image-sidebar">
    <input id="image-search" aria-label="Tìm tên ảnh" />
    <nav id="image-filters" aria-label="Trạng thái ảnh"></nav>
    <ol id="image-list"></ol>
  </aside>
  <section class="canvas-panel">
    <div class="canvas-toolbar"></div>
    <div id="page-stage" tabindex="0">
      <img id="page-image" alt="" />
      <svg id="overlay" aria-label="Bounding boxes"></svg>
    </div>
  </section>
  <aside class="inspector">
    <ol id="block-list"></ol>
    <canvas id="crop-preview"></canvas>
    <textarea id="text-editor" aria-label="Văn bản đã nhận dạng"></textarea>
    <div id="point-editor"></div>
  </aside>
</main>
<script type="module" src="/static/app.mjs"></script>
```

CSS uses a `260px minmax(0, 1fr) 380px` desktop grid, dark canvas, sticky
topbar, high-contrast focus states, green/orange/blue polygon classes, and a
minimum supported width of 1280px.

- [ ] **Step 6: Mount static files and serve the shell**

In `create_app`, mount:

```python
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

static_dir = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(static_dir / "index.html")
```

- [ ] **Step 7: Run static and geometry tests**

```bash
node --test tests/frontend/geometry.test.mjs
.venv/bin/python -m unittest tests.test_labeler_static -v
```

Expected: both groups pass.

---

### Task 8: Browser Wiring, SVG Editing, Autosave, and Batch Controls

**Files:**
- Create: `ocr_labeler/static/app.mjs`
- Modify: `ocr_labeler/static/state.mjs`
- Modify: `ocr_labeler/static/styles.css`
- Modify: `tests/frontend/state.test.mjs`

**Interfaces:**
- Consumes every approved API route and pure state/geometry functions.
- Produces one browser controller with no global mutable model outside the
  module.
- Autosave sends the current annotation revision 500 ms after the last change.

- [ ] **Step 1: Add failing state tests for manual blocks and save acknowledgements**

```javascript
// append to tests/frontend/state.test.mjs
import {
  acknowledgeSave,
  addBlock,
  deleteBlock,
} from "../../ocr_labeler/static/state.mjs";

test("manual block creation and deletion participate in history", () => {
  const initial = createEditorState(annotation);
  const added = addBlock(initial, [[5,5],[15,5],[15,15],[5,15]]);
  const id = added.selectedId;
  assert.equal(added.annotation.blocks.at(-1).source, "manual");
  const removed = deleteBlock(added, id);
  assert.equal(removed.annotation.blocks.length, 2);
  assert.equal(undo(removed).annotation.blocks.length, 3);
});

test("save acknowledgement replaces revision without adding undo history", () => {
  const changed = updateText(createEditorState(annotation), "a", "Saved");
  const acknowledged = acknowledgeSave(changed, {
    ...changed.annotation,
    revision: 2,
  });
  assert.equal(acknowledged.annotation.revision, 2);
  assert.equal(acknowledged.dirty, false);
  assert.equal(acknowledged.undoStack.length, changed.undoStack.length);
});
```

- [ ] **Step 2: Run the state tests and verify the new exports fail**

```bash
node --test tests/frontend/state.test.mjs
```

Expected: missing `acknowledgeSave` and `addBlock` exports.

- [ ] **Step 3: Complete editor-state behavior**

Implement:

```javascript
export function acknowledgeSave(state, savedAnnotation) {
  return {
    ...state,
    annotation: normalized(savedAnnotation),
    dirty: false,
  };
}

export function addBlock(state, polygon) {
  const id = crypto.randomUUID();
  const next = mutate(state, (annotation) => {
    annotation.blocks.push({
      id,
      order: annotation.blocks.length,
      text: "",
      polygon,
      score: null,
      source: "manual",
    });
    return annotation;
  });
  return { ...next, selectedId: id };
}
```

Complete delete, selection, status, redo, and movement exports. Limit history
to 50 snapshots.

- [ ] **Step 4: Implement API and rendering primitives in `app.mjs`**

Use a single module-scoped controller:

```javascript
const controller = {
  images: [],
  currentImageId: null,
  editor: null,
  view: { scale: 1, offsetX: 0, offsetY: 0 },
  mode: "select",
  autosaveTimer: null,
  drag: null,
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers ?? {}) },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(payload.detail ?? response.statusText);
  }
  return response.status === 204 ? null : response.json();
}
```

Render lists with `document.createElement` and `textContent`; do not interpolate
OCR text into `innerHTML`.

- [ ] **Step 5: Implement SVG selection, movement, corner editing, and add mode**

Render each block as one `<g data-block-id>`, one polygon, four corner circles,
and one numbered label. Convert pointer coordinates through
`screenToImage`. On pointer-up, commit exactly one history mutation. During
pointer-move, draw a transient polygon so hundreds of mouse events do not each
create undo history.

Keyboard contract:

- `Delete`: delete selected block;
- `Ctrl+Z`: undo;
- `Ctrl+Shift+Z` or `Ctrl+Y`: redo;
- `A`: toggle add mode outside text inputs;
- `Escape`: cancel transient drag/add;
- `Space` plus drag: pan;
- wheel: zoom around the pointer.

- [ ] **Step 6: Implement inspector, crop preview, reading order, and completion**

The inspector:

- binds textarea `input` to `updateText`;
- redraws crop preview using the selected polygon's bounding rectangle;
- exposes eight numeric point inputs;
- reorders by HTML drag-and-drop and calls `reorderBlock`;
- renders confidence below 0.60 with the orange class;
- prevents completion when a block is blank and shows the server-compatible
  validation message.

- [ ] **Step 7: Implement debounced autosave and revision conflict handling**

```javascript
function scheduleAutosave() {
  clearTimeout(controller.autosaveTimer);
  setSaveStatus("Đang lưu");
  controller.autosaveTimer = setTimeout(saveCurrent, 500);
}

async function saveCurrent() {
  try {
    const saved = await api(
      `/api/images/${controller.currentImageId}/annotation`,
      {
        method: "PUT",
        body: JSON.stringify(controller.editor.annotation),
      },
    );
    controller.editor = acknowledgeSave(controller.editor, saved);
    setSaveStatus("Đã lưu");
  } catch (error) {
    setSaveStatus(
      error.message.includes("revision") ? "Xung đột" : "Lỗi lưu",
    );
  }
}
```

Changing images must await a pending save. A conflict never retries with a
blind overwrite; the UI offers reload.

- [ ] **Step 8: Implement current OCR, batch polling, cancellation, and export**

- `OCR ảnh này` confirms before sending `replace_existing: true`.
- `OCR toàn folder` posts once and polls `/api/batch` every 750 ms.
- Polling stops on `completed`, `cancelled`, or `failed`.
- `Dừng` posts cancellation and remains disabled until the state changes.
- Export displays the manifest path and record count.
- Batch progress is `(processed + skipped + failed) / total`.

- [ ] **Step 9: Run all frontend tests**

```bash
node --test tests/frontend/state.test.mjs tests/frontend/geometry.test.mjs
```

Expected: all editor state and geometry tests pass with no external Node
packages.

---

### Task 9: CLI Launcher, Documentation, and Real GPU/Browser Verification

**Files:**
- Create: `ocr_labeler/cli.py`
- Create: `run_labeler.py`
- Create: `tests/test_labeler_cli.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `parse_args(argv)`, `build_settings(args)`, and `main(argv=None)`.
- Produces one documented launch command using the existing `.venv`.

- [ ] **Step 1: Write failing CLI-default tests**

```python
# tests/test_labeler_cli.py
import unittest
from pathlib import Path

from ocr_labeler.cli import build_parser, build_settings


class LabelerCLITests(unittest.TestCase):
    def test_defaults_target_exported_model_and_single_gpu(self):
        args = build_parser().parse_args([])
        settings = build_settings(args)
        self.assertEqual(settings.device, "gpu:0")
        self.assertEqual(settings.text_rec_input_shape, (3, 48, 1600))
        self.assertEqual(
            settings.rec_model_dir,
            Path("runs/vi_rec_3datasets_v1/inference/best_accuracy"),
        )

    def test_cpu_requires_explicit_flag_value(self):
        args = build_parser().parse_args(["--device", "cpu"])
        self.assertEqual(build_settings(args).device, "cpu")
```

- [ ] **Step 2: Verify the CLI test fails**

```bash
.venv/bin/python -m unittest tests.test_labeler_cli -v
```

Expected: missing `ocr_labeler.cli`.

- [ ] **Step 3: Implement parsing and one-worker Uvicorn launch**

```python
# ocr_labeler/cli.py
import argparse
from pathlib import Path

import uvicorn

from .app import create_app
from .settings import LabelerSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PaddleOCR Vietnamese labeler")
    parser.add_argument(
        "--rec-model-dir",
        type=Path,
        default=Path("runs/vi_rec_3datasets_v1/inference/best_accuracy"),
    )
    parser.add_argument("--det-model-dir", type=Path)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--images", type=Path)
    return parser


def build_settings(args) -> LabelerSettings:
    return LabelerSettings(
        rec_model_dir=args.rec_model_dir,
        det_model_dir=args.det_model_dir,
        device=args.device,
        host=args.host,
        port=args.port,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    settings = build_settings(args).validate()
    app = create_app(settings=settings, initial_workspace=args.images)
    uvicorn.run(app, host=settings.host, port=settings.port, workers=1)
    return 0
```

```python
# run_labeler.py
from ocr_labeler.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

Update `create_app` to accept `initial_workspace: Path | None` and open it in
lifespan after engine readiness succeeds.

- [ ] **Step 4: Document installation, launch, storage, and recovery**

Add a README section with these exact commands:

```bash
source .venv/bin/activate
python -m pip install -r requirements-labeler.txt
python run_labeler.py \
  --images /home/tieubaoca/Documents/ocr-md/images \
  --device gpu:0
```

Document:

- URL `http://127.0.0.1:8010`;
- default detector and recognition model;
- output directory `.paddleocr-labeler`;
- direct-child scanning;
- batch skip/resume behavior;
- `--det-model-dir`, `--rec-model-dir`, and `--device cpu`;
- why one Uvicorn worker is required.

- [ ] **Step 5: Run every automated test**

```bash
.venv/bin/python -m unittest discover -s tests -v
node --test tests/frontend/state.test.mjs tests/frontend/geometry.test.mjs
node --check ocr_labeler/static/app.mjs
node --check ocr_labeler/static/state.mjs
node --check ocr_labeler/static/geometry.mjs
```

Expected: all Python/JavaScript tests pass and syntax checks produce no errors.

- [ ] **Step 6: Start the real service and verify model readiness**

```bash
.venv/bin/python run_labeler.py \
  --images /home/tieubaoca/Documents/ocr-md/images \
  --device gpu:0
```

In another terminal:

```bash
curl -s http://127.0.0.1:8010/api/health
nvidia-smi
```

Expected: health reports ready, PaddleOCR 3.7.0 uses `gpu:0`, and exactly one
service process owns the model allocation.

- [ ] **Step 7: Run a real current-image OCR smoke test**

Use the browser or API to OCR:

```text
/home/tieubaoca/Documents/ocr-md/images/2_14.png
```

Expected:

- PP-OCRv6 medium detector returns multiple page regions;
- fine-tuned recognition returns Vietnamese text;
- response polygons have four points inside the 1654×2339 image;
- low-confidence regions remain present;
- the service does not flatten the whole page into one recognition crop.

- [ ] **Step 8: Verify the complete browser flow**

At `http://127.0.0.1:8010`:

1. select a polygon and edit Vietnamese text;
2. drag the polygon and one corner;
3. add and delete a manual region;
4. change reading order;
5. undo and redo;
6. wait for `Đã lưu`, reload, and confirm persistence;
7. start batch, cancel after at least one image, restart it, and confirm saved
   images are skipped;
8. export JSONL and parse every line with:

```bash
.venv/bin/python -c "import json, pathlib; p=pathlib.Path('/home/tieubaoca/Documents/ocr-md/images/.paddleocr-labeler/manifest.jsonl'); rows=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]; print(len(rows), 'valid rows')"
```

Expected: the UI and saved data meet all ten acceptance criteria in the design
specification.

- [ ] **Step 9: Record final evidence**

Record:

- Python and Node test totals;
- health payload;
- real OCR duration and block count;
- browser flow result;
- manifest path and valid row count;
- GPU memory usage during idle and OCR.

Do not claim a commit or push because the workspace has no Git repository.

---

## Plan Self-Review Checklist

- [x] Every design-spec section maps to at least one task.
- [x] All interfaces consumed by later tasks are produced by an earlier task.
- [x] No task requires parallel GPU inference or a second model instance.
- [x] Runtime has no Node dependency.
- [x] All disk mutations are confined to `.paddleocr-labeler`.
- [x] Current-image OCR cannot overwrite non-empty annotations without explicit
  confirmation.
- [x] Revision conflict behavior is covered in storage and API tests.
- [x] Source-image replacement prevents unsafe save/export.
- [x] Automated verification is followed by real GPU and browser verification.

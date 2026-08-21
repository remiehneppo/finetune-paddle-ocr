from __future__ import annotations

import hashlib
import json
import tempfile
from threading import Event
import time
import unittest
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image

from vl_layout_labeler.app import create_app
from vl_layout_labeler.batch import BatchManager, GPUCoordinator
from vl_layout_labeler.catalog import WorkspaceCatalog
from pydantic import ValidationError

from vl_layout_labeler.geometry import (
    clamp_polygon,
    crop_box_from_polygon,
    polygon_area,
    polygon_to_xywh,
)
from vl_layout_labeler.layout_engine import normalize_layout_result
from vl_layout_labeler.models import Annotation, Block, ImageInfo
from vl_layout_labeler.settings import LabelerSettings
from vl_layout_labeler.storage import (
    AnnotationStore,
    ExportError,
    RevisionConflict,
    split_layout_pages,
)
from vl_layout_labeler.task_map import (
    PP_DOCLAYOUTV3_LABELS,
    SKIP_LAYOUT_LABELS,
    map_layout_label,
)
from vl_layout_labeler.vl_client import build_chat_payload


def image_info(path: str = "page.png") -> ImageInfo:
    return ImageInfo(path=path, width=20, height=10, sha256="a" * 64)


class VLLayoutLabelerTests(unittest.TestCase):
    def test_task_mapping_and_skip_list(self):
        self.assertEqual(len(PP_DOCLAYOUTV3_LABELS), 25)
        self.assertEqual(PP_DOCLAYOUTV3_LABELS[0], "abstract")
        self.assertEqual(PP_DOCLAYOUTV3_LABELS[-1], "vision_footnote")
        self.assertEqual(map_layout_label("table"), "table")
        self.assertEqual(map_layout_label("display_formula"), "formula")
        self.assertEqual(map_layout_label("footer"), "ocr")
        self.assertIsNone(map_layout_label("image"))
        self.assertIsNone(map_layout_label("unknown-class"))
        self.assertIn("seal", SKIP_LAYOUT_LABELS)

    def test_layout_normalization_accepts_polygon_and_xyxy(self):
        record = Mock(relative_path="page.png", width=20, height=10, sha256="a" * 64)
        annotation = normalize_layout_result(
            {
                "boxes": [
                    {"label": "table", "coordinate": [[1, 2], [10, 2], [10, 8], [1, 8]], "score": 0.8},
                    {"label": "image", "coordinate": [0, 0, 5, 5], "score": 0.9},
                    {"label": "text", "coordinate": [2, 1, 8, 4], "score": 0.7},
                ]
            },
            record,
        )
        self.assertEqual(len(annotation.blocks), 3)
        self.assertEqual(
            [block.task for block in annotation.blocks], ["table", None, "ocr"]
        )
        self.assertEqual(annotation.blocks[2].polygon[2], (8.0, 4.0))

    def test_sidecar_v1_migrates_in_memory_to_v2(self):
        annotation = Annotation.model_validate(
            {
                "version": 1,
                "image": image_info().model_dump(),
                "blocks": [
                    {
                        "order": 0,
                        "polygon": [(1, 1), (10, 1), (10, 8), (1, 8)],
                        "layout_label": "text",
                        "task": "ocr",
                    }
                ],
            }
        )
        self.assertEqual(annotation.version, 2)
        empty_completed = Annotation.model_validate(
            {
                "version": 1,
                "image": image_info().model_dump(),
                "status": "completed",
                "blocks": [],
            }
        )
        self.assertEqual(empty_completed.status, "edited")

    def test_completion_allows_layout_only_but_validates_active_blocks(self):
        layout_only = Block(
            order=0,
            polygon=[(1, 1), (10, 1), (10, 8), (1, 8)],
            layout_label="image",
            task=None,
        )
        completed = Annotation(
            image=image_info(), status="completed", blocks=[layout_only]
        )
        self.assertEqual(completed.status, "completed")
        with self.assertRaises(ValidationError):
            Annotation(
                image=image_info(),
                status="completed",
                blocks=[layout_only.model_copy(update={"task": "ocr"})],
            )
        with self.assertRaises(ValidationError):
            Annotation(
                image=image_info(),
                status="completed",
                blocks=[layout_only.model_copy(update={"layout_label": "manual"})],
            )
        with self.assertRaises(ValidationError):
            Annotation(
                image=image_info(),
                status="completed",
                blocks=[layout_only.model_copy(update={"polygon": [(1, 1)] * 4})],
            )
        with self.assertRaisesRegex(ValidationError, "OTSL|HTML"):
            Annotation(
                image=image_info(),
                status="completed",
                blocks=[
                    layout_only.model_copy(
                        update={
                            "layout_label": "table",
                            "task": "table",
                            "text": "<table><tr><td>A</td></tr></table>",
                        }
                    )
                ],
            )

    def test_polygon_helpers_clamp_bbox_and_area(self):
        polygon = clamp_polygon([(-2, -1), (21, -1), (21, 8), (-2, 8)], 20, 10)
        self.assertEqual(polygon, [(0.0, 0.0), (19.0, 0.0), (19.0, 8.0), (0.0, 8.0)])
        self.assertEqual(polygon_to_xywh(polygon), [0.0, 0.0, 19.0, 8.0])
        self.assertEqual(polygon_area(polygon), 152.0)

    def test_layout_split_is_deterministic_and_non_empty(self):
        pages = [{"record": Mock(image_id=str(index))} for index in range(11)]
        first = split_layout_pages(pages)
        second = split_layout_pages(pages)
        self.assertEqual(
            [[page["record"].image_id for page in split] for split in first],
            [[page["record"].image_id for page in split] for split in second],
        )
        self.assertEqual([len(split) for split in first], [10, 1])
        with self.assertRaises(ExportError):
            split_layout_pages(pages[:1])

    def test_layout_engine_passes_model_name_with_local_model_dir(self):
        settings = LabelerSettings().validate(require_runtime_models=False)
        constructor = Mock(return_value="pipeline")
        paddleocr_module = Mock(LayoutDetection=constructor)
        with patch.dict(
            "sys.modules",
            {"paddleocr": paddleocr_module},
        ):
            engine = __import__(
                "vl_layout_labeler.layout_engine", fromlist=["LayoutDetectionEngine"]
            ).LayoutDetectionEngine.create(settings)
        self.assertEqual(engine.pipeline, "pipeline")
        constructor.assert_called_once_with(
            model_name="PP-DocLayoutV3",
            model_dir=str(settings.layout_model_dir.expanduser()),
            device=settings.device,
        )

    def test_crop_bbox_is_clamped(self):
        self.assertEqual(crop_box_from_polygon([(-5, -2), (25, -2), (25, 12), (-5, 12)], 20, 10), (0, 0, 20, 10))

    def test_payload_is_image_first_and_deterministic(self):
        payload = build_chat_payload("paddleocr-vl", "data:image/png;base64,abc", "formula", 123)
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[1]["text"], "Formula Recognition:")
        self.assertEqual(payload["temperature"], 0)

    def test_store_revision_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            Image.new("RGB", (20, 10), "white").save(image_path)
            catalog = WorkspaceCatalog.open(root)
            record = catalog.list_images()[0]
            store = AnnotationStore(root)
            annotation = store.load(record)
            block = Block(order=0, polygon=[(1, 1), (10, 1), (10, 8), (1, 8)], layout_label="text", task="ocr", text="hello")
            saved = store.save(record, annotation.model_copy(update={"blocks": [block], "status": "edited"}))
            self.assertEqual(saved.revision, 1)
            with self.assertRaises(RevisionConflict):
                store.save(record, annotation.model_copy(update={"blocks": [block], "status": "edited"}))

    @unittest.skipUnless(__import__("importlib").util.find_spec("datasets"), "datasets not installed")
    def test_export_has_hf_schema_and_only_completed_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 10), "white").save(root / "page1.png")
            Image.new("RGB", (20, 10), "gray").save(root / "page2.png")
            catalog = WorkspaceCatalog.open(root)
            store = AnnotationStore(root)
            tasks = (
                ("page1.png", "text", "ocr", "hello"),
                ("page2.png", "table", "table", "<fcel>A<nl>"),
            )
            for name, layout_label, task, text in tasks:
                record = next(
                    item for item in catalog.list_images() if item.name == name
                )
                draft = store.load(record)
                block = Block(
                    order=0,
                    polygon=[(1, 1), (8, 1), (8, 8), (1, 8)],
                    layout_label=layout_label,
                    task=task,
                    text=text,
                )
                layout_only = Block(
                    order=1,
                    polygon=[(9, 1), (15, 1), (15, 8), (9, 8)],
                    layout_label="image",
                    task=None,
                    text="ignored",
                )
                saved = store.save(
                    record,
                    draft.model_copy(
                        update={"blocks": [block, layout_only], "status": "edited"}
                    ),
                )
                store.save(
                    record,
                    Annotation.model_validate(
                        {**saved.model_dump(mode="python"), "status": "completed"}
                    ),
                )
            result = store.export_hf(catalog, root / "export")
            self.assertEqual(result["samples"], 2)
            self.assertEqual(result["train_pages"], 1)
            self.assertEqual(result["validation_pages"], 1)
            from datasets import load_from_disk

            dataset = load_from_disk(root / "export")
            self.assertEqual(set(dataset), {"train", "validation"})
            page_ids = {
                split: set(dataset[split]["source_page_id"]) for split in dataset
            }
            self.assertFalse(page_ids["train"].intersection(page_ids["validation"]))
            rows = [dataset[split][0] for split in ("train", "validation")]
            self.assertEqual({row["task"] for row in rows}, {"ocr", "table"})
            self.assertEqual(
                dataset["train"].column_names,
                ["image", "text", "task", "source_page_id"],
            )
            self.assertTrue(list((root / "export" / "crops").glob("*.png")))

    def test_hf_export_rejects_html_table_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("page1.png", "page2.png"):
                Image.new("RGB", (20, 10), "white").save(root / name)
            catalog = WorkspaceCatalog.open(root)
            store = AnnotationStore(root)
            records = catalog.list_images()
            draft = store.load(records[0])
            invalid_legacy = draft.model_copy(
                update={
                    "status": "completed",
                    "blocks": [
                        Block(
                            order=0,
                            polygon=[(1, 1), (8, 1), (8, 8), (1, 8)],
                            layout_label="table",
                            task="table",
                            text="<table><tr><td>A</td></tr></table>",
                        )
                    ],
                }
            )
            with (
                patch.object(
                    store,
                    "_load_saved",
                    side_effect=lambda record: invalid_legacy
                    if record.image_id == records[0].image_id
                    else None,
                ),
                self.assertRaisesRegex(ExportError, "OTSL|HTML"),
            ):
                store.export_hf(catalog, root / "export")

    def test_layout_export_has_coco_taxonomy_read_order_and_immutable_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, color in (("page1.png", "white"), ("page2.png", "gray")):
                Image.new("RGB", (20, 10), color).save(root / name)
            catalog = WorkspaceCatalog.open(root)
            store = AnnotationStore(root)
            original_hashes = {
                record.name: hashlib.sha256(record.path.read_bytes()).hexdigest()
                for record in catalog.list_images()
            }
            for record in catalog.list_images():
                draft = store.load(record)
                blocks = [
                    Block(
                        order=0,
                        polygon=[(-2, -1), (10, -1), (10, 8), (-2, 8)],
                        layout_label="text",
                        task="ocr",
                        text="hello",
                    ),
                    Block(
                        order=1,
                        polygon=[(10, 1), (15, 1), (15, 8), (10, 8)],
                        layout_label="table",
                        task="table",
                        text="ignored",
                        skipped=True,
                    ),
                    Block(
                        order=2,
                        polygon=[(15, 1), (19, 1), (19, 8), (15, 8)],
                        layout_label="image",
                        task=None,
                    ),
                ]
                saved = store.save(
                    record,
                    draft.model_copy(update={"blocks": blocks, "status": "edited"}),
                )
                store.save(
                    record,
                    Annotation.model_validate(
                        {**saved.model_dump(mode="python"), "status": "completed"}
                    ),
                )
            result = store.export_layout(catalog, root / "layout-export")
            self.assertEqual(result["pages"], 2)
            payloads = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((root / "layout-export" / "annotations").glob("instance_*.json"))
            ]
            self.assertEqual(sum(len(payload["images"]) for payload in payloads), 2)
            self.assertTrue(all(len(payload["categories"]) == 25 for payload in payloads))
            annotations = [item for payload in payloads for item in payload["annotations"]]
            self.assertEqual([item["read_order"] for item in annotations], [0, 1, 0, 1])
            self.assertEqual(annotations[0]["segmentation"][0][:4], [0.0, 0.0, 10.0, 0.0])
            self.assertGreater(annotations[0]["area"], 0)
            for name, digest in original_hashes.items():
                exported = root / "layout-export" / "images" / name
                self.assertEqual(hashlib.sha256(exported.read_bytes()).hexdigest(), digest)

    def test_export_all_rolls_back_when_one_branch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 10), "white").save(root / "page.png")
            store = AnnotationStore(root)
            catalog = WorkspaceCatalog.open(root)

            def fake_hf(_catalog, output):
                output.mkdir()
                (output / "partial").write_text("data", encoding="utf-8")
                return {"path": str(output), "samples": 2}

            with patch.object(store, "export_hf", side_effect=fake_hf), patch.object(
                store, "export_layout", side_effect=ExportError("layout failed")
            ):
                with self.assertRaises(ExportError):
                    store.export_all(catalog, root / "all-export")
            self.assertFalse((root / "all-export").exists())

    def test_batch_cancel(self):
        started = Event()

        class SlowLayout:
            def detect(self, record):
                started.set()
                time.sleep(0.05)
                return Mock(revision=0)

        coordinator = GPUCoordinator(SlowLayout(), Mock())
        manager = BatchManager(coordinator)
        catalog = Mock()
        catalog.list_images.return_value = [Mock(name="one.png", error=None), Mock(name="two.png", error=None)]
        store = Mock()
        store.load.return_value = Mock(blocks=[], revision=0)
        snapshot = manager.start("detect", catalog, store)
        self.assertEqual(snapshot.operation, "detect")
        self.assertTrue(started.wait(1))
        manager.cancel()
        for _ in range(100):
            if manager.snapshot().state == "cancelled":
                break
            time.sleep(0.005)
        self.assertEqual(manager.snapshot().state, "cancelled")
        self.assertEqual(store.save.call_count, 1)

    def test_api_detect_prelabel_complete_and_revision_conflict(self):
        class FakeLayout:
            def detect(self, record):
                return Annotation(
                    image=ImageInfo(
                        path=record.relative_path,
                        width=record.width,
                        height=record.height,
                        sha256=record.sha256,
                    ),
                    status="detected",
                    blocks=[
                        Block(
                            order=0,
                            polygon=[(1, 1), (10, 1), (10, 8), (1, 8)],
                            layout_label="text",
                            task="ocr",
                        ),
                        Block(
                            order=1,
                            polygon=[(11, 1), (19, 1), (19, 8), (11, 8)],
                            layout_label="image",
                            task=None,
                        ),
                    ],
                )

        class FakeVL:
            def prelabel(self, *args):
                return "hello"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 10), "white").save(root / "page.png")
            app = create_app(
                LabelerSettings().validate(require_runtime_models=False),
                layout_engine=FakeLayout(),
                vl_client=FakeVL(),
            )
            with TestClient(app) as client:
                self.assertEqual(
                    client.post("/api/workspace/open", json={"path": str(root)}).status_code,
                    200,
                )
                image_id = client.get("/api/images").json()["images"][0]["image_id"]
                detected = client.post(
                    f"/api/images/{image_id}/detect",
                    json={"replace_existing": True},
                )
                self.assertEqual(detected.status_code, 200)
                stale = detected.json()
                self.assertTrue(
                    {"/api/export", "/api/export/hf", "/api/export/layout", "/api/export/all"}
                    <= {route.path for route in app.routes}
                )
                incomplete = client.post(f"/api/images/{image_id}/complete")
                self.assertEqual(incomplete.status_code, 422)
                layout_only = detected.json()["blocks"][1]["id"]
                rejected = client.post(
                    f"/api/images/{image_id}/prelabel",
                    json={"block_ids": [layout_only], "replace_existing": True},
                )
                self.assertEqual(rejected.status_code, 422)
                prelabelled = client.post(
                    f"/api/images/{image_id}/prelabel",
                    json={"block_ids": None, "replace_existing": True},
                )
                self.assertEqual(prelabelled.json()["blocks"][0]["text"], "hello")
                self.assertEqual(prelabelled.json()["blocks"][1]["text"], "")
                conflict = client.put(
                    f"/api/images/{image_id}/annotation", json=stale
                )
                self.assertEqual(conflict.status_code, 409)
                completed = client.post(f"/api/images/{image_id}/complete")
                self.assertEqual(completed.status_code, 200)
                self.assertEqual(completed.json()["status"], "completed")
                edited_payload = completed.json()
                edited_payload["blocks"][0]["text"] = ""
                edited = client.put(
                    f"/api/images/{image_id}/annotation", json=edited_payload
                )
                self.assertEqual(edited.status_code, 200)
                self.assertEqual(edited.json()["status"], "edited")

    def test_static_ui_exposes_layout_and_atomic_exports(self):
        html = Path("vl_layout_labeler/static/index.html").read_text(encoding="utf-8")
        script = Path("vl_layout_labeler/static/app.mjs").read_text(encoding="utf-8")
        for element_id in (
            "layout-label",
            "target-editor",
            "visual-tab",
            "raw-tab",
            "visual-editor",
            "text",
            "export-hf",
            "export-layout",
            "export-all",
        ):
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn("Không xuất VL", html)
        self.assertIn('from "./target_codec.mjs"', script)
        self.assertIn("inspectTarget(block.task, block.text)", script)
        self.assertIn("holder.rowSpan = cell.rowspan", script)
        self.assertIn("holder.colSpan = cell.colspan", script)
        self.assertIn('targetButton("Gộp →"', script)
        self.assertIn('targetButton("Tách"', script)
        self.assertIn('block.task || "layout-only"', script)
        self.assertIn('exportDataset("/api/export/all"', script)

    def test_all_export_endpoints_dispatch_to_the_expected_exporter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 10), "white").save(root / "page.png")
            app = create_app(
                LabelerSettings().validate(require_runtime_models=False),
                layout_engine=Mock(),
                vl_client=Mock(),
            )
            with patch.object(
                AnnotationStore,
                "export_hf",
                return_value={"path": "/tmp/hf", "samples": 2},
            ) as export_hf, patch.object(
                AnnotationStore,
                "export_layout",
                return_value={"path": "/tmp/layout", "pages": 2, "annotations": 2},
            ) as export_layout, patch.object(
                AnnotationStore,
                "export_all",
                return_value={"path": "/tmp/all", "vl": {}, "layout": {}},
            ) as export_all:
                with TestClient(app) as client:
                    client.post("/api/workspace/open", json={"path": str(root)})
                    self.assertEqual(
                        client.post(
                            "/api/export", json={"output_dir": "/tmp/legacy"}
                        ).status_code,
                        200,
                    )
                    self.assertEqual(
                        client.post(
                            "/api/export/hf", json={"output_dir": "/tmp/hf"}
                        ).status_code,
                        200,
                    )
                    self.assertEqual(
                        client.post(
                            "/api/export/layout", json={"output_dir": "/tmp/layout"}
                        ).status_code,
                        200,
                    )
                    self.assertEqual(
                        client.post(
                            "/api/export/all", json={"output_dir": "/tmp/all"}
                        ).status_code,
                        200,
                    )
            self.assertEqual(export_hf.call_count, 2)
            export_layout.assert_called_once()
            export_all.assert_called_once()


if __name__ == "__main__":
    unittest.main()

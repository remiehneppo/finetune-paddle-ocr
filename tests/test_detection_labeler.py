import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from ocr_labeler.app import create_app
from ocr_labeler.catalog import WorkspaceCatalog
from ocr_labeler.detection_cli import build_settings, main, parse_args
from ocr_labeler.models import Block
from ocr_labeler.ocr_engine import PaddleOCRDetectionEngine
from ocr_labeler.settings import LabelerSettings
from ocr_labeler.storage import AnnotationStore


class FakeDetectionResult(dict):
    pass


class FakeDetectionPipeline:
    def __init__(self):
        self.closed = False

    def predict(self, path):
        self.path = path
        return [
            FakeDetectionResult(
                dt_polys=[
                    [[1, 2], [40, 2], [40, 12], [1, 12]],
                    [[2, 15], [55, 15], [55, 25], [2, 25]],
                ],
                dt_scores=[0.98, 0.42],
            )
        ]

    def close(self):
        self.closed = True


class FakeDetectionEngine:
    def recognize(self, record):
        pipeline = FakeDetectionPipeline()
        return PaddleOCRDetectionEngine(
            settings=LabelerSettings(task="detection"),
            pipeline=pipeline,
        ).recognize(record)


class DetectionEngineTests(unittest.TestCase):
    def test_detection_result_becomes_editable_placeholder_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (60, 30), "white").save(root / "page.png")
            record = WorkspaceCatalog.open(root).list_images()[0]
            pipeline = FakeDetectionPipeline()
            engine = PaddleOCRDetectionEngine(
                settings=LabelerSettings(task="detection"),
                pipeline=pipeline,
            )

            annotation = engine.recognize(record)
            engine.close()

        self.assertEqual(annotation.status, "ocr")
        self.assertEqual(annotation.ocr.task, "detection")
        self.assertIsNone(annotation.ocr.rec_model)
        self.assertEqual([block.text for block in annotation.blocks], ["text", "text"])
        self.assertEqual([block.score for block in annotation.blocks], [0.98, 0.42])
        self.assertTrue(pipeline.closed)

    def test_detection_mode_does_not_require_recognition_model_files(self):
        settings = LabelerSettings(
            task="detection",
            rec_model_dir=Path("/definitely/missing/recognition"),
            device="cpu",
        )
        self.assertIs(settings.validate(), settings)
        self.assertEqual(settings.data_dir_name, ".paddleocr-det-labeler")

    def test_detection_normalizer_skips_bad_polygons_and_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (60, 30), "white").save(root / "page.png")
            record = WorkspaceCatalog.open(root).list_images()[0]
            pipeline = FakeDetectionPipeline()
            pipeline.predict = lambda path: [
                FakeDetectionResult(
                    dt_polys=[
                        [[1, 2], [40, 2], [40, 12], [1, 12]],
                        [[1, 2], [40, 2], [1, 12]],
                        [[1, 2], [40, 2], [40, 12], [1, 12]],
                    ],
                    dt_scores=[0.8, 0.7, float("nan")],
                )
            ]
            annotation = PaddleOCRDetectionEngine(
                settings=LabelerSettings(task="detection"),
                pipeline=pipeline,
            ).recognize(record)

        self.assertEqual(len(annotation.blocks), 1)
        self.assertEqual(annotation.blocks[0].score, 0.8)


class DetectionStorageTests(unittest.TestCase):
    def test_export_matches_paddleocr_detection_format_and_is_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (64, 32), "white").save(root / "page.png")
            catalog = WorkspaceCatalog.open(root)
            record = catalog.list_images()[0]
            store = AnnotationStore(root, ".paddleocr-det-labeler")
            annotation = store.create_draft(record)
            annotation.blocks = [
                Block(
                    order=0,
                    text="text",
                    polygon=[(1.2, 2.2), (50.4, 2), (50, 20.8), (1, 20)],
                    score=None,
                    source="manual",
                ),
                Block(
                    order=1,
                    text="###",
                    polygon=[(2, 22), (20, 22), (20, 30), (2, 30)],
                    score=None,
                    source="manual",
                ),
            ]
            store.save(record, annotation)

            labels_path = store.export_detection_labels(catalog)
            image_path, payload = labels_path.read_text(encoding="utf-8").strip().split("\t", 1)
            labels = json.loads(payload)

        self.assertEqual(image_path, "page.png")
        self.assertEqual(labels[0]["transcription"], "text")
        self.assertEqual(labels[0]["points"][0], [1, 2])
        self.assertEqual(labels[1]["transcription"], "###")
        self.assertIn(".paddleocr-det-labeler", str(labels_path))
        self.assertFalse((root / ".paddleocr-labeler").exists())


class DetectionAPITests(unittest.TestCase):
    def test_detect_save_and_export_use_detection_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (60, 30), "white").save(root / "page.png")
            settings = LabelerSettings(task="detection", device="cpu")
            with TestClient(
                create_app(
                    settings=settings,
                    engine=FakeDetectionEngine(),
                    initial_workspace=root,
                )
            ) as client:
                health = client.get("/api/health").json()
                image_id = client.get("/api/images").json()["images"][0]["image_id"]
                detected = client.post(
                    f"/api/images/{image_id}/detect",
                    json={"replace_existing": False},
                )
                saved = client.put(
                    f"/api/images/{image_id}/annotation",
                    json=detected.json(),
                )
                exported = client.post("/api/export")

            self.assertEqual(health["task"], "detection")
            self.assertEqual(health["workspace"], str(root.resolve()))
            self.assertIsNone(health["rec_model"])
            self.assertEqual(detected.status_code, 200)
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(exported.json()["format"], "paddleocr_detection")
            self.assertTrue(Path(exported.json()["path"]).is_file())
            self.assertIn(".paddleocr-det-labeler", exported.json()["path"])


class DetectionCLITests(unittest.TestCase):
    def test_detection_cli_defaults_and_model_tuning(self):
        settings = build_settings(parse_args([]))
        self.assertEqual(settings.task, "detection")
        self.assertEqual(settings.det_model_name, "PP-OCRv6_medium_det")
        self.assertEqual(settings.text_det_limit_side_len, 1600)
        self.assertEqual(settings.port, 8011)

        tuned = build_settings(
            parse_args(
                [
                    "--device",
                    "cpu",
                    "--det-limit-side-len",
                    "1280",
                    "--det-box-thresh",
                    "0.5",
                ]
            )
        )
        self.assertEqual(tuned.text_det_limit_side_len, 1280)
        self.assertEqual(tuned.text_det_box_thresh, 0.5)

    def test_main_forces_single_worker(self):
        app = object()
        with (
            patch("ocr_labeler.detection_cli.create_app", return_value=app) as create_app_mock,
            patch("ocr_labeler.detection_cli.uvicorn.run") as run,
        ):
            result = main(["--device", "cpu"])

        self.assertEqual(result, 0)
        settings = create_app_mock.call_args.kwargs["settings"]
        self.assertEqual(settings.task, "detection")
        run.assert_called_once_with(
            app,
            host="127.0.0.1",
            port=8011,
            workers=1,
        )


if __name__ == "__main__":
    unittest.main()

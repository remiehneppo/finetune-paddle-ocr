import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dataset_admission import RejectionReport, inspect_image_file, open_image_value
from labeler_catalog import WorkspaceCatalog
from ocr_labeler.catalog import WorkspaceCatalog as OCRWorkspaceCatalog
from vl_layout_labeler.catalog import WorkspaceCatalog as VLWorkspaceCatalog


class SharedAdmissionRuntimeTests(unittest.TestCase):
    def test_rejection_report_preserves_recognition_and_detection_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rejected.jsonl"
            source = type("Source", (), {"root": Path("pages"), "labels": Path("labels.txt")})()
            with RejectionReport(path) as report:
                report.reject(Path("ocr"), "train", 3, "invalid_image", "bad")
                report.add(source, 7, "invalid_box", "outside", box_index=2)

            rows = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(rows[0], {
            "dataset": "ocr",
            "split": "train",
            "row_index": 3,
            "reason": "invalid_image",
            "detail": "bad",
        })
        self.assertEqual(rows[1], {
            "dataset": "pages",
            "label_file": "labels.txt",
            "line_number": 7,
            "box_index": 2,
            "reason": "invalid_box",
            "detail": "outside",
        })
        self.assertEqual(report.counts, {"invalid_box": 1, "invalid_image": 1})

    def test_catalog_facades_share_one_implementation(self):
        self.assertIs(OCRWorkspaceCatalog, WorkspaceCatalog)
        self.assertIs(VLWorkspaceCatalog, WorkspaceCatalog)

    def test_open_image_value_accepts_pil_bytes_and_array_with_same_limits(self):
        image = Image.new("L", (3, 2), 128)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        for value in (image, {"bytes": buffer.getvalue()}, image.copy()):
            with self.subTest(value_type=type(value).__name__):
                admitted = open_image_value(value, Path("."), 6, convert_rgb=True)
                self.assertEqual(admitted.mode, "RGB")
                self.assertEqual(admitted.size, (3, 2))

        with self.assertRaisesRegex(ValueError, "limit"):
            open_image_value(image, Path("."), 5)

    def test_inspect_image_file_returns_dimensions_bytes_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            Image.new("RGB", (4, 3), "white").save(path)
            width, height, data, digest = inspect_image_file(path, 12)
            self.assertEqual((width, height), (4, 3))
            self.assertEqual(len(data), path.stat().st_size)
            self.assertEqual(len(digest), 64)

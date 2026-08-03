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
        self.assertFalse(pipeline.kwargs["use_doc_orientation_classify"])
        self.assertFalse(pipeline.kwargs["use_doc_unwarping"])
        self.assertFalse(pipeline.kwargs["use_textline_orientation"])
        self.assertEqual(pipeline.kwargs["text_rec_score_thresh"], 0.0)

    def test_normalization_filters_blank_text_and_invalid_polygons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (60, 30), "white").save(root / "page.png")
            record = WorkspaceCatalog.open(root).list_images()[0]
            pipeline = FakePipeline()
            pipeline.predict = lambda path, **kwargs: [
                FakeResult(
                    rec_texts=["Giữ lại", "  ", "Ba điểm", "Không hữu hạn"],
                    rec_scores=[0.9, 0.8, 0.7, 0.6],
                    rec_polys=[
                        [[1, 2], [40, 2], [40, 12], [1, 12]],
                        [[1, 14], [40, 14], [40, 24], [1, 24]],
                        [[1, 2], [40, 2], [1, 12]],
                        [[1, 2], [float("inf"), 2], [40, 12], [1, 12]],
                    ],
                )
            ]

            annotation = PaddleOCREngine(
                settings=LabelerSettings(rec_model_dir=Path("runs/model")),
                pipeline=pipeline,
            ).recognize(record)

        self.assertEqual(annotation.text, "Giữ lại")
        self.assertEqual([block.order for block in annotation.blocks], [0])
        self.assertEqual([block.text for block in annotation.blocks], ["Giữ lại"])

    def test_normalization_skips_malformed_scores_without_losing_neighbors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (60, 30), "white").save(root / "page.png")
            record = WorkspaceCatalog.open(root).list_images()[0]
            pipeline = FakePipeline()
            pipeline.predict = lambda path, **kwargs: [
                FakeResult(
                    rec_texts=[
                        "Trước",
                        "Không số",
                        "NaN",
                        "Vô hạn",
                        "Âm",
                        "Quá một",
                        "Sau",
                    ],
                    rec_scores=[0.9, "không hợp lệ", float("nan"), float("inf"), -0.1, 1.1, 0.4],
                    rec_polys=[
                        [[1, y], [40, y], [40, y + 8], [1, y + 8]]
                        for y in (1, 3, 5, 7, 9, 11, 13)
                    ],
                )
            ]

            annotation = PaddleOCREngine(
                settings=LabelerSettings(rec_model_dir=Path("runs/model")),
                pipeline=pipeline,
            ).recognize(record)

        self.assertEqual([block.text for block in annotation.blocks], ["Trước", "Sau"])
        self.assertEqual([block.score for block in annotation.blocks], [0.9, 0.4])
        self.assertEqual([block.order for block in annotation.blocks], [0, 1])

    def test_settings_use_quality_defaults(self):
        settings = LabelerSettings(rec_model_dir=Path("runs/model"))

        self.assertEqual(settings.det_model_name, "PP-OCRv6_medium_det")
        self.assertEqual(settings.text_rec_input_shape, (3, 48, 1600))
        self.assertEqual(settings.device, "gpu:0")
        self.assertEqual(settings.text_rec_score_thresh, 0.0)
        self.assertEqual(settings.confidence_warning_threshold, 0.60)


if __name__ == "__main__":
    unittest.main()

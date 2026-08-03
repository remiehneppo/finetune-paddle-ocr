import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

import finetune_det


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PADDLEOCR_DIR = PROJECT_ROOT / "PaddleOCR"
SOURCE_CONFIG = PADDLEOCR_DIR / "configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml"


class PaddleDetectionLoaderSmokeTests(unittest.TestCase):
    def test_prepared_validation_sample_runs_through_native_paddle_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            rows = []
            for index, color in enumerate(("white", "gray")):
                filename = f"page-{index}.png"
                Image.new("RGB", (128, 96), color).save(dataset / filename)
                payload = [
                    {
                        "transcription": "text",
                        "points": [[8, 10], [110, 10], [110, 45], [8, 45]],
                    }
                ]
                rows.append(f"{filename}\t{json.dumps(payload)}\n")
            metadata = dataset / ".paddleocr-det-labeler"
            metadata.mkdir()
            (metadata / "det_labels.txt").write_text("".join(rows), encoding="utf-8")

            args = finetune_det.parse_args(
                [
                    "--dataset-dir",
                    str(dataset),
                    "--paddleocr-dir",
                    str(PADDLEOCR_DIR),
                    "--prepare-only",
                ]
            )
            work = root / "run"
            work.mkdir()
            source = finetune_det.resolve_dataset_source(dataset, 0)
            finetune_det.prepare_datasets([source], args, work)
            config_path = work / "resolved.yml"
            finetune_det.create_resolved_config(
                SOURCE_CONFIG,
                config_path,
                work / "prepared",
                work / "output",
                args,
            )
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

            sys.path.insert(0, str(PADDLEOCR_DIR))
            try:
                from ppocr.data.simple_dataset import SimpleDataSet

                paddle_dataset = SimpleDataSet(
                    config, "Eval", logging.getLogger("det-loader-test"), seed=2026
                )
                sample = paddle_dataset[0]
            finally:
                if sys.path and sys.path[0] == str(PADDLEOCR_DIR):
                    sys.path.pop(0)

        self.assertIsNotNone(sample)
        self.assertEqual(len(sample), 4)
        image, shape, polys, ignore_tags = sample
        self.assertEqual(image.shape[0], 3)
        self.assertEqual(shape.shape[0], 4)
        self.assertEqual(polys.shape[-1], 2)
        self.assertFalse(ignore_tags[0])


if __name__ == "__main__":
    unittest.main()

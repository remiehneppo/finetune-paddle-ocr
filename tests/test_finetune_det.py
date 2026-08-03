import json
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

import finetune_det


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PADDLEOCR_DIR = PROJECT_ROOT / "PaddleOCR"
SOURCE_CONFIG = PADDLEOCR_DIR / "configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml"


def write_image(path: Path, color: str, size=(100, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def write_export(root: Path, rows: list[tuple[str, object]]) -> Path:
    output = root / ".paddleocr-det-labeler" / "det_labels.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            f"{image_path}\t{json.dumps(payload, ensure_ascii=False)}\n"
            for image_path, payload in rows
        ),
        encoding="utf-8",
    )
    return output


def valid_box(text="text"):
    return {
        "transcription": text,
        "points": [[2, 3], [80, 3], [80, 30], [2, 30]],
    }


def args_for(*dataset_dirs: Path):
    return finetune_det.parse_args(
        [
            "--dataset-dir",
            *(str(path) for path in dataset_dirs),
            "--paddleocr-dir",
            str(PADDLEOCR_DIR),
            "--prepare-only",
        ]
    )


class DetectionFinetuneSourceTests(unittest.TestCase):
    def test_accepts_workspace_metadata_directory_and_label_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = write_export(root, [])
            sources = [
                finetune_det.resolve_dataset_source(root, 0),
                finetune_det.resolve_dataset_source(labels.parent, 1),
                finetune_det.resolve_dataset_source(labels, 2),
            ]
            expected_root = root.resolve()
            expected_labels = labels.resolve()

        self.assertEqual([source.root for source in sources], [expected_root] * 3)
        self.assertEqual([source.labels for source in sources], [expected_labels] * 3)

    def test_rejects_inference_directory_as_pretrained_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "training .pdparams"):
                finetune_det.download_pretrained(directory, Path(directory))


class DetectionFinetunePreparationTests(unittest.TestCase):
    def test_filters_bad_samples_and_writes_native_detection_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            dataset = base / "dataset"
            write_image(dataset / "one.png", "white")
            write_image(dataset / "two.png", "gray")
            write_image(dataset / "bad_box.png", "blue")
            (dataset / "broken.png").write_bytes(b"not an image")
            write_export(
                dataset,
                [
                    ("one.png", [valid_box(), valid_box("###")]),
                    ("two.png", [valid_box("")]),
                    (
                        "bad_box.png",
                        [{"transcription": "text", "points": [[2, 2], [80, 30], [2, 30], [80, 2]]}],
                    ),
                    ("broken.png", [valid_box()]),
                    ("missing.png", [valid_box()]),
                    ("one.png", []),
                ],
            )
            args = args_for(dataset)
            source = finetune_det.resolve_dataset_source(dataset, 0)
            work = base / "run"
            work.mkdir()

            summary = finetune_det.prepare_datasets([source], args, work)
            prepared_rows = []
            for filename in ("train.txt", "validation.txt"):
                prepared_rows.extend(
                    (work / "prepared" / filename).read_text(encoding="utf-8").splitlines()
                )

            self.assertEqual(summary["unique_images"], 2)
            self.assertEqual(summary["train_samples"], 1)
            self.assertEqual(summary["validation_samples"], 1)
            self.assertGreaterEqual(summary["rejection_counts"]["invalid_box"], 1)
            self.assertGreaterEqual(summary["rejection_counts"]["invalid_image"], 1)
            self.assertGreaterEqual(summary["rejection_counts"]["missing_image"], 1)
            payloads = [json.loads(row.split("\t", 1)[1]) for row in prepared_rows]
            labels = [label for payload in payloads for label in payload]
            self.assertIn("###", [label["transcription"] for label in labels])
            self.assertTrue(all(label["transcription"] in {"text", "###"} for label in labels))
            self.assertTrue(
                all((work / "prepared" / row.split("\t", 1)[0]).is_file() for row in prepared_rows)
            )

    def test_mixes_datasets_and_prevents_hash_leakage_between_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first = base / "first"
            second = base / "second"
            write_image(first / "a.png", "red")
            write_image(first / "b.png", "green")
            write_image(second / "c.png", "blue")
            write_image(second / "duplicate.png", "red")
            write_export(first, [("a.png", [valid_box()]), ("b.png", [valid_box()])])
            write_export(second, [("c.png", [valid_box()]), ("duplicate.png", [valid_box()])])
            args = args_for(first, second)
            sources = [
                finetune_det.resolve_dataset_source(first, 0),
                finetune_det.resolve_dataset_source(second, 1),
            ]
            work = base / "run"
            work.mkdir()

            summary = finetune_det.prepare_datasets(sources, args, work)
            train_paths = {
                row.split("\t", 1)[0]
                for row in (work / "prepared/train.txt").read_text().splitlines()
            }
            validation_paths = {
                row.split("\t", 1)[0]
                for row in (work / "prepared/validation.txt").read_text().splitlines()
            }

        self.assertEqual(summary["unique_images"], 3)
        self.assertEqual(summary["rejection_counts"]["duplicate_image"], 1)
        self.assertFalse(train_paths & validation_paths)


class DetectionFinetuneConfigTests(unittest.TestCase):
    def test_resolved_config_preserves_architecture_loss_and_transform_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            prepared.mkdir()
            target = root / "resolved.yml"
            args = finetune_det.parse_args(
                [
                    "--dataset-dir", str(root),
                    "--paddleocr-dir", str(PADDLEOCR_DIR),
                    "--epochs", "25",
                    "--batch-size", "3",
                    "--learning-rate", "0.00005",
                ]
            )

            finetune_det.create_resolved_config(
                SOURCE_CONFIG, target, prepared, root / "output", args
            )
            source = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
            resolved = yaml.safe_load(target.read_text(encoding="utf-8"))

        self.assertEqual(resolved["Architecture"], source["Architecture"])
        self.assertEqual(resolved["Loss"], source["Loss"])
        self.assertEqual(
            finetune_det._transform_names(resolved["Train"]["dataset"]["transforms"]),
            finetune_det._transform_names(source["Train"]["dataset"]["transforms"]),
        )
        self.assertEqual(resolved["Global"]["model_name"], "PP-OCRv6_medium_det")
        self.assertEqual(resolved["Global"]["epoch_num"], 25)
        self.assertEqual(resolved["Train"]["loader"]["batch_size_per_card"], 3)
        self.assertEqual(resolved["Optimizer"]["lr"]["learning_rate"], 0.00005)
        maps = {
            name: value
            for transform in resolved["Train"]["dataset"]["transforms"]
            for name, value in transform.items()
            if name in {"MakeBorderMap", "MakeShrinkMap"}
        }
        self.assertEqual(maps["MakeBorderMap"]["total_epoch"], 25)
        self.assertEqual(maps["MakeShrinkMap"]["total_epoch"], 25)


if __name__ == "__main__":
    unittest.main()

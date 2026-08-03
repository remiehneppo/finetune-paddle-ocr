import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

import finetune


class FakeSplit:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else ["image", "text"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class FinetuneTests(unittest.TestCase):
    def test_load_hf_dataset_reads_hub_style_parquet_snapshot(self):
        from datasets import Dataset, Features, Image as HFImage, Value

        buffer = io.BytesIO()
        Image.new("RGB", (20, 10), "white").save(buffer, format="PNG")
        features = Features({"image": HFImage(), "text": Value("string")})
        dataset = Dataset.from_dict(
            {
                "image": [{"bytes": buffer.getvalue(), "path": "sample.png"}],
                "text": ["Việt Nam"],
            },
            features=features,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            dataset.to_parquet(data_dir / "train-00000-of-00001.parquet")
            loaded = finetune.load_hf_dataset(root)

            self.assertIn("train", loaded)
            self.assertEqual(loaded["train"].column_names, ["image", "text"])
            self.assertEqual(loaded["train"][0]["text"], "Việt Nam")
            self.assertIsInstance(loaded["train"][0]["image"]["bytes"], bytes)

    def test_normalize_text_uses_text_when_label_is_empty(self):
        row = {"label": "  ", "text": "  Tiếng\tViệt\n "}
        self.assertEqual(finetune.normalize_text(row), "Tiếng Việt")

    def test_bundled_dictionary_covers_vietnamese_nfc(self):
        characters = finetune.load_character_set(
            Path(finetune.__file__).with_name("vietnamese_dict.txt")
        )
        alphabet = (
            "aăâbcdđeêghiklmnoôơpqrstuưvxy"
            "AĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXY"
            "áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
            "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
            "ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
            "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
        )
        self.assertEqual(finetune.unsupported_characters(alphabet, characters), [])

    def test_split_is_deterministic_and_keeps_training_data(self):
        samples = [finetune.PreparedSample(str(i), str(i), 0) for i in range(10)]
        first = finetune.split_train_validation(samples, 0.2, 7)
        second = finetune.split_train_validation(samples, 0.2, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 8)
        self.assertEqual(len(first[1]), 2)

    def test_prepare_datasets_mixes_multiple_sources(self):
        def rows(prefix):
            return FakeSplit(
                [
                    {
                        "image": Image.new("RGB", (20, 10), "white"),
                        "text": f"{prefix}{index}",
                    }
                    for index in range(3)
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dirs = [root / "source_a", root / "source_b"]
            for path in dataset_dirs:
                path.mkdir()
            args = argparse.Namespace(
                dataset_dir=dataset_dirs,
                max_text_length=80,
                max_image_pixels=10000,
                validation_ratio=0.34,
                seed=7,
            )
            (root / "run").mkdir()
            with patch.object(
                finetune, "load_hf_dataset", side_effect=[rows("a"), rows("b")]
            ):
                summary = finetune.prepare_datasets(
                    args,
                    root / "run",
                    Path(finetune.__file__).with_name("vietnamese_dict.txt"),
                )

            train_lines = (root / "run" / "prepared" / "train.txt").read_text()
            validation_lines = (
                root / "run" / "prepared" / "validation.txt"
            ).read_text()

        self.assertEqual(summary["train_samples"], 4)
        self.assertEqual(summary["validation_samples"], 2)
        self.assertIn("dataset_000", train_lines + validation_lines)
        self.assertIn("dataset_001", train_lines + validation_lines)

    def test_process_split_filters_empty_text_and_invalid_image(self):
        rows = [
            {"image": Image.new("RGB", (20, 10), "white"), "text": "Việt Nam"},
            {"image": Image.new("RGB", (20, 10), "white"), "text": "  "},
            {"image": {"bytes": b"not an image"}, "label": "lỗi"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with finetune.RejectionReport(root / "rejected.jsonl") as report:
                result = finetune.process_split(
                    FakeSplit(rows),
                    root,
                    0,
                    "train",
                    root,
                    set("Việt Namlỗi "),
                    80,
                    10000,
                    report,
                )
                counts = report.counts.copy()

            self.assertEqual(len(result), 1)
            self.assertTrue((root / result[0].image_path).is_file())
            self.assertEqual(counts["empty_text"], 1)
            self.assertEqual(counts["invalid_image"], 1)
            records = [
                json.loads(line)
                for line in (root / "rejected.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 2)

    def test_create_resolved_config_updates_shape_and_training_paths(self):
        base = {
            "Global": {},
            "Optimizer": {"lr": {}},
            "Architecture": {"Head": {"head_list": [{"NRTRHead": {}}]}},
            "Train": {
                "dataset": {"transforms": [{"RecConAug": {}}]},
                "sampler": {},
                "loader": {},
            },
            "Eval": {
                "dataset": {"transforms": [{"RecResizeImg": {}}]},
                "loader": {},
            },
        }
        args = argparse.Namespace(
            epochs=20,
            learning_rate=0.0001,
            max_text_length=60,
            seed=1,
            image_width=512,
            batch_size=24,
            num_workers=6,
            pretrained_model="weights.pdparams",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.yml"
            target = root / "target.yml"
            source.write_text(yaml.safe_dump(base), encoding="utf-8")
            finetune.create_resolved_config(
                source, target, root / "prepared", root / "output", root / "dict.txt", args
            )
            result = yaml.safe_load(target.read_text(encoding="utf-8"))

        self.assertEqual(result["Global"]["max_text_length"], 60)
        self.assertEqual(result["Train"]["sampler"]["scales"][1], [512, 48])
        self.assertEqual(result["Train"]["sampler"]["first_bs"], 24)
        self.assertEqual(
            result["Eval"]["dataset"]["transforms"][0]["RecResizeImg"]["image_shape"],
            [3, 48, 512],
        )


if __name__ == "__main__":
    unittest.main()

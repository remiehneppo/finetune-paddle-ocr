import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import finetune_doclayout_v3 as finetune
from PIL import Image
from vl_layout_labeler.task_map import PP_DOCLAYOUTV3_LABELS


def make_export(root: Path, *, bad: str | None = None) -> Path:
    dataset = root / "layout"
    (dataset / "images").mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(dataset / "images/page-0.png")
    Image.new("RGB", (64, 64), "gray").save(dataset / "images/page-1.png")
    categories = [{"id": index, "name": name} for index, name in enumerate(PP_DOCLAYOUTV3_LABELS, 1)]

    def split(image_id: int, name: str, start: int) -> dict:
        annotations = []
        for offset, category_id in enumerate(range(1, 26)):
            annotations.append(
                {
                    "id": offset + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]],
                    "bbox": [10, 10, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                    "read_order": offset,
                }
            )
        return {"images": [{"id": image_id, "file_name": name, "width": 64, "height": 64}], "annotations": annotations, "categories": categories}

    train = split(1, "page-0.png", 0)
    validation = split(2, "page-1.png", 25)
    if bad == "taxonomy":
        train["categories"][0]["name"] = "wrong"
    if bad == "order":
        train["annotations"][1]["read_order"] = 99
    if bad == "leakage":
        validation["images"][0] = train["images"][0].copy()
        for annotation in validation["annotations"]:
            annotation["image_id"] = 1
    if bad == "bbox":
        train["annotations"][0]["bbox"] = [10, 10, 0, 10]
    (dataset / "annotations").mkdir()
    (dataset / "annotations/instance_train.json").write_text(json.dumps(train), encoding="utf-8")
    (dataset / "annotations/instance_val.json").write_text(json.dumps(validation), encoding="utf-8")
    (dataset / "export_manifest.json").write_text(json.dumps({"format": "COCOInstSegDataset", "categories": 25}), encoding="utf-8")
    return dataset


class DocLayoutValidationTests(unittest.TestCase):
    def test_valid_export_preserves_25_class_counts_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            result = finetune.validate_dataset(make_export(Path(directory)))
        self.assertEqual(result["classes"], 25)
        self.assertEqual(result["splits"]["train"]["annotations"], 25)
        self.assertEqual(result["per_class"]["validation"]["table"], 1)
        self.assertEqual(len(result["fingerprint"]), 64)

    def test_rejects_taxonomy_order_geometry_read_order_and_leakage(self):
        for bad, message in (("taxonomy", "taxonomy"), ("order", "read_order"), ("bbox", "bbox"), ("leakage", "leakage")):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(finetune.ContractError, message):
                    finetune.validate_dataset(make_export(Path(directory), bad=bad))


class DocLayoutCliTests(unittest.TestCase):
    def test_defaults_and_boundaries(self):
        args = finetune.parse_args(["--dataset-dir", "layout"])
        self.assertEqual(args.device, "gpu:0")
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.num_workers, 4)
        for option, value in (("--batch-size", "0"), ("--epochs", "0"), ("--learning-rate", "0")):
            bad = finetune.parse_args(["--dataset-dir", "layout", option, value])
            with self.assertRaises(finetune.ContractError):
                finetune.validate_args(bad)

    def test_rejects_resume_for_check_only_and_non_single_gpu(self):
        args = finetune.parse_args(["--dataset-dir", "layout", "--mode", "check-only", "--resume-from", "x.pdparams"])
        with self.assertRaisesRegex(finetune.ContractError, "check-only"):
            finetune.validate_args(args)
        args = finetune.parse_args(["--dataset-dir", "layout", "--device", "gpu:0,1"])
        with self.assertRaisesRegex(finetune.ContractError, "gpu:0"):
            finetune.validate_args(args)

    def test_check_only_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_export(root)
            work = root / "run"
            args = finetune.parse_args([
                "--dataset-dir", str(dataset), "--work-dir", str(work),
                "--mode", "check-only", "--skip-backend-check",
            ])
            with patch.object(finetune, "run_engine") as engine:
                self.assertEqual(finetune.run(args), 0)
            engine.assert_called_once()
            self.assertEqual(engine.call_args.args[1], "check_dataset")
            self.assertTrue((work / "resolved-config.yaml").is_file())
            summary = json.loads((work / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "passed")


if __name__ == "__main__":
    unittest.main()

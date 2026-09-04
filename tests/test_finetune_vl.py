import argparse
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image

import evaluate_paddleocr_vl
import finetune_vl
import merge_paddleocr_vl_lora
from prepared_run_planning import PreparedRunPlan, PreparedRunPlanner


class FakeSplit:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else ["image", "text"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        extra = 2 if add_special_tokens else 0
        return list(range(len(text) + extra))


def png_bytes(size=(20, 10), mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, size, 127).save(buffer, format="PNG")
    return buffer.getvalue()


class FinetuneVLTests(unittest.TestCase):
    def test_prepared_from_cli_contract(self):
        prepared = Path("/data/prepared-run")
        args = finetune_vl.parse_args(["--prepared-from", str(prepared)])
        self.assertIsNone(args.dataset_dir)
        self.assertEqual(args.prepared_from, prepared)

        invalid_argv = (
            [],
            ["--dataset-dir", "/data/source", "--prepared-from", str(prepared)],
            ["--prepared-from", str(prepared), "--prepare-only"],
            [
                "--prepared-from",
                str(prepared),
                "--resume-from",
                "/runs/train/adapter/checkpoint-1",
            ],
        )
        for argv in invalid_argv:
            with self.subTest(argv=argv):
                parsed = finetune_vl.parse_args(argv)
                with self.assertRaisesRegex(ValueError, "prepared|dataset|resume"):
                    finetune_vl.validate_args(parsed)

    def _write_prepared_fixture(
        self,
        root,
        name,
        task="ocr",
        train_count=1,
        validation_count=1,
        model="PaddlePaddle/PaddleOCR-VL-1.6",
    ):
        prepared_run = root / name
        prepared = prepared_run / "prepared"
        images = prepared / "images" / "source-000"
        images.mkdir(parents=True)
        train_samples = []
        validation_samples = []
        prompt = finetune_vl.prompt_for_task(task)
        if task == "ocr":
            targets = [f"{name}-train-{index}" for index in range(train_count)]
            validation_targets = [
                f"{name}-validation-{index}" for index in range(validation_count)
            ]
        elif task == "table":
            targets = ["<fcel>header<ecel><nl>"] * train_count
            validation_targets = ["<fcel>header<ecel><nl>"] * validation_count
        else:
            raise AssertionError(f"unsupported fixture task: {task}")
        for split, count, samples, target_values in (
            ("train", train_count, train_samples, targets),
            ("validation", validation_count, validation_samples, validation_targets),
        ):
            for index in range(count):
                filename = f"{split}-{index}.png"
                (images / filename).write_bytes(png_bytes())
                samples.append(
                    finetune_vl.PreparedSample(
                        f"images/source-000/{filename}", target_values[index], 0, prompt
                    )
                )
        train_jsonl = prepared / "train-source-000.jsonl"
        validation_jsonl = prepared / "validation-source-000.jsonl"
        finetune_vl.write_erniekit_jsonl(train_jsonl, train_samples)
        finetune_vl.write_erniekit_jsonl(validation_jsonl, validation_samples)
        summary = {
            "task": task,
            "prompt": prompt,
            "model": model,
            "sources": [
                {
                    "dataset": f"/data/{name}",
                    "train_samples": train_count,
                    "validation_samples": validation_count,
                    "train_jsonl": "prepared/train-source-000.jsonl",
                    "validation_jsonl": "prepared/validation-source-000.jsonl",
                }
            ],
            "train_samples": train_count,
            "validation_samples": validation_count,
            "train_probabilities": [1.0],
            "validation_probabilities": [1.0],
            "rejected": {},
        }
        (prepared_run / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return prepared_run

    def test_prepared_from_supports_single_and_multiple_paths_and_weights(self):
        single = finetune_vl.parse_args(["--prepared-from", "/one"])
        self.assertIsInstance(single.prepared_from, Path)
        self.assertIsNone(single.prepared_weight)
        multiple = finetune_vl.parse_args(
            ["--prepared-from", "/one", "/two", "--prepared-weight", "95", "5"]
        )
        self.assertEqual(multiple.prepared_from, [Path("/one"), Path("/two")])
        self.assertEqual(multiple.prepared_weight, [95.0, 5.0])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [root / "one", root / "two"]
            for run in runs:
                run.mkdir()
            args = finetune_vl.parse_args(
                [
                    "--prepared-from",
                    *(str(run) for run in runs),
                    "--prepared-weight",
                    "95",
                    "5",
                ]
            )
            with (
                patch.object(finetune_vl, "validate_erniekit_source"),
                patch.object(
                    finetune_vl,
                    "require_local_model_snapshot",
                    return_value=Path("/models/PaddleOCR-VL-1.6"),
                ),
            ):
                args.erniekit_dir = root / "erniekit"
                finetune_vl.validate_args(args)
            self.assertEqual(args.prepared_weight, [0.95, 0.05])

    def test_prepared_weight_validation_rejects_missing_mismatch_and_nonpositive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "one"
            second = root / "two"
            first.mkdir()
            second.mkdir()
            cases = (
                ([str(first), str(second)], None, "require one"),
                ([str(first), str(second)], [1.0], "one value"),
                ([str(first), str(second)], [0.0, 1.0], "positive"),
                ([str(first), str(second)], [-1.0, 1.0], "positive"),
                ([str(first), str(second)], [float("nan"), 1.0], "positive"),
                ([str(first), str(second)], [float("inf"), 1.0], "positive"),
            )
            for paths, weights, message in cases:
                args = finetune_vl.parse_args(["--prepared-from", *paths])
                args.prepared_weight = weights
                with self.subTest(weights=weights):
                    with self.assertRaisesRegex(ValueError, message):
                        finetune_vl.validate_args(args)

    def test_single_prepared_weight_defaults_and_normalizes_to_one(self):
        prepared = Path("/one")
        self.assertEqual(
            finetune_vl.normalize_prepared_weights([prepared], None), [1.0]
        )
        self.assertEqual(
            finetune_vl.normalize_prepared_weights([prepared], [95.0]), [1.0]
        )

    def test_prepared_run_plan_is_immutable_and_round_trips_metadata(self):
        source = {
            "dataset": "/data/one",
            "train_samples": 1,
            "tags": ["trusted"],
        }
        plan = PreparedRunPlan.from_summary(
            {
                "model": "model",
                "prepared_from": "/runs/one",
                "tasks": ["ocr"],
                "prompts": [finetune_vl.prompt_for_task("ocr")],
                "sources": [source],
                "source_runs": ["/runs/one"],
                "train_samples": 1,
                "validation_samples": 1,
                "train_probabilities": [1.0],
                "validation_probabilities": [1.0],
                "prepared_from_runs": ["/runs/one"],
                "prepared_weights": [1.0],
                "rejected": {},
                "custom_metadata": {"producer": "legacy"},
            }
        )

        source["dataset"] = "/data/mutated"
        source["tags"].append("mutated")
        self.assertEqual(plan.sources[0]["dataset"], "/data/one")
        self.assertEqual(plan.sources[0]["tags"], ("trusted",))
        with self.assertRaises((AttributeError, TypeError)):
            plan.tasks = ("table",)
        self.assertEqual(plan.to_summary()["prepared_from"], "/runs/one")
        self.assertEqual(
            plan.to_summary()["custom_metadata"], {"producer": "legacy"}
        )

    def test_prepared_run_planner_keeps_single_run_compatibility(self):
        prepared = Path("/runs/one")
        planner = PreparedRunPlanner(
            read_run=lambda path: {
                "model": "model",
                "prepared_from": str(path),
                "tasks": ["ocr"],
                "prompts": [finetune_vl.prompt_for_task("ocr")],
                "sources": [{"dataset": "/data/one"}],
                "train_samples": 1,
                "validation_samples": 1,
                "train_probabilities": [1.0],
                "validation_probabilities": [1.0],
            },
            normalize_weights=finetune_vl.normalize_prepared_weights,
            prompt_for_task=finetune_vl.prompt_for_task,
        )

        summary = planner.plan([prepared], None).to_summary()
        self.assertEqual(summary["prepared_from"], str(prepared))
        self.assertEqual(summary["prepared_from_runs"], [str(prepared)])
        self.assertEqual(summary["source_runs"], [str(prepared)])

    def test_aggregate_prepared_runs_scales_probabilities_and_unions_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ocr = self._write_prepared_fixture(root, "ocr", train_count=2)
            table = self._write_prepared_fixture(
                root, "table", task="table", train_count=1
            )
            work_dir = root / "aggregate"
            summary = finetune_vl.aggregate_prepared_runs(
                [ocr, table], [95, 5], work_dir
            )

            self.assertEqual(summary["train_samples"], 3)
            self.assertEqual(summary["validation_samples"], 2)
            self.assertEqual(summary["sources"][0]["dataset"], "/data/ocr")
            self.assertEqual(summary["sources"][1]["dataset"], "/data/table")
            self.assertEqual(summary["train_probabilities"], [0.95, 0.05])
            self.assertEqual(summary["validation_probabilities"], [0.95, 0.05])
            self.assertEqual(summary["tasks"], ["ocr", "table"])
            self.assertEqual(summary["prepared_from_runs"], [str(ocr.resolve()), str(table.resolve())])
            self.assertEqual(summary["prepared_weights"], [0.95, 0.05])
            self.assertEqual(summary["prepared_weight_policy"], "relative_normalized")
            self.assertFalse((work_dir / "prepared").exists())
            self.assertEqual(
                yaml.safe_load((work_dir / "summary.json").read_text()), summary
            )

    def test_aggregate_preserves_internal_source_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._write_prepared_fixture(root, "first")
            second = self._write_prepared_fixture(root, "second")
            first_summary_path = first / "summary.json"
            first_summary = json.loads(first_summary_path.read_text())
            first_source = first_summary["sources"][0]
            duplicate = dict(first_source)
            duplicate["dataset"] = "/data/first-secondary"
            first_summary["sources"].append(duplicate)
            first_summary["train_samples"] = 2
            first_summary["validation_samples"] = 2
            first_summary["train_probabilities"] = [0.75, 0.25]
            first_summary["validation_probabilities"] = [0.6, 0.4]
            first_summary_path.write_text(json.dumps(first_summary))

            summary = finetune_vl.load_prepared_runs(
                [first, second], [3, 1], root / "aggregate"
            )

            self.assertEqual(summary["prepared_weights"], [0.75, 0.25])
            self.assertEqual(
                summary["train_probabilities"], [0.5625, 0.1875, 0.25]
            )
            self.assertEqual(len(summary["validation_probabilities"]), 3)
            for actual, expected in zip(
                summary["validation_probabilities"], [0.45, 0.3, 0.25], strict=True
            ):
                self.assertAlmostEqual(actual, expected)

    def test_aggregate_resolved_config_flattens_all_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._write_prepared_fixture(root, "first")
            second = self._write_prepared_fixture(root, "second")
            summary = finetune_vl.load_prepared_runs(
                [first, second], [95, 5], root / "aggregate"
            )
            args = argparse.Namespace(
                model="PaddlePaddle/PaddleOCR-VL-1.6",
                epochs=1,
                learning_rate=1e-4,
                lora_rank=32,
                min_pixels=50_176,
                max_pixels=451_584,
                max_seq_len=2048,
                gradient_accumulation_steps=16,
                num_workers=2,
                prefetch_factor=2,
                seed=2026,
                flash_attention=True,
                devices="0",
                save_steps=1,
                smoke_steps=None,
                resume_from=None,
            )
            config = finetune_vl.create_resolved_config(
                root / "aggregate" / "resolved.yaml",
                root / "aggregate",
                summary,
                args,
            )

            self.assertEqual(
                config["train_dataset_path"],
                ",".join(source["train_jsonl"] for source in summary["sources"]),
            )
            self.assertEqual(config["train_dataset_prob"], "0.95,0.05")
            self.assertEqual(config["eval_dataset_prob"], "0.95,0.05")

    def test_prepared_jsonl_rejects_invalid_prompt_and_target_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_run = self._write_prepared_fixture(root, "table", task="table")
            train_jsonl = prepared_run / "prepared" / "train-source-000.jsonl"
            payload = json.loads(train_jsonl.read_text())
            payload["text_info"][0]["text"] = "OCR:"
            train_jsonl.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "task mask contract"):
                finetune_vl.read_prepared_run(prepared_run)

            payload["text_info"][0]["text"] = "Table Recognition:"
            payload["text_info"][1]["text"] = "<table><tr><td>x</td></tr></table>"
            train_jsonl.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "target schema"):
                finetune_vl.read_prepared_run(prepared_run)

    def test_aggregate_prepared_runs_rejects_different_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._write_prepared_fixture(root, "one")
            second = self._write_prepared_fixture(root, "two", model="other-model")
            with self.assertRaisesRegex(ValueError, "different base models"):
                finetune_vl.aggregate_prepared_runs(
                    [first, second], [0.5, 0.5], root / "aggregate"
                )

    def test_load_prepared_run_validates_jsonl_and_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_run = root / "prepared-run"
            prepared = prepared_run / "prepared"
            images = prepared / "images" / "source-000"
            images.mkdir(parents=True)
            for name in ("train-0.png", "train-1.png", "validation-0.png"):
                (images / name).write_bytes(png_bytes())

            train_jsonl = prepared / "train-source-000.jsonl"
            validation_jsonl = prepared / "validation-source-000.jsonl"
            finetune_vl.write_erniekit_jsonl(
                train_jsonl,
                [
                    finetune_vl.PreparedSample(
                        "images/source-000/train-0.png", "một", 0
                    ),
                    finetune_vl.PreparedSample(
                        "images/source-000/train-1.png", "hai", 0
                    ),
                ],
            )
            finetune_vl.write_erniekit_jsonl(
                validation_jsonl,
                [
                    finetune_vl.PreparedSample(
                        "images/source-000/validation-0.png", "ba", 0
                    )
                ],
            )
            summary = {
                "prompt": "OCR:",
                "sources": [
                    {
                        "dataset": "/data/source",
                        "train_samples": 2,
                        "validation_samples": 1,
                        "train_jsonl": "prepared/train-source-000.jsonl",
                        "validation_jsonl": "prepared/validation-source-000.jsonl",
                    }
                ],
                "train_samples": 2,
                "validation_samples": 1,
                "train_probabilities": [1.0],
                "validation_probabilities": [1.0],
                "rejected": {},
            }
            (prepared_run / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            work_dir = root / "new-run"
            work_dir.mkdir()

            loaded = finetune_vl.load_prepared_run(prepared_run, work_dir)

            self.assertEqual(loaded["train_samples"], 2)
            self.assertEqual(loaded["validation_samples"], 1)
            self.assertEqual(loaded["prepared_from"], str(prepared_run.resolve()))
            self.assertEqual(
                loaded["sources"][0]["train_jsonl"], str(train_jsonl.resolve())
            )
            self.assertEqual(
                json.loads((work_dir / "summary.json").read_text()), loaded
            )
            self.assertFalse((work_dir / "prepared").exists())

            (images / "train-1.png").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "train-1.png"):
                finetune_vl.load_prepared_run(prepared_run, root / "missing-image-run")

            (images / "train-1.png").write_bytes(png_bytes())
            summary["sources"][0]["train_samples"] = 3
            (prepared_run / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "train_samples"):
                finetune_vl.load_prepared_run(prepared_run, root / "bad-count-run")

    def test_main_reuses_prepared_run_without_preparing_or_copying(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_run = root / "prepared-run"
            prepared = prepared_run / "prepared"
            images = prepared / "images" / "source-000"
            images.mkdir(parents=True)
            (images / "train.png").write_bytes(png_bytes())
            (images / "validation.png").write_bytes(png_bytes())
            train_jsonl = prepared / "train-source-000.jsonl"
            validation_jsonl = prepared / "validation-source-000.jsonl"
            finetune_vl.write_erniekit_jsonl(
                train_jsonl,
                [finetune_vl.PreparedSample("images/source-000/train.png", "train", 0)],
            )
            finetune_vl.write_erniekit_jsonl(
                validation_jsonl,
                [
                    finetune_vl.PreparedSample(
                        "images/source-000/validation.png", "validation", 0
                    )
                ],
            )
            (prepared_run / "summary.json").write_text(
                json.dumps(
                    {
                        "prompt": "OCR:",
                        "sources": [
                            {
                                "dataset": "/data/source",
                                "train_samples": 1,
                                "validation_samples": 1,
                                "train_jsonl": str(train_jsonl),
                                "validation_jsonl": str(validation_jsonl),
                            }
                        ],
                        "train_samples": 1,
                        "validation_samples": 1,
                        "train_probabilities": [1.0],
                        "validation_probabilities": [1.0],
                        "rejected": {},
                    }
                ),
                encoding="utf-8",
            )

            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "preprocessor_config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"weights")
            (model / "tokenizer.model").write_bytes(b"tokenizer")
            erniekit = root / "erniekit"
            workflow = erniekit / "erniekit" / "train" / "ocr_vl_sft" / "workflow.py"
            peft = erniekit / "ernie" / "utils" / "peft_utils.py"
            workflow.parent.mkdir(parents=True)
            peft.parent.mkdir(parents=True)
            workflow.write_text("freeze_vision initialize_lora_model")
            peft.write_text("mark_only_lora_as_trainable")
            work_dir = root / "new-run"

            with (
                patch.object(
                    finetune_vl,
                    "load_tokenizer",
                    side_effect=AssertionError("tokenizer must not load"),
                ),
                patch.object(
                    finetune_vl,
                    "prepare_datasets",
                    side_effect=AssertionError("dataset must not be prepared"),
                ),
                patch.object(finetune_vl, "inspect_model") as inspect,
            ):
                result = finetune_vl.main(
                    [
                        "--prepared-from",
                        str(prepared_run),
                        "--erniekit-dir",
                        str(erniekit),
                        "--model",
                        str(model),
                        "--work-dir",
                        str(work_dir),
                        "--inspect-model",
                    ]
                )

            config = yaml.safe_load((work_dir / "resolved.yaml").read_text())
            prepared_was_created = (work_dir / "prepared").exists()

        self.assertEqual(result, 0)
        self.assertEqual(config["train_dataset_path"], str(train_jsonl.resolve()))
        self.assertEqual(config["eval_dataset_path"], str(validation_jsonl.resolve()))
        self.assertFalse(prepared_was_created)
        inspect.assert_called_once()

    def test_expected_merged_weight_uses_hf_transpose_contract(self):
        base = np.arange(6, dtype=np.float32).reshape(3, 2)
        lora_a = np.array([[1.0], [2.0]], dtype=np.float32)
        lora_b = np.array([[3.0, 4.0, 5.0]], dtype=np.float32)

        merged = merge_paddleocr_vl_lora.expected_merged_weight(
            base, lora_a, lora_b, scaling=2.0
        )

        np.testing.assert_array_equal(merged, base + (lora_a @ lora_b * 2.0).T)

    def test_expected_serialized_weight_matches_bfloat16_merge_rounding(self):
        import ml_dtypes

        dtype = ml_dtypes.bfloat16
        base = np.array([[-2.0]], dtype=dtype)
        lora_a = np.array([[-1.9765625]], dtype=dtype)
        lora_b = np.array([[-1.1484375]], dtype=dtype)

        merged = merge_paddleocr_vl_lora.expected_serialized_merged_weight(
            base, lora_a, lora_b, scaling=2.0, output_dtype=dtype
        )

        self.assertEqual(merged.dtype, dtype)
        self.assertEqual(float(merged[0, 0]), 2.53125)

    def test_compare_logits_reports_argmax_and_error(self):
        report = merge_paddleocr_vl_lora.compare_logits(
            np.array([[0.0, 2.0]], dtype=np.float32),
            np.array([[0.25, 2.0]], dtype=np.float32),
        )

        self.assertTrue(report["argmax_equal"])
        self.assertEqual(report["max_abs_error"], 0.25)

    def test_logit_tolerance_rejects_drift_even_when_argmax_is_equal(self):
        acceptable = {
            "max_abs_error": 0.4,
            "mean_abs_error": 0.05,
            "argmax_equal": True,
        }
        merge_paddleocr_vl_lora.validate_logits_comparison(
            acceptable, max_abs_error=0.5, mean_abs_error=0.1
        )
        within_numeric_tolerance = dict(acceptable, argmax_equal=False)
        merge_paddleocr_vl_lora.validate_logits_comparison(
            within_numeric_tolerance, max_abs_error=0.5, mean_abs_error=0.1
        )
        with self.assertRaisesRegex(RuntimeError, "tolerance"):
            merge_paddleocr_vl_lora.validate_logits_comparison(
                acceptable, max_abs_error=0.1, mean_abs_error=0.1
            )

    def test_select_best_checkpoint_uses_cer_then_exact_match(self):
        selected = finetune_vl.select_best_checkpoint(
            [
                {"checkpoint": "checkpoint-100", "cer": 0.12, "exact_match": 0.7},
                {"checkpoint": "checkpoint-200", "cer": 0.08, "exact_match": 0.6},
                {"checkpoint": "checkpoint-300", "cer": 0.08, "exact_match": 0.8},
            ]
        )
        self.assertEqual(selected["checkpoint"], "checkpoint-300")

    def test_evaluator_builds_native_ocr_message(self):
        self.assertEqual(
            evaluate_paddleocr_vl.ocr_messages("line.png"),
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "line.png"},
                        {"type": "text", "text": "OCR:"},
                    ],
                }
            ],
        )

    def test_evaluator_uses_deterministic_native_generation(self):
        self.assertEqual(
            evaluate_paddleocr_vl.deterministic_generation_kwargs(64),
            {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 64,
                "use_cache": True,
            },
        )

    def test_evaluator_decode_new_tokens_removes_prompt_prefix(self):
        class FakeProcessor:
            def batch_decode(self, token_ids, **kwargs):
                self.token_ids = token_ids.tolist()
                self.kwargs = kwargs
                return ["  xin chào  "]

        processor = FakeProcessor()
        decoded = evaluate_paddleocr_vl.decode_new_tokens(
            processor, np.array([[10, 11, 20, 21]]), 2
        )

        self.assertEqual(decoded, "  xin chào  ")
        self.assertEqual(processor.token_ids, [[20, 21]])
        self.assertTrue(processor.kwargs["skip_special_tokens"])

    def test_evaluator_candidate_coverage_requires_every_fixture_once(self):
        predictions = [
            {"candidate": candidate, "dataset": "a", "image": image}
            for candidate in ("base", "merged")
            for image in ("one.png", "two.png")
        ]
        evaluate_paddleocr_vl.validate_candidate_coverage(
            predictions, ("base", "merged"), 2
        )

        with self.assertRaisesRegex(ValueError, "coverage"):
            evaluate_paddleocr_vl.validate_candidate_coverage(
                predictions[:-1], ("base", "merged"), 2
            )
        with self.assertRaisesRegex(ValueError, "coverage"):
            evaluate_paddleocr_vl.validate_candidate_coverage(
                [*predictions, predictions[-1]], ("base", "merged"), 2
            )

    def test_evaluator_reuses_only_matching_base_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            row = {
                "dataset": "a",
                "image": "/tmp/one.png",
                "target": "xin chào",
                "task": "ocr",
            }
            base = {**row, "candidate": "base", "prediction": "xin chào"}
            merged = {**row, "candidate": "merged", "prediction": "khác"}
            path.write_text(
                "\n".join(json.dumps(item) for item in (base, merged)) + "\n",
                encoding="utf-8",
            )

            loaded = evaluate_paddleocr_vl.load_base_predictions(path, [row])

        self.assertEqual(loaded, [base])

    def test_evaluator_aliases_remote_inputs_embeds_mask_keyword(self):
        calls = []

        class FakeMaskingUtils:
            @staticmethod
            def create_causal_mask(*, input_embeds, attention_mask=None):
                calls.append((input_embeds, attention_mask))
                return "mask"

        evaluate_paddleocr_vl.install_masking_utils_compatibility(FakeMaskingUtils)
        result = FakeMaskingUtils.create_causal_mask(
            inputs_embeds="embeds", attention_mask="attention"
        )

        self.assertEqual(result, "mask")
        self.assertEqual(calls, [("embeds", "attention")])

    def test_evaluator_rejects_invalid_target_or_image_match_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.png").write_bytes(png_bytes())
            validation = root / "validation-source-000.jsonl"
            finetune_vl.write_erniekit_jsonl(
                validation,
                [finetune_vl.PreparedSample("sample.png", "một", 0)],
            )
            payload = json.loads(validation.read_text())
            payload["text_info"][1]["tag"] = "mask"
            payload["image_info"][0]["matched_text_index"] = 1
            validation.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "target|image"):
                evaluate_paddleocr_vl.load_validation_rows([validation], 1)

    def test_evaluator_loads_deterministic_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.png"
            image.write_bytes(png_bytes())
            validation = root / "validation-source-007.jsonl"
            finetune_vl.write_erniekit_jsonl(
                validation,
                [
                    finetune_vl.PreparedSample("sample.png", "một", 0),
                    finetune_vl.PreparedSample("sample.png", "hai", 0),
                ],
            )

            rows = evaluate_paddleocr_vl.load_validation_rows([validation], 1)

        self.assertEqual(
            rows,
            [
                {
                    "dataset": "source-007",
                    "image": str(image.resolve()),
                    "target": "một",
                    "prompt": "OCR:",
                    "task": "ocr",
                }
            ],
        )

    def test_evaluator_uses_native_processor_for_base_and_merged_candidates(self):
        calls = []

        class FakeTensor:
            def __init__(self, values):
                self.values = values
                self.shape = (len(values), len(values[0]))

            def to(self, device):
                calls.append(("tensor.to", device))
                return self

            def __getitem__(self, key):
                row_slice, column_slice = key
                rows = self.values[row_slice]
                return FakeTensor([row[column_slice] for row in rows])

            def tolist(self):
                return self.values

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls.append(("processor.load", path, kwargs))
                return cls()

            def apply_chat_template(self, messages, **kwargs):
                calls.append(("chat", messages, kwargs))
                return "rendered-native-prompt"

            def __call__(self, **kwargs):
                calls.append(("processor", kwargs))
                return {
                    "input_ids": FakeTensor([[10, 11, 12]]),
                    "pixel_values": FakeTensor([[1, 2]]),
                }

            def batch_decode(self, token_ids, **kwargs):
                calls.append(("decode", token_ids.tolist(), kwargs))
                return ["native output"]

        class FakeModel:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls.append(("model.load", path, kwargs))
                return cls()

            def to(self, device):
                calls.append(("model.to", device))
                return self

            def eval(self):
                calls.append(("model.eval",))
                return self

            def generate(self, **kwargs):
                calls.append(("generate", kwargs))
                return FakeTensor([[10, 11, 12, 20, 21]])

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def empty_cache():
                calls.append(("empty_cache",))

            @staticmethod
            def synchronize():
                calls.append(("synchronize",))

        class FakeInferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

        class FakeTorch:
            bfloat16 = "bfloat16"
            cuda = FakeCuda()

            @staticmethod
            def inference_mode():
                return FakeInferenceMode()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.png"
            image.write_bytes(png_bytes())
            validation = root / "validation-source-000.jsonl"
            finetune_vl.write_erniekit_jsonl(
                validation,
                [finetune_vl.PreparedSample("sample.png", "native output", 0)],
            )
            base = root / "base"
            merged = root / "merged"
            base.mkdir()
            merged.mkdir()
            args = argparse.Namespace(
                base_model=base,
                merged_model=merged,
                validation_jsonl=[validation],
                output_dir=root / "metrics",
                samples_per_dataset=1,
                max_new_tokens=64,
            )
            report = evaluate_paddleocr_vl.evaluate(
                args,
                torch_module=FakeTorch,
                auto_processor_class=FakeProcessor,
                auto_model_class=FakeModel,
                image_loader=lambda image_path: f"rgb:{image_path}",
            )

        self.assertEqual(set(report["candidates"]), {"base", "merged"})
        self.assertEqual(report["fixture_count"], 1)
        self.assertEqual([call[0] for call in calls].count("processor.load"), 2)
        chat_call = next(call for call in calls if call[0] == "chat")
        self.assertEqual(
            chat_call[2], {"tokenize": False, "add_generation_prompt": True}
        )
        processor_call = next(call for call in calls if call[0] == "processor")
        self.assertEqual(processor_call[1]["text"], ["rendered-native-prompt"])
        self.assertEqual(processor_call[1]["images"], [f"rgb:{image.resolve()}"])
        generation_call = next(call for call in calls if call[0] == "generate")
        self.assertFalse(generation_call[1]["do_sample"])
        self.assertEqual(generation_call[1]["num_beams"], 1)
        self.assertEqual(generation_call[1]["max_new_tokens"], 64)

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "datasets not installed")
    def test_load_hf_dataset_reads_save_to_disk(self):
        from datasets import Dataset, Features, Value
        from datasets import Image as HFImage

        dataset = Dataset.from_dict(
            {
                "image": [{"bytes": png_bytes(), "path": "sample.png"}],
                "label": ["Việt Nam"],
            },
            features=Features({"image": HFImage(), "label": Value("string")}),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset"
            dataset.save_to_disk(path)
            loaded = finetune_vl.load_hf_dataset(path)
            row = loaded[0]
        self.assertEqual(row["label"], "Việt Nam")
        self.assertIsInstance(row["image"]["bytes"], bytes)

    def test_loader_contract_is_reused(self):
        self.assertIs(
            finetune_vl.load_hf_dataset, finetune_vl.rec_loader.load_hf_dataset
        )
        self.assertEqual(
            finetune_vl.normalize_text({"label": " ", "text": " Tiếng\tViệt "}),
            "Tiếng Việt",
        )

    def test_training_requires_explicit_local_model_snapshot(self):
        with self.assertRaisesRegex(FileNotFoundError, "download_pretrained_models"):
            finetune_vl.require_local_model_snapshot(finetune_vl.DEFAULT_MODEL)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text("{}")
            (model / "preprocessor_config.json").write_text("{}")
            with self.assertRaisesRegex(FileNotFoundError, "weights|tokenizer"):
                finetune_vl.require_local_model_snapshot(str(model))
            (model / "model.safetensors").write_bytes(b"weights")
            (model / "tokenizer.model").write_bytes(b"tokenizer")
            self.assertEqual(
                finetune_vl.require_local_model_snapshot(str(model)), model.resolve()
            )

    def test_work_and_merge_outputs_cannot_live_inside_base_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            adapter = root / "adapter"
            model.mkdir()
            adapter.mkdir()
            (model / "config.json").write_text(
                json.dumps({"model_type": "paddleocr_vl"})
            )
            (adapter / "lora_config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "immutable base model"):
                finetune_vl.validate_work_dir_isolation(model / "run", model)
            with self.assertRaisesRegex(ValueError, "immutable base model"):
                merge_paddleocr_vl_lora.validate_inputs(
                    model, adapter, model / "export"
                )

    def test_visual_token_reserve_is_included_in_sequence_budget(self):
        self.assertEqual(
            finetune_vl.total_multimodal_tokens(
                CharacterTokenizer(), "abc", visual_tokens=10
            ),
            len("OCR:abc") + 1 + 10,
        )
        self.assertEqual(
            finetune_vl.smart_resize_dimensions(
                100, 200, min_pixels=50_176, max_pixels=451_584
            ),
            (168, 336),
        )
        self.assertEqual(
            finetune_vl.visual_token_count(
                100,
                200,
                min_pixels=50_176,
                max_pixels=451_584,
            ),
            72,
        )

    def test_process_split_decodes_rgb_and_rejects_bad_rows(self):
        rows = [
            {"image": {"bytes": png_bytes(mode="L")}, "text": "Tiếng Việt"},
            {"image": {"bytes": b"not an image"}, "text": "hỏng"},
            {"image": {"bytes": png_bytes(size=(1, 10))}, "text": "rỗng"},
            {"image": {"bytes": png_bytes(size=(40, 40))}, "text": "quá lớn"},
            {"image": {"bytes": png_bytes()}, "label": " ", "text": " "},
            {"image": {"bytes": png_bytes()}, "text": "nul\x00text"},
            {"image": {"bytes": png_bytes()}, "text": "012345678901234567890123456789"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with finetune_vl.RejectionReport(root / "rejected.jsonl") as report:
                samples = finetune_vl.process_split(
                    FakeSplit(rows),
                    dataset_dir=root,
                    dataset_index=0,
                    split_name="train",
                    prepared_dir=root / "prepared",
                    max_image_pixels=1_000,
                    max_seq_len=100,
                    tokenizer=CharacterTokenizer(),
                    report=report,
                )
                counts = dict(report.counts)

            saved = Image.open(root / "prepared" / samples[0].image_path)
            self.assertEqual(saved.mode, "RGB")
            saved.close()
            self.assertEqual(samples[0].text, "Tiếng Việt")
            self.assertEqual(
                counts,
                {
                    "invalid_image": 2,
                    "pixel_limit_exceeded": 1,
                    "empty_text": 1,
                    "control_character": 1,
                    "token_budget_exceeded": 1,
                },
            )

    def test_split_is_seeded_without_leakage(self):
        samples = [finetune_vl.PreparedSample(str(i), str(i), 0) for i in range(20)]
        first = finetune_vl.split_train_validation(samples, 0.2, 19)
        second = finetune_vl.split_train_validation(samples, 0.2, 19)
        self.assertEqual(first, second)
        self.assertFalse(set(first[0]).intersection(first[1]))
        self.assertEqual((len(first[0]), len(first[1])), (16, 4))

    def test_sqrt_probabilities_soft_balance_sources(self):
        self.assertEqual(finetune_vl.sqrt_probabilities([9, 1]), [0.75, 0.25])
        with self.assertRaisesRegex(ValueError, "positive"):
            finetune_vl.sqrt_probabilities([9, 0])

    def test_jsonl_uses_only_ocr_prompt_and_mask_contract(self):
        sample = finetune_vl.PreparedSample("images/sample.png", "Việt Nam", 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            finetune_vl.write_erniekit_jsonl(path, [sample])
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload,
            {
                "image_info": [
                    {"image_url": "images/sample.png", "matched_text_index": 0}
                ],
                "text_info": [
                    {"text": "OCR:", "tag": "mask"},
                    {"text": "Việt Nam", "tag": "no_mask"},
                ],
            },
        )

    def test_prepare_writes_per_source_splits_summary_and_probabilities(self):
        def source(prefix, size):
            return FakeSplit(
                [
                    {"image": {"bytes": png_bytes()}, "text": f"{prefix}{index}"}
                    for index in range(size)
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [root / "large", root / "small"]
            for source_dir in sources:
                source_dir.mkdir()
            args = argparse.Namespace(
                dataset_dir=sources,
                validation_ratio=0.2,
                seed=7,
                max_image_pixels=10_000,
                min_pixels=1568,
                max_pixels=1568,
                max_seq_len=100,
                dataset_task=["ocr", "ocr"],
            )
            with patch.object(
                finetune_vl,
                "load_hf_dataset",
                side_effect=[source("a", 10), source("b", 2)],
            ):
                summary = finetune_vl.prepare_datasets(
                    args, root / "run", CharacterTokenizer()
                )

            prepared = root / "run" / "prepared"
            self.assertTrue((prepared / "train-source-000.jsonl").is_file())
            self.assertTrue((prepared / "validation-source-001.jsonl").is_file())
            self.assertEqual(summary["sources"][0]["train_samples"], 8)
            self.assertEqual(summary["sources"][1]["validation_samples"], 1)
            expected = finetune_vl.sqrt_probabilities([8, 1])
            self.assertEqual(summary["train_probabilities"], expected)
            self.assertEqual(
                json.loads((root / "run" / "summary.json").read_text()), summary
            )

    def test_resolved_config_is_lora_only_16gb_profile(self):
        args = argparse.Namespace(
            model="PaddlePaddle/PaddleOCR-VL-1.6",
            epochs=3,
            learning_rate=1e-4,
            lora_rank=32,
            min_pixels=50_176,
            max_pixels=451_584,
            max_seq_len=2048,
            gradient_accumulation_steps=16,
            num_workers=2,
            prefetch_factor=2,
            seed=2026,
            flash_attention=True,
            devices="0",
            save_steps=1,
            smoke_steps=None,
            resume_from=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = {
                "train_samples": 220_691,
                "sources": [
                    {
                        "train_jsonl": str(
                            root / "prepared" / "train-source-000.jsonl"
                        ),
                        "validation_jsonl": str(
                            root / "prepared" / "validation-source-000.jsonl"
                        ),
                    }
                ],
                "train_probabilities": [1.0],
                "validation_probabilities": [1.0],
            }
            target = root / "resolved.yaml"
            config = finetune_vl.create_resolved_config(target, root, summary, args)
            loaded = yaml.safe_load(target.read_text(encoding="utf-8"))

        self.assertEqual(loaded, config)
        self.assertEqual(config["stage"], "OCR-VL-SFT")
        self.assertEqual(config["fine_tuning"], "LoRA")
        self.assertEqual(config["lora_rank"], 32)
        self.assertEqual(config["freeze_config"], "freeze_vision")
        self.assertEqual(
            config["batch_size"]
            * config["packing_size"]
            * config["gradient_accumulation_steps"],
            16,
        )
        self.assertEqual(
            (config["min_pixels"], config["max_pixels"]), (50_176, 451_584)
        )
        self.assertTrue(config["recompute"])
        self.assertEqual(config["recompute_granularity"], "full")
        self.assertEqual(config["compute_type"], "bf16")
        self.assertEqual(config["fp16_opt_level"], "O2")
        self.assertTrue(config["use_huggingface_model"])
        self.assertTrue(config["convert_from_hf"])
        self.assertTrue(config["save_to_hf"])
        self.assertFalse(config["do_eval"])
        self.assertFalse(config["overwrite_output_dir"])
        self.assertEqual(config["max_steps"], 41_380)
        self.assertEqual(config["save_steps"], 1)

    def test_resolved_config_accounts_for_all_selected_data_parallel_devices(self):
        args = argparse.Namespace(
            model="/models/PaddleOCR-VL-1.6",
            epochs=3,
            learning_rate=1e-4,
            lora_rank=32,
            min_pixels=50_176,
            max_pixels=451_584,
            max_seq_len=2048,
            gradient_accumulation_steps=16,
            num_workers=2,
            prefetch_factor=2,
            seed=2026,
            flash_attention=True,
            devices="0,1",
            smoke_steps=None,
            resume_from=None,
        )
        summary = {
            "train_samples": 160,
            "sources": [
                {
                    "train_jsonl": "/data/train.jsonl",
                    "validation_jsonl": "/data/validation.jsonl",
                }
            ],
            "train_probabilities": [1.0],
            "validation_probabilities": [1.0],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = finetune_vl.create_resolved_config(
                root / "resolved.yaml", root, summary, args
            )

        self.assertEqual(config["max_steps"], 15)

    def test_resume_and_adapter_base_model_must_match_resolved_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_model = root / "original-model"
            other_model = root / "other-model"
            original_model.mkdir()
            other_model.mkdir()
            resolved = root / "resolved.yaml"
            resolved.write_text(
                yaml.safe_dump({"model_name_or_path": str(original_model)}),
                encoding="utf-8",
            )
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "lora_config.json").write_text(
                json.dumps({"base_model_name_or_path": str(original_model)}),
                encoding="utf-8",
            )

            self.assertEqual(
                finetune_vl.resolve_run_model(resolved, str(original_model)),
                original_model.resolve(),
            )
            finetune_vl.validate_adapter_base_model(adapter, original_model)
            with self.assertRaisesRegex(ValueError, "base model"):
                finetune_vl.resolve_run_model(resolved, str(other_model))
            with self.assertRaisesRegex(ValueError, "base model"):
                finetune_vl.validate_adapter_base_model(adapter, other_model)

    def test_smoke_and_resume_are_resolved_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "run" / "adapter" / "checkpoint-2"
            checkpoint.mkdir(parents=True)
            args = argparse.Namespace(smoke_steps=3, resume_from=checkpoint)
            overrides = finetune_vl.runtime_overrides(args)
        self.assertEqual(
            overrides,
            {"max_steps": 3, "resume_from_checkpoint": str(checkpoint.resolve())},
        )
        self.assertNotIn("overwrite_output_dir", overrides)

    def test_work_dir_refuses_overwrite_but_accepts_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            (run / "summary.json").write_text("{}")
            checkpoint = run / "adapter" / "checkpoint-1"
            checkpoint.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                finetune_vl.make_work_dir(run, None)
            self.assertEqual(finetune_vl.make_work_dir(run, checkpoint), run.resolve())

    def test_resume_writes_runtime_config_without_mutating_resolved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "resolved.yaml"
            original.write_text("max_steps: 100\noverwrite_output_dir: false\n")
            checkpoint = root / "adapter" / "checkpoint-4"
            checkpoint.mkdir(parents=True)
            target = finetune_vl.create_resume_config(
                original,
                root,
                argparse.Namespace(smoke_steps=3, resume_from=checkpoint),
            )
            self.assertEqual(
                original.read_text(), "max_steps: 100\noverwrite_output_dir: false\n"
            )
            resumed = yaml.safe_load(target.read_text())
        self.assertEqual(resumed["max_steps"], 3)
        self.assertEqual(resumed["resume_from_checkpoint"], str(checkpoint.resolve()))
        self.assertFalse(resumed["overwrite_output_dir"])

    def test_trainable_parameter_guard_requires_lora_and_small_fraction(self):
        valid = finetune_vl.TrainableParameterReport(
            trainable=8_000_000,
            total=1_000_000_000,
            names=(
                "model.model.layers.0.self_attn.q_proj.lora_A",
                "model.model.layers.0.self_attn.q_proj.lora_B",
            ),
        )
        finetune_vl.validate_trainable_parameters(valid)
        with self.assertRaisesRegex(RuntimeError, "LoRA"):
            finetune_vl.validate_trainable_parameters(
                finetune_vl.TrainableParameterReport(10, 1000, ("weight",))
            )
        with self.assertRaisesRegex(RuntimeError, "base model"):
            finetune_vl.validate_trainable_parameters(
                finetune_vl.TrainableParameterReport(
                    900,
                    1000,
                    ("model.model.layers.0.self_attn.q_proj.lora_A",),
                )
            )
        with self.assertRaisesRegex(RuntimeError, "outside"):
            finetune_vl.validate_trainable_parameters(
                finetune_vl.TrainableParameterReport(
                    10, 1000, ("model.connector.proj.lora_A",)
                )
            )

    def test_trainable_report_requires_exact_parameter_names_from_compat_layer(self):
        output = (
            "PADDLEOCR_VL_TRAINABLE_PARAMETER_NAMES="
            '["model.layers.0.self_attn.q_proj.lora_A", '
            '"model.layers.0.self_attn.q_proj.lora_B"]\n'
            "Frozen parameters: 9.92e+08 || Trainable parameters:8.00e+06 || "
            "Total parameters:1.00e+09|| Trainable:0.80%\n"
        )
        report = finetune_vl.parse_trainable_parameter_output(output)
        self.assertEqual(
            report.names,
            (
                "model.layers.0.self_attn.q_proj.lora_A",
                "model.layers.0.self_attn.q_proj.lora_B",
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "names"):
            finetune_vl.parse_trainable_parameter_output(
                "Start to wrap model with LoRA config\n"
                "Trainable parameters:8.00e+06 || Total parameters:1.00e+09"
            )

    def test_checkpoint_compat_forwards_paddleformers_save_arguments(self):
        from erniekit_compat.sitecustomize import patch_pretraining_trainer_class

        calls = []

        class BaseTrainer:
            def save_model(
                self,
                output_dir=None,
                merge_tensor_parallel=False,
                last_fc_to_hf=False,
            ):
                calls.append((output_dir, merge_tensor_parallel, last_fc_to_hf))

        class IncompatibleTrainer(BaseTrainer):
            def save_model(self, output_dir=None):
                super().save_model(output_dir)

        patch_pretraining_trainer_class(IncompatibleTrainer)
        trainer = IncompatibleTrainer()
        trainer.args = argparse.Namespace(should_save=False, output_dir="fallback")
        trainer.save_model("checkpoint-1", True, True)

        self.assertEqual(calls, [("checkpoint-1", True, True)])

    def test_runtime_dependency_versions_must_match_pinned_profile(self):
        expected = dict(finetune_vl.ERNIEKIT_RUNTIME_VERSIONS)
        finetune_vl.validate_dependency_versions(expected)
        incompatible = dict(expected)
        incompatible["paddleformers"] = "9.9.9"
        with self.assertRaisesRegex(RuntimeError, "paddleformers"):
            finetune_vl.validate_dependency_versions(incompatible)
        missing = dict(expected)
        del missing["transformers"]
        with self.assertRaisesRegex(RuntimeError, "transformers"):
            finetune_vl.validate_dependency_versions(missing)

    def test_adapter_scope_requires_all_decoder_projections_and_no_vision(self):
        valid = [
            f"model.layers.0.{path}.lora_{side}"
            for path in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.up_proj",
                "mlp.gate_proj",
                "mlp.down_proj",
            )
            for side in ("A", "B")
        ]
        finetune_vl.validate_lora_parameter_names(valid)
        with self.assertRaisesRegex(RuntimeError, "vision"):
            finetune_vl.validate_lora_parameter_names(
                valid + ["visual.encoder.layers.0.self_attn.q_proj.lora_A"]
            )
        with self.assertRaisesRegex(RuntimeError, "k_proj"):
            finetune_vl.validate_lora_parameter_names(
                [name for name in valid if "k_proj" not in name]
            )

    def test_logged_command_keeps_nested_python_in_selected_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_bin = root / "selected" / "bin"
            selected_bin.mkdir(parents=True)
            selected_python = selected_bin / "python"
            selected_python.symlink_to(Path(sys.executable))
            output, _ = finetune_vl.run_logged_command(
                [
                    str(selected_python),
                    "-c",
                    (
                        "import subprocess; "
                        "subprocess.run(['python', '-c', "
                        "'import sys; print(sys.executable)'], check=True)"
                    ),
                ],
                root,
                root / "command.log",
            )

        self.assertIn(str(selected_python), output)

    def test_process_cleanup_signals_the_whole_launcher_group(self):
        class FakeProcess:
            pid = 4321

            def __init__(self):
                self.wait_timeouts = []

            def wait(self, timeout=None):
                self.wait_timeouts.append(timeout)
                return 0

        process = FakeProcess()
        with patch("finetune_vl.os.killpg") as kill_group:
            finetune_vl.terminate_process_group(process)

        kill_group.assert_called_once_with(4321, finetune_vl.signal.SIGTERM)
        self.assertEqual(process.wait_timeouts, [10])

    def test_inspect_accepts_known_erniekit_v15_dry_run_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = root / "erniekit" / "train" / "ocr_vl_sft" / "workflow.py"
            peft = root / "ernie" / "utils" / "peft_utils.py"
            workflow.parent.mkdir(parents=True)
            peft.parent.mkdir(parents=True)
            workflow.write_text("freeze_vision initialize_lora_model")
            peft.write_text("mark_only_lora_as_trainable")
            fake_python = root / ".venv" / "bin" / "python"
            fake_python.parent.mkdir(parents=True)
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "echo 'PADDLEOCR_VL_LORA_SCOPE=text_decoder_only'\n"
                "echo 'PADDLEOCR_VL_TRAINABLE_PARAMETER_NAMES=[\"model.layers.0.self_attn.q_proj.lora_A\"]'\n"
                "echo 'Frozen parameters: 9.59e+08 || Trainable parameters:4.13e+06 || Total parameters:9.63e+08'\n"
                "echo \"AttributeError: 'FinetuningArguments' object has no attribute 'is_train_mm'\"\n"
                "exit 1\n"
            )
            fake_python.chmod(0o755)
            work_dir = root / "run"
            report = finetune_vl.inspect_model(root, root / "resolved.yaml", work_dir)
            metrics_exist = (
                work_dir / "metrics" / "trainable_parameters.json"
            ).is_file()

        self.assertEqual(report.trainable, 4_130_000)
        self.assertEqual(report.total, 963_000_000)
        self.assertTrue(metrics_exist)

    def test_erniekit_compat_targets_decoder_and_excludes_vision(self):
        from erniekit_compat.sitecustomize import decoder_target_modules

        patterns = decoder_target_modules()
        self.assertEqual(len(patterns), 7)
        decoder_modules = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.down_proj",
        ]
        for name in decoder_modules:
            self.assertTrue(any(re.fullmatch(pattern, name) for pattern in patterns))
            self.assertTrue(
                any(re.fullmatch(pattern, f"{name}.weight") for pattern in patterns)
            )
        self.assertFalse(
            any(
                re.fullmatch(
                    pattern, "visual.vision_model.encoder.layers.0.self_attn.q_proj"
                )
                for pattern in patterns
            )
        )

    def test_export_uses_paddleocr_vl_mergekit_compatibility_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            work_dir = root / "run"
            fixture = root / "validation.jsonl"
            fixture.write_text("{}\n")
            command = finetune_vl.build_export_command(
                root,
                "/models/PaddleOCR-VL-1.6",
                work_dir,
                fixture_jsonl=fixture,
                min_pixels=50_176,
                max_pixels=451_584,
            )

        self.assertEqual(command[0], str(python))
        self.assertTrue(command[1].endswith("merge_paddleocr_vl_lora.py"))
        self.assertEqual(
            command[2:],
            [
                "--base-model",
                "/models/PaddleOCR-VL-1.6",
                "--adapter-dir",
                str((work_dir / "adapter").resolve()),
                "--output-dir",
                str((work_dir / "adapter" / "export").resolve()),
                "--fixture-jsonl",
                str(fixture.resolve()),
                "--min-pixels",
                "50176",
                "--max-pixels",
                "451584",
            ],
        )

    def test_evaluation_command_uses_native_runtime_and_all_validation_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            work_dir = root / "run"
            merged_model = work_dir / "adapter" / "export"
            paths = [root / "validation-a.jsonl", root / "validation-b.jsonl"]
            command = finetune_vl.build_evaluation_command(
                root,
                "/models/PaddleOCR-VL-1.6",
                work_dir,
                merged_model=merged_model,
                validation_jsonls=paths,
                samples_per_dataset=7,
                max_new_tokens=128,
            )

        self.assertEqual(command[0], str(python.resolve()))
        self.assertTrue(command[1].endswith("evaluate_paddleocr_vl.py"))
        self.assertEqual(
            command[command.index("--merged-model") + 1],
            str(merged_model.resolve()),
        )
        self.assertEqual(
            command[
                command.index("--validation-jsonl") + 1 : command.index("--output-dir")
            ],
            [str(path.resolve()) for path in paths],
        )
        self.assertEqual(command[command.index("--samples-per-dataset") + 1], "7")
        self.assertEqual(command[command.index("--max-new-tokens") + 1], "128")
        for obsolete in (
            "--adapter-dir",
            "--min-pixels",
            "--max-pixels",
            "--max-checkpoints",
            "--task",
        ):
            self.assertNotIn(obsolete, command)

    def test_copy_inference_assets_includes_remote_model_code(self):
        expected = {
            "config.json",
            "configuration_paddleocr_vl.py",
            "modeling_paddleocr_vl.py",
            "image_processing_paddleocr_vl.py",
            "processing_paddleocr_vl.py",
            "processor_config.json",
            "chat_template.jinja",
            "inference.yml",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            export = root / "export"
            model.mkdir()
            for name in expected:
                (model / name).write_text(name, encoding="utf-8")

            copied = set(finetune_vl.copy_inference_assets(str(model), export))

        self.assertEqual(copied, expected)

    def test_parses_paddleformers_scientific_trainable_report(self):
        report = finetune_vl.parse_trainable_parameter_output(
            "PADDLEOCR_VL_TRAINABLE_PARAMETER_NAMES="
            '["model.layers.0.self_attn.q_proj.lora_A"]\n'
            "Frozen parameters: 9.92e+08 || Trainable parameters:8.00e+06 || "
            "Total parameters:1.00e+09|| Trainable:0.80%\n"
        )
        self.assertEqual(report.trainable, 8_000_000)
        self.assertEqual(report.total, 1_000_000_000)
        finetune_vl.validate_trainable_parameters(report)

    def test_metrics_report_cer_exact_and_normalized_edit_distance(self):
        report = finetune_vl.compute_ocr_metrics(
            [
                {"dataset": "a", "target": "abc", "prediction": "abc"},
                {"dataset": "b", "target": "abcd", "prediction": "abXd"},
            ]
        )
        self.assertEqual(report["overall"]["exact_match"], 0.5)
        self.assertAlmostEqual(report["overall"]["cer"], 1 / 7)
        self.assertAlmostEqual(report["overall"]["normalized_edit_distance"], 0.875)
        self.assertEqual(report["datasets"]["a"]["exact_match"], 1.0)

    def test_metrics_and_quality_gate_are_task_aware_and_fail_regression(self):
        base_rows = [
            {
                "dataset": "a",
                "task": "ocr",
                "target": "abc",
                "prediction": "abc",
            }
        ]
        merged_rows = [dict(base_rows[0], prediction="axc")]
        reports = {
            "base": finetune_vl.compute_ocr_metrics(base_rows),
            "merged": finetune_vl.compute_ocr_metrics(merged_rows),
        }
        failures = evaluate_paddleocr_vl.quality_gate_failures(
            reports,
            [],
            min_normalized_edit_distance=0.5,
            max_cer=1.0,
        )
        self.assertIn("ocr", reports["merged"]["tasks"])
        self.assertTrue(
            any(failure["type"] == "regression_vs_base" for failure in failures)
        )

    def test_generation_status_marks_limit_without_eos_as_truncated(self):
        generated = np.array([[10, 11, 20, 21]])
        status = evaluate_paddleocr_vl.generated_token_status(
            generated, 2, 2, (99,)
        )
        self.assertTrue(status["truncated"])
        self.assertEqual(status["finish_reason"], "length")
        ended = evaluate_paddleocr_vl.generated_token_status(
            np.array([[10, 11, 20, 99]]), 2, 2, (99,)
        )
        self.assertFalse(ended["truncated"])
        self.assertEqual(ended["finish_reason"], "eos")

    def test_adapter_candidates_include_final_and_latest_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            adapter = work_dir / "adapter"
            adapter.mkdir()
            (adapter / "lora_config.json").write_text("{}")
            for step in (10, 30, 20):
                checkpoint = adapter / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "lora_config.json").write_text("{}")
            candidates = finetune_vl.adapter_candidates(work_dir, 2)
        self.assertEqual(
            [candidate.name for candidate in candidates],
            ["adapter", "checkpoint-30", "checkpoint-20"],
        )

    def test_candidate_evaluation_selects_best_checkpoint_and_cleans_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            adapter = work_dir / "adapter"
            adapter.mkdir()
            (adapter / "lora_config.json").write_text("{}")
            checkpoint = adapter / "checkpoint-20"
            checkpoint.mkdir()
            (checkpoint / "lora_config.json").write_text("{}")
            model = work_dir / "model"
            model.mkdir()
            validation = work_dir / "validation.jsonl"
            validation.write_text("{}\n")
            metrics = {
                "adapter": (0.2, 0.5, 0.8),
                "checkpoint-20": (0.1, 0.7, 0.9),
            }
            evaluation_commands = []

            def fake_run(command, *_args, **_kwargs):
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                if str(command[1]).endswith("evaluate_paddleocr_vl.py"):
                    evaluation_commands.append(command)
                    candidate = Path(
                        command[command.index("--merged-model") + 1]
                    ).name
                    cer, exact_match, normalized = metrics[candidate]
                    report = {
                        "status": "passed",
                        "failures": [],
                        "candidates": {
                            "merged": {
                                "overall": {
                                    "samples": 1,
                                    "cer": cer,
                                    "exact_match": exact_match,
                                    "normalized_edit_distance": normalized,
                                }
                            }
                        },
                    }
                    (output_dir / "ocr_metrics.json").write_text(
                        json.dumps(report), encoding="utf-8"
                    )
                    (output_dir / "ocr_predictions.jsonl").write_text(
                        "{}\n", encoding="utf-8"
                    )
                return "", None

            args = argparse.Namespace(
                eval_max_checkpoints=1,
                erniekit_dir=work_dir,
                min_pixels=50_176,
                max_pixels=451_584,
                devices="0",
                eval_max_new_tokens=1024,
                eval_task_max_new_tokens=[],
                min_normalized_edit_distance=0.5,
                max_cer=1.0,
                smoke_steps=None,
            )
            with (
                patch.object(finetune_vl, "validate_adapter_scope"),
                patch.object(finetune_vl, "validate_adapter_base_model"),
                patch.object(finetune_vl, "run_logged_command", side_effect=fake_run),
            ):
                selected, report = finetune_vl.evaluate_adapter_candidates(
                    args,
                    work_dir,
                    model,
                    [validation],
                    validation,
                    1,
                )

            self.assertEqual(selected, checkpoint.resolve())
            self.assertEqual(report["selected"]["checkpoint"], "checkpoint-20")
            self.assertNotIn("--base-predictions-jsonl", evaluation_commands[0])
            self.assertIn("--base-predictions-jsonl", evaluation_commands[1])
            self.assertIn("--report-only", evaluation_commands[0])
            self.assertIn("--report-only", evaluation_commands[1])
            self.assertFalse((work_dir / ".checkpoint-evaluation").exists())

    def test_failed_evaluation_exits_nonzero_unless_report_only(self):
        failed = {"status": "failed", "fixture_count": 1, "candidates": {}}
        with (
            patch.object(evaluate_paddleocr_vl, "parse_args") as parse_args,
            patch.object(evaluate_paddleocr_vl, "evaluate", return_value=failed),
        ):
            parse_args.return_value = argparse.Namespace(report_only=False)
            self.assertEqual(evaluate_paddleocr_vl.main([]), 1)
            parse_args.return_value = argparse.Namespace(report_only=True)
            self.assertEqual(evaluate_paddleocr_vl.main([]), 0)

    def test_export_promotion_restores_previous_directory_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            destination = root / "export"
            candidate.mkdir()
            destination.mkdir()
            (candidate / "model.safetensors").write_text("new")
            (destination / "model.safetensors").write_text("old")
            real_replace = os.replace
            calls = 0

            def fail_candidate_promotion(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("promotion failed")
                return real_replace(source, target)

            with (
                patch.object(finetune_vl.os, "replace", side_effect=fail_candidate_promotion),
                self.assertRaisesRegex(OSError, "promotion failed"),
            ):
                finetune_vl.promote_export_directory(candidate, destination)

            self.assertEqual(
                (destination / "model.safetensors").read_text(), "old"
            )
            self.assertTrue(candidate.is_dir())

            finetune_vl.promote_export_directory(candidate, destination)
            self.assertEqual(
                (destination / "model.safetensors").read_text(), "new"
            )
            self.assertFalse(candidate.exists())

    def test_export_status_distinguishes_verified_failed_and_skipped(self):
        self.assertEqual(
            finetune_vl.export_verification_status(True, None), "unverified"
        )
        self.assertEqual(
            finetune_vl.export_verification_status(False, {"status": "failed"}),
            "failed",
        )
        self.assertEqual(
            finetune_vl.export_verification_status(False, {"status": "passed"}),
            "passed",
        )

    def test_selected_checkpoint_is_confined_to_adapter_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            checkpoint = work_dir / "adapter" / "checkpoint-20"
            checkpoint.mkdir(parents=True)
            (checkpoint / "lora_config.json").write_text("{}")
            selected = finetune_vl.resolve_selected_adapter(
                work_dir, {"checkpoint": "checkpoint-20"}
            )
            self.assertEqual(selected, checkpoint.resolve())
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                finetune_vl.resolve_selected_adapter(
                    work_dir, {"checkpoint": "../outside"}
                )

    def test_download_script_fetches_full_vl_snapshot_with_revision_and_resume(self):
        script = Path(finetune_vl.__file__).with_name("download_pretrained_models.sh")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            trace = root / "trace.txt"
            fake_cli = fake_bin / "huggingface-cli"
            fake_cli.write_text(
                "#!/usr/bin/env bash\n"
                'printf \'%s\\n\' "$@" > "$HF_TRACE"\n'
                'args=("$@")\n'
                "for ((i=0; i<${#args[@]}; i++)); do\n"
                "  if [[ ${args[$i]} == --local-dir ]]; then\n"
                "    target=${args[$((i+1))]}\n"
                '    mkdir -p "$target"\n'
                "    printf '{}' > \"$target/config.json\"\n"
                "  fi\n"
                "done\n",
                encoding="utf-8",
            )
            fake_cli.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
            env["HF_TRACE"] = str(trace)
            result = subprocess.run(
                [
                    "bash",
                    str(script),
                    "vl",
                    str(root / "models"),
                    "--revision",
                    "revision-123",
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            arguments = trace.read_text().splitlines() if trace.exists() else []

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(arguments[0:2], ["download", finetune_vl.DEFAULT_MODEL])
        self.assertIn("--local-dir", arguments)
        self.assertIn("--resume-download", arguments)
        self.assertEqual(arguments[arguments.index("--revision") + 1], "revision-123")

    def test_download_script_supports_modern_hf_cli_with_automatic_resume(self):
        script = Path(finetune_vl.__file__).with_name("download_pretrained_models.sh")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            trace = root / "trace.txt"
            fake_cli = fake_bin / "hf"
            fake_cli.write_text(
                "#!/usr/bin/env bash\n"
                'printf \'%s\\n\' "$@" > "$HF_TRACE"\n'
                'args=("$@")\n'
                "for ((i=0; i<${#args[@]}; i++)); do\n"
                "  if [[ ${args[$i]} == --local-dir ]]; then\n"
                "    target=${args[$((i+1))]}\n"
                '    mkdir -p "$target"\n'
                "    printf '{}' > \"$target/config.json\"\n"
                "  fi\n"
                "done\n",
                encoding="utf-8",
            )
            fake_cli.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
            env["HF_TRACE"] = str(trace)
            result = subprocess.run(
                ["bash", str(script), "vl", str(root / "models")],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            arguments = trace.read_text().splitlines() if trace.exists() else []

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(arguments[0:2], ["download", finetune_vl.DEFAULT_MODEL])
        self.assertNotIn("--resume-download", arguments)

    def test_download_all_includes_paddleocr_vl_snapshot(self):
        script = Path(finetune_vl.__file__).with_name("download_pretrained_models.sh")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_curl = fake_bin / "curl"
            fake_curl.write_text(
                "#!/usr/bin/env bash\n"
                'args=("$@")\n'
                "for ((i=0; i<${#args[@]}; i++)); do\n"
                "  if [[ ${args[$i]} == --output ]]; then\n"
                '    mkdir -p "$(dirname "${args[$((i+1))]}")"\n'
                '    printf weights > "${args[$((i+1))]}"\n'
                "  fi\n"
                "done\n",
                encoding="utf-8",
            )
            fake_curl.chmod(0o755)
            fake_hf = fake_bin / "hf"
            fake_hf.write_text(
                "#!/usr/bin/env bash\n"
                'args=("$@")\n'
                "for ((i=0; i<${#args[@]}; i++)); do\n"
                "  if [[ ${args[$i]} == --local-dir ]]; then\n"
                "    target=${args[$((i+1))]}\n"
                '    mkdir -p "$target"\n'
                "    printf '{}' > \"$target/config.json\"\n"
                "  fi\n"
                "done\n",
                encoding="utf-8",
            )
            fake_hf.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
            output = root / "models"
            result = subprocess.run(
                ["bash", str(script), "all", str(output)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            downloaded = (output / "PaddleOCR-VL-1.6" / "config.json").is_file()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(downloaded)


if __name__ == "__main__":
    unittest.main()

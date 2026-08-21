import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import evaluate_paddleocr_vl
import finetune_vl
from paddleocr_vl_tasks import LAYOUT_TASKS, TASK_PROMPTS


class FakeSplit:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else ["image", "text", "task"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        extra = 2 if add_special_tokens else 0
        return list(range(len(text) + extra))


def png_bytes():
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (20, 10), 127).save(buffer, format="PNG")
    return buffer.getvalue()


def create_prepared_run(root: Path, tasks: list[str], *, legacy: bool = False) -> Path:
    label = "mixed" if len(tasks) > 1 else tasks[0]
    prepared_run = root / f"prepared-{label}"
    prepared = prepared_run / "prepared"
    images = prepared / "images" / "source-000"
    images.mkdir(parents=True)
    samples = []
    targets = {
        "ocr": "target-ocr",
        "table": "<fcel>target<table><nl>".replace("<table>", "-table"),
        "formula": r"\[x^2\]",
        "chart": "| A | B |\n|---|---|\n| 1 | 2 |",
    }
    for index, task in enumerate(tasks):
        name = f"train-{index}.png"
        (images / name).write_bytes(png_bytes())
        samples.append(
            finetune_vl.PreparedSample(
                f"images/source-000/{name}",
                targets[task],
                0,
                TASK_PROMPTS[task],
            )
        )
    (images / "validation.png").write_bytes(png_bytes())
    train_jsonl = prepared / "train-source-000.jsonl"
    validation_jsonl = prepared / "validation-source-000.jsonl"
    finetune_vl.write_erniekit_jsonl(train_jsonl, samples)
    finetune_vl.write_erniekit_jsonl(
        validation_jsonl,
        [
            finetune_vl.PreparedSample(
                "images/source-000/validation.png",
                targets[tasks[0]],
                0,
                TASK_PROMPTS[tasks[0]],
            )
        ],
    )
    summary = {
        "sources": [
            {
                "dataset": "/data/source",
                "train_samples": len(samples),
                "validation_samples": 1,
                "train_jsonl": "prepared/train-source-000.jsonl",
                "validation_jsonl": "prepared/validation-source-000.jsonl",
            }
        ],
        "train_samples": len(samples),
        "validation_samples": 1,
        "train_probabilities": [1.0],
        "validation_probabilities": [1.0],
        "rejected": {},
    }
    if legacy:
        summary["prompt"] = "OCR:"
    elif len(tasks) == 1:
        summary["task"] = tasks[0]
        summary["prompt"] = TASK_PROMPTS[tasks[0]]
        summary["tasks"] = tasks
        summary["prompts"] = [TASK_PROMPTS[tasks[0]]]
    else:
        summary["task"] = "mixed"
        summary["tasks"] = sorted(tasks)
        summary["prompts"] = [TASK_PROMPTS[task] for task in sorted(tasks)]
    (prepared_run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return prepared_run


class FinetuneVLLayoutTests(unittest.TestCase):
    def test_shared_cli_defaults_to_ocr_and_accepts_layout_defaults(self):
        self.assertEqual(finetune_vl.parse_args([]).task, "ocr")
        for task in LAYOUT_TASKS:
            self.assertEqual(finetune_vl.parse_args(["--task", task]).task, task)

    def test_mixed_jsonl_prompts_come_from_sample_task(self):
        samples = [
            finetune_vl.PreparedSample(
                "images/sample.png", "target", 0, TASK_PROMPTS[task]
            )
            for task in LAYOUT_TASKS
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            finetune_vl.write_erniekit_jsonl(path, samples)
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            [row["text_info"][0]["text"] for row in rows],
            [TASK_PROMPTS[task] for task in LAYOUT_TASKS],
        )
        self.assertTrue(all(row["text_info"][0]["tag"] == "mask" for row in rows))
        self.assertTrue(all(row["text_info"][1]["tag"] == "no_mask" for row in rows))

    def test_token_budget_uses_sample_prompt(self):
        tokenizer = CharacterTokenizer()
        for task in LAYOUT_TASKS:
            prompt = TASK_PROMPTS[task]
            self.assertEqual(
                finetune_vl.total_multimodal_tokens(
                    tokenizer, "abc", visual_tokens=10, prompt=prompt
                ),
                len(prompt + "abc") + 1 + 10,
            )

    def test_ocr_target_preserves_multiline_layout(self):
        target = finetune_vl.normalize_target(
            {"text": "  Dòng một\r\nDòng hai\rDòng ba  "}, "ocr"
        )
        self.assertEqual(target, "  Dòng một\nDòng hai\nDòng ba  ")

    def test_multiple_sources_without_task_column_require_dataset_mapping(self):
        args = argparse.Namespace(dataset_dir=[Path("a"), Path("b")], task="ocr")
        split = FakeSplit([{"image": {}, "text": "x"}])
        with self.assertRaisesRegex(ValueError, "dataset-task"):
            finetune_vl.dataset_default_task(args, 0, split, None)
        args.dataset_task = ["ocr", "formula"]
        self.assertEqual(
            finetune_vl.dataset_default_task(args, 1, split, None), "formula"
        )

    def test_table_target_requires_otsl_and_rejects_html(self):
        from paddleocr_vl_tasks import validate_target_for_task

        validate_target_for_task("<fcel>A<nl>", "table")
        with self.assertRaisesRegex(ValueError, "OTSL|HTML"):
            validate_target_for_task("<table><tr><td>A</td></tr></table>", "table")
        with self.assertRaisesRegex(ValueError, "same number"):
            validate_target_for_task("<fcel>A<nl><fcel>B<ecel><nl>", "table")
        with self.assertRaisesRegex(ValueError, "left"):
            validate_target_for_task("<lcel><nl>", "table")
        with self.assertRaisesRegex(ValueError, "rectangle|overlap|merged"):
            validate_target_for_task(
                "<fcel>A<lcel><lcel><nl><ucel><xcel><fcel>B<nl>",
                "table",
            )

    def test_chart_target_requires_a_rectangular_markdown_table(self):
        from paddleocr_vl_tasks import validate_target_for_task

        validate_target_for_task(
            "| Nhãn | Giá trị |\n| :--- | ---: |\n| A \\| B | 12 |",
            "chart",
        )
        with self.assertRaisesRegex(ValueError, "separator"):
            validate_target_for_task("| A | B |\n| C | D |\n| 1 | 2 |", "chart")
        with self.assertRaisesRegex(ValueError, "same number"):
            validate_target_for_task("| A | B |\n| --- | --- |\n| 1 |", "chart")

    def test_prepared_run_accepts_mixed_and_legacy_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mixed_run = create_prepared_run(root, ["table", "formula", "chart"])
            loaded = finetune_vl.load_prepared_run(mixed_run, root / "mixed-train")
            self.assertEqual(loaded["task"], "mixed")
            self.assertEqual(loaded["tasks"], ["chart", "formula", "table"])
            self.assertEqual(
                loaded["prompts"],
                [
                    TASK_PROMPTS["chart"],
                    TASK_PROMPTS["formula"],
                    TASK_PROMPTS["table"],
                ],
            )

            legacy_run = create_prepared_run(root, ["ocr"], legacy=True)
            legacy = finetune_vl.load_prepared_run(legacy_run, root / "ocr-train")
            self.assertEqual(legacy["task"], "ocr")
            self.assertEqual(legacy["prompt"], "OCR:")

    def test_evaluator_uses_prompt_embedded_in_each_row(self):
        prompt = TASK_PROMPTS["chart"]
        self.assertEqual(
            evaluate_paddleocr_vl.ocr_messages("chart.png", prompt)[0]["content"][1],
            {"type": "text", "text": prompt},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chart.png").write_bytes(png_bytes())
            validation = root / "validation-chart.jsonl"
            finetune_vl.write_erniekit_jsonl(
                validation,
                [
                    finetune_vl.PreparedSample(
                        "chart.png",
                        "| A | B |\n| --- | --- |\n| 1 | 2 |",
                        0,
                        prompt,
                    )
                ],
            )
            rows = evaluate_paddleocr_vl.load_validation_rows([validation], 1)
            self.assertEqual(
                rows[0]["target"], "| A | B |\n| --- | --- |\n| 1 | 2 |"
            )
            self.assertEqual(rows[0]["prompt"], prompt)

            bad = root / "validation-bad.jsonl"
            payload = json.loads(validation.read_text(encoding="utf-8"))
            payload["text_info"][0]["text"] = "Not A Real Prompt:"
            bad.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt"):
                evaluate_paddleocr_vl.load_validation_rows([bad], 1)

    def test_prepare_mixed_layout_from_dataset_task_column(self):
        rows = [
            {
                "image": {"bytes": png_bytes()},
                "text": "<fcel>A<nl>",
                "task": "table",
            },
            {
                "image": {"bytes": png_bytes()},
                "text": r"E = mc^2",
                "task": "formula",
            },
            {
                "image": {"bytes": png_bytes()},
                "text": "| Năm | Doanh thu |\n|---|---:|\n| 2026 | 10 |",
                "task": "chart",
            },
            {
                "image": {"bytes": png_bytes()},
                "text": "<fcel>B<nl>",
                "task": "table",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            work_dir = root / "run-mixed"
            with (
                patch.object(
                    finetune_vl, "load_tokenizer", return_value=CharacterTokenizer()
                ),
                patch.object(
                    finetune_vl, "load_hf_dataset", return_value=FakeSplit(rows)
                ),
            ):
                result = finetune_vl.main(
                    [
                        "--dataset-dir",
                        str(dataset_dir),
                        "--work-dir",
                        str(work_dir),
                        "--prepare-only",
                        "--min-pixels",
                        "1568",
                        "--max-pixels",
                        "1568",
                        "--max-seq-len",
                        "1000",
                        "--validation-ratio",
                        "0.25",
                    ]
                )

            summary = json.loads((work_dir / "summary.json").read_text(encoding="utf-8"))
            train_rows = [
                json.loads(line)
                for line in Path(summary["sources"][0]["train_jsonl"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            validation_rows = [
                json.loads(line)
                for line in Path(summary["sources"][0]["validation_jsonl"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(result, 0)
        self.assertEqual(summary["task"], "mixed")
        self.assertEqual(summary["tasks"], ["chart", "formula", "table"])
        observed_prompts = {
            row["text_info"][0]["text"] for row in [*train_rows, *validation_rows]
        }
        self.assertEqual(
            observed_prompts,
            {
                TASK_PROMPTS["table"],
                TASK_PROMPTS["formula"],
                TASK_PROMPTS["chart"],
            },
        )
        chart_targets = [
            row["text_info"][1]["text"]
            for row in [*train_rows, *validation_rows]
            if row["text_info"][0]["text"] == TASK_PROMPTS["chart"]
        ]
        self.assertEqual(
            chart_targets,
            ["| Năm | Doanh thu |\n|---|---:|\n| 2026 | 10 |"],
        )


if __name__ == "__main__":
    unittest.main()

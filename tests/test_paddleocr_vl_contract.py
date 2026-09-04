import unittest

from paddleocr_vl_contract import (
    LAYOUT_TO_TASK,
    PP_DOCLAYOUTV3_LABELS,
    SKIP_LAYOUT_LABELS,
    TASK_PROMPTS,
    normalize_summary_tasks,
    normalize_target_text,
    task_for_layout_label,
    validate_erniekit_record_contract,
)
from paddleocr_vl_tasks import TASK_PROMPTS as LEGACY_TASK_PROMPTS
from paddleocr_vl_tasks import validate_target_for_task as legacy_validator


class PaddleOCRVLContractTests(unittest.TestCase):
    def test_legacy_facade_uses_canonical_task_prompts_and_validator(self):
        self.assertIs(LEGACY_TASK_PROMPTS, TASK_PROMPTS)
        legacy_validator("<fcel>value<nl>", "table")

    def test_layout_taxonomy_and_mapping_share_one_contract(self):
        self.assertEqual(len(PP_DOCLAYOUTV3_LABELS), 25)
        self.assertEqual(task_for_layout_label("display_formula"), "formula")
        self.assertEqual(task_for_layout_label("image"), None)
        self.assertEqual(set(SKIP_LAYOUT_LABELS), set(PP_DOCLAYOUTV3_LABELS) - set(LAYOUT_TO_TASK))

    def test_normalize_target_text_prefers_nonempty_label_and_normalizes_newlines(self):
        self.assertEqual(
            normalize_target_text({"label": "  Một\r\nHai\r  ", "text": "unused"}),
            "  Một\nHai\n  ",
        )
        self.assertEqual(normalize_target_text({"label": "", "text": "Ba"}), "Ba")
        self.assertEqual(normalize_target_text({"label": "  ", "text": "  "}), "")

    def test_summary_normalization_migrates_legacy_ocr_and_mixed_tasks(self):
        legacy = {}
        self.assertEqual(normalize_summary_tasks(legacy), ["ocr"])
        self.assertEqual(legacy["prompt"], "OCR:")

        mixed = {"task": "mixed", "tasks": ["table", "ocr", "table"]}
        self.assertEqual(normalize_summary_tasks(mixed), ["ocr", "table"])
        self.assertEqual(mixed["prompts"], ["OCR:", "Table Recognition:"])
        self.assertNotIn("prompt", mixed)

    def test_erniekit_record_contract_returns_task_and_image_reference(self):
        task, image_url = validate_erniekit_record_contract(
            [{"matched_text_index": 0, "image_url": "images/page.png"}],
            [
                {"tag": "mask", "text": "Table Recognition:"},
                {"tag": "no_mask", "text": "<fcel>A<nl>"},
            ],
            ["Table Recognition:"],
        )
        self.assertEqual(task, "table")
        self.assertEqual(image_url, "images/page.png")

        with self.assertRaisesRegex(ValueError, "task mask contract"):
            validate_erniekit_record_contract(
                [{"matched_text_index": 0, "image_url": "images/page.png"}],
                [
                    {"tag": "mask", "text": "OCR:"},
                    {"tag": "no_mask", "text": "<table>bad</table>"},
                ],
                ["Table Recognition:"],
            )

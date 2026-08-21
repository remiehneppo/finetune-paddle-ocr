"""Map PP-DocLayoutV3 class labels to PaddleOCR-VL training tasks."""

from __future__ import annotations

from paddleocr_vl_tasks import TASK_PROMPTS, prompt_for_task

PP_DOCLAYOUTV3_LABELS: tuple[str, ...] = (
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
)
PP_DOCLAYOUTV3_LABEL_SET = frozenset(PP_DOCLAYOUTV3_LABELS)

LAYOUT_TO_TASK: dict[str, str] = {
    "table": "table",
    "chart": "chart",
    "display_formula": "formula",
    "inline_formula": "formula",
    "text": "ocr",
    "vertical_text": "ocr",
    "content": "ocr",
    "doc_title": "ocr",
    "paragraph_title": "ocr",
    "aside_text": "ocr",
    "abstract": "ocr",
    "reference": "ocr",
    "reference_content": "ocr",
    "footnote": "ocr",
    "header": "ocr",
    "footer": "ocr",
}

SKIP_LAYOUT_LABELS: frozenset[str] = frozenset(
    label for label in PP_DOCLAYOUTV3_LABELS if label not in LAYOUT_TO_TASK
)

SUPPORTED_TASKS = tuple(TASK_PROMPTS.keys())


def map_layout_label(layout_label: str) -> str | None:
    """Return the VL task for a known layout class, else ``None``."""
    label = layout_label.strip()
    if label not in PP_DOCLAYOUTV3_LABEL_SET:
        return None
    return LAYOUT_TO_TASK.get(label)


def is_skippable_layout_label(layout_label: str) -> bool:
    return layout_label.strip() in SKIP_LAYOUT_LABELS


def prompt_for_block_task(task: str) -> str:
    return prompt_for_task(task)


__all__ = [
    "LAYOUT_TO_TASK",
    "PP_DOCLAYOUTV3_LABELS",
    "PP_DOCLAYOUTV3_LABEL_SET",
    "SKIP_LAYOUT_LABELS",
    "SUPPORTED_TASKS",
    "is_skippable_layout_label",
    "map_layout_label",
    "prompt_for_block_task",
]

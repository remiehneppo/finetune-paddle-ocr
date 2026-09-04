"""Backward-compatible layout mapping facade for PaddleOCR-VL labeler code."""

from __future__ import annotations

from paddleocr_vl_contract import (
    LAYOUT_TO_TASK,
    PP_DOCLAYOUTV3_LABELS,
    PP_DOCLAYOUTV3_LABEL_SET,
    SKIP_LAYOUT_LABELS,
    TASK_PROMPTS,
    is_skippable_layout_label,
    prompt_for_task,
    task_for_layout_label,
)

SUPPORTED_TASKS = tuple(TASK_PROMPTS)
map_layout_label = task_for_layout_label


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

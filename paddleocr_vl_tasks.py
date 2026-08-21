"""Shared PaddleOCR-VL task prompt definitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TASK_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
LAYOUT_TASKS = ("table", "formula", "chart")
PROMPT_TO_TASK = {prompt: task for task, prompt in TASK_PROMPTS.items()}


def prompt_for_task(task: str) -> str:
    try:
        return TASK_PROMPTS[task]
    except KeyError as exc:
        choices = ", ".join(TASK_PROMPTS)
        raise ValueError(
            f"Unsupported PaddleOCR-VL task {task!r}; choose {choices}"
        ) from exc


def task_for_prompt(prompt: str) -> str:
    try:
        return PROMPT_TO_TASK[prompt]
    except KeyError as exc:
        choices = ", ".join(TASK_PROMPTS.values())
        raise ValueError(
            f"Unsupported PaddleOCR-VL prompt {prompt!r}; choose {choices}"
        ) from exc


def resolve_row_task(row: Mapping[str, Any], default_task: str = "ocr") -> str:
    value = row.get("task")
    if value is None or value == "":
        return default_task
    if not isinstance(value, str):
        raise TypeError(f"Dataset task must be a string, got {type(value).__name__}")
    task = value.strip()
    prompt_for_task(task)
    return task

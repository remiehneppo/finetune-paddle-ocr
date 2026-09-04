"""Canonical PaddleOCR-VL task, target, layout, and prepared-data contract."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

PaddleOCRVLTask = Literal["ocr", "table", "formula", "chart"]


@dataclass(frozen=True)
class TaskTargetSpec:
    task: PaddleOCRVLTask
    prompt: str


TASK_SPECS: tuple[TaskTargetSpec, ...] = (
    TaskTargetSpec("ocr", "OCR:"),
    TaskTargetSpec("table", "Table Recognition:"),
    TaskTargetSpec("formula", "Formula Recognition:"),
    TaskTargetSpec("chart", "Chart Recognition:"),
)
TASK_PROMPTS = {spec.task: spec.prompt for spec in TASK_SPECS}
LAYOUT_TASKS = tuple(spec.task for spec in TASK_SPECS if spec.task != "ocr")
PROMPT_TO_TASK = {spec.prompt: spec.task for spec in TASK_SPECS}

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
LAYOUT_TO_TASK: dict[str, PaddleOCRVLTask] = {
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
SKIP_LAYOUT_LABELS = frozenset(
    label for label in PP_DOCLAYOUTV3_LABELS if label not in LAYOUT_TO_TASK
)

OTSL_CELL_TOKENS = ("<fcel>", "<ecel>", "<xcel>", "<lcel>", "<ucel>")
OTSL_TAGS = frozenset((*OTSL_CELL_TOKENS, "<nl>"))
HTML_TABLE_PATTERN = re.compile(r"</?(?:table|thead|tbody|tfoot|tr|th|td)\b", re.I)
ANGLE_TAG_PATTERN = re.compile(r"<[^>\n]+>")
OTSL_TOKEN_PATTERN = re.compile(r"<(?:fcel|ecel|lcel|ucel|xcel|nl)>")
MARKDOWN_ALIGNMENT_PATTERN = re.compile(r"^:?-{3,}:?$")


def prompt_for_task(task: str) -> str:
    try:
        return TASK_PROMPTS[task]
    except KeyError as exc:
        choices = ", ".join(TASK_PROMPTS)
        raise ValueError(
            f"Unsupported PaddleOCR-VL task {task!r}; choose {choices}"
        ) from exc


def task_for_prompt(prompt: str) -> PaddleOCRVLTask:
    try:
        return PROMPT_TO_TASK[prompt]
    except KeyError as exc:
        choices = ", ".join(TASK_PROMPTS.values())
        raise ValueError(
            f"Unsupported PaddleOCR-VL prompt {prompt!r}; choose {choices}"
        ) from exc


def task_for_layout_label(layout_label: str) -> PaddleOCRVLTask | None:
    label = layout_label.strip()
    if label not in PP_DOCLAYOUTV3_LABEL_SET:
        return None
    return LAYOUT_TO_TASK.get(label)


def is_skippable_layout_label(layout_label: str) -> bool:
    return layout_label.strip() in SKIP_LAYOUT_LABELS


def resolve_row_task(
    row: Mapping[str, Any], default_task: str = "ocr"
) -> PaddleOCRVLTask:
    value = row.get("task")
    if value is None or value == "":
        prompt_for_task(default_task)
        return default_task  # type: ignore[return-value]
    if not isinstance(value, str):
        raise TypeError(f"Dataset task must be a string, got {type(value).__name__}")
    task = value.strip()
    prompt_for_task(task)
    return task  # type: ignore[return-value]


def normalize_target_text(row: Mapping[str, Any]) -> str:
    value: str | None = None
    for column in ("label", "text"):
        candidate = row.get(column)
        if isinstance(candidate, str) and candidate.strip():
            value = candidate
            break
    if value is None:
        return ""
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return value if value.strip() else ""


def _parse_otsl(text: str) -> list[list[str]]:
    if HTML_TABLE_PATTERN.search(text):
        raise ValueError("table target must use OTSL, not HTML")
    unknown_tags = sorted(set(ANGLE_TAG_PATTERN.findall(text)) - OTSL_TAGS)
    if unknown_tags:
        raise ValueError(
            "table target contains non-OTSL tags: " + ", ".join(unknown_tags)
        )
    matches = list(OTSL_TOKEN_PATTERN.finditer(text))
    if not matches:
        raise ValueError("table target must contain an OTSL cell token")
    if matches[0].start() != 0:
        raise ValueError("OTSL target must start with a cell token")
    rows: list[list[str]] = []
    row: list[str] = []
    has_content = False
    for index, match in enumerate(matches):
        token = match.group(0)
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.end() : next_start]
        if token == "<nl>":
            if content.strip():
                raise ValueError("OTSL must not contain content after <nl>")
            if not row:
                raise ValueError("OTSL must not contain an empty row")
            rows.append(row)
            row = []
            continue
        if token != "<fcel>" and content.strip():
            raise ValueError(f"{token} must not contain text")
        if token == "<fcel>" and content.strip():
            has_content = True
        row.append(token)
    if row:
        raise ValueError("every OTSL row must end with <nl>")
    if not rows:
        raise ValueError("OTSL target must contain at least one row")
    width = len(rows[0])
    if any(len(candidate) != width for candidate in rows):
        raise ValueError("all OTSL rows must contain the same number of cells")
    for row_index, cells in enumerate(rows):
        for column_index, token in enumerate(cells):
            left = cells[column_index - 1] if column_index else None
            above = rows[row_index - 1][column_index] if row_index else None
            if token == "<lcel>" and left not in {"<fcel>", "<ecel>", "<lcel>"}:
                raise ValueError("<lcel> must extend a cell on its left")
            if token == "<ucel>" and above not in {"<fcel>", "<ecel>", "<ucel>"}:
                raise ValueError("<ucel> must extend a cell above it")
            if token == "<xcel>" and not (
                left in {"<ucel>", "<xcel>"}
                and above in {"<lcel>", "<xcel>"}
            ):
                raise ValueError("<xcel> must be inside a two-dimensional merged cell")
    covered: set[tuple[int, int]] = set()
    for row_index, cells in enumerate(rows):
        for column_index, token in enumerate(cells):
            if token not in {"<fcel>", "<ecel>"}:
                continue
            colspan = 1
            while column_index + colspan < width and cells[column_index + colspan] in {
                "<lcel>",
                "<xcel>",
            }:
                colspan += 1
            rowspan = 1
            while row_index + rowspan < len(rows) and rows[row_index + rowspan][
                column_index
            ] in {"<ucel>", "<xcel>"}:
                rowspan += 1
            for row_offset in range(rowspan):
                for column_offset in range(colspan):
                    position = (row_index + row_offset, column_index + column_offset)
                    if position in covered:
                        raise ValueError("OTSL merged-cell rectangles must not overlap")
                    covered.add(position)
                    expected = token
                    if row_offset == 0 and column_offset > 0:
                        expected = "<lcel>"
                    elif row_offset > 0 and column_offset == 0:
                        expected = "<ucel>"
                    elif row_offset > 0 and column_offset > 0:
                        expected = "<xcel>"
                    actual = rows[position[0]][position[1]]
                    if actual != expected:
                        raise ValueError(
                            "OTSL merged-cell tokens must form complete rectangles"
                        )
    if len(covered) != len(rows) * width:
        raise ValueError("OTSL merged-cell tokens must belong to a cell rectangle")
    if not has_content:
        raise ValueError("OTSL target must contain text in at least one <fcel>")
    return rows


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if "|" not in stripped:
        raise ValueError("each Markdown table row must contain |")
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith(r"\|"):
        stripped = stripped[:-1]
    cells: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(stripped):
        character = stripped[index]
        following = stripped[index + 1] if index + 1 < len(stripped) else None
        if character == "\\" and following in {"|", "\\"}:
            current.append(following)
            index += 2
            continue
        if character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
        index += 1
    cells.append("".join(current).strip())
    return cells


def _validate_markdown_table(text: str) -> None:
    lines = text.split("\n")
    if any(not line.strip() for line in lines):
        raise ValueError("Markdown table must not contain blank rows")
    if len(lines) < 3:
        raise ValueError(
            "Markdown table requires a header, separator, and at least one data row"
        )
    rows = [_split_markdown_row(line) for line in lines]
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError("all Markdown table rows must contain the same number of cells")
    if any(not MARKDOWN_ALIGNMENT_PATTERN.fullmatch(cell) for cell in rows[1]):
        raise ValueError("Markdown table separator cells must use --- with optional :")
    if not any(cell.strip() for row in [rows[0], *rows[2:]] for cell in row):
        raise ValueError("Markdown table must contain data")


def validate_target_for_task(text: str, task: str) -> None:
    prompt_for_task(task)
    if not text.strip():
        raise ValueError("target must not be empty")
    if task == "table":
        _parse_otsl(text)
    elif task == "chart":
        _validate_markdown_table(text)
    elif task == "formula" and HTML_TABLE_PATTERN.search(text):
        raise ValueError("formula target must be LaTeX, not HTML")


def normalize_summary_tasks(summary: dict[str, Any]) -> list[PaddleOCRVLTask]:
    raw_tasks = summary.get("tasks")
    if isinstance(raw_tasks, list) and raw_tasks:
        normalized: list[str] = []
        for task in raw_tasks:
            if not isinstance(task, str):
                raise TypeError("Prepared summary tasks must be strings")
            prompt_for_task(task)
            normalized.append(task)
        tasks = sorted(set(normalized))
    else:
        legacy_task = summary.get("task", "ocr")
        if legacy_task == "mixed":
            raise ValueError(
                "Prepared summary task='mixed' requires a non-empty tasks list"
            )
        if not isinstance(legacy_task, str):
            raise TypeError("Prepared summary task must be a string")
        prompt_for_task(legacy_task)
        tasks = [legacy_task]

    prompts = [prompt_for_task(task) for task in tasks]
    if "prompt" in summary and summary["prompt"] not in prompts:
        raise ValueError(
            f"Prepared summary prompt {summary['prompt']!r} does not match tasks {tasks}"
        )
    if "prompts" in summary:
        recorded = summary["prompts"]
        if not isinstance(recorded, list) or set(recorded) != set(prompts):
            raise ValueError("Prepared summary prompts do not match tasks")
    summary["tasks"] = tasks
    summary["prompts"] = prompts
    if len(tasks) == 1:
        summary["task"] = tasks[0]
        summary["prompt"] = prompts[0]
    else:
        summary["task"] = "mixed"
        summary.pop("prompt", None)
    return tasks  # type: ignore[return-value]


def validate_erniekit_record_contract(
    image_info: Any,
    text_info: Any,
    allowed_prompts: Sequence[str],
) -> tuple[PaddleOCRVLTask, str]:
    if not isinstance(image_info, list) or len(image_info) != 1:
        raise ValueError("Invalid image_info contract")
    if not isinstance(text_info, list) or len(text_info) != 2:
        raise ValueError("Invalid text_info contract")
    image = image_info[0]
    prompt_row, target = text_info
    if (
        not isinstance(image, Mapping)
        or image.get("matched_text_index") != 0
        or not isinstance(prompt_row, Mapping)
        or prompt_row.get("tag") != "mask"
        or not isinstance(prompt_row.get("text"), str)
        or prompt_row["text"] not in set(allowed_prompts)
        or not isinstance(target, Mapping)
        or target.get("tag") != "no_mask"
        or not isinstance(target.get("text"), str)
        or not target["text"]
    ):
        raise ValueError("Invalid task mask contract")
    task = task_for_prompt(prompt_row["text"])
    try:
        validate_target_for_task(target["text"], task)
    except ValueError as exc:
        raise ValueError(f"Invalid {task} target schema: {exc}") from exc
    image_url = image.get("image_url")
    if not isinstance(image_url, str) or not image_url:
        raise ValueError("Invalid task mask contract")
    return task, image_url


__all__ = [
    "ANGLE_TAG_PATTERN",
    "HTML_TABLE_PATTERN",
    "LAYOUT_TASKS",
    "LAYOUT_TO_TASK",
    "OTSL_CELL_TOKENS",
    "OTSL_TAGS",
    "OTSL_TOKEN_PATTERN",
    "PaddleOCRVLTask",
    "PP_DOCLAYOUTV3_LABELS",
    "PP_DOCLAYOUTV3_LABEL_SET",
    "PROMPT_TO_TASK",
    "SKIP_LAYOUT_LABELS",
    "TASK_PROMPTS",
    "TASK_SPECS",
    "TaskTargetSpec",
    "is_skippable_layout_label",
    "normalize_summary_tasks",
    "normalize_target_text",
    "prompt_for_task",
    "resolve_row_task",
    "task_for_layout_label",
    "task_for_prompt",
    "validate_erniekit_record_contract",
    "validate_target_for_task",
]

"""Shared PaddleOCR-VL task prompt definitions."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

TASK_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
LAYOUT_TASKS = ("table", "formula", "chart")
PROMPT_TO_TASK = {prompt: task for task, prompt in TASK_PROMPTS.items()}
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
        next_start = (
            matches[index + 1].start() if index + 1 < len(matches) else len(text)
        )
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
            while (
                column_index + colspan < width
                and cells[column_index + colspan] in {"<lcel>", "<xcel>"}
            ):
                colspan += 1
            rowspan = 1
            while (
                row_index + rowspan < len(rows)
                and rows[row_index + rowspan][column_index]
                in {"<ucel>", "<xcel>"}
            ):
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
        raise ValueError(
            "all Markdown table rows must contain the same number of cells"
        )
    if any(not MARKDOWN_ALIGNMENT_PATTERN.fullmatch(cell) for cell in rows[1]):
        raise ValueError("Markdown table separator cells must use --- with optional :")
    if not any(cell.strip() for row in [rows[0], *rows[2:]] for cell in row):
        raise ValueError("Markdown table must contain data")


def validate_target_for_task(text: str, task: str) -> None:
    """Validate the official PaddleOCR-VL target format for one task."""
    prompt_for_task(task)
    if not text.strip():
        raise ValueError("target must not be empty")
    if task == "table":
        _parse_otsl(text)
    elif task == "chart":
        _validate_markdown_table(text)
    elif task == "formula" and HTML_TABLE_PATTERN.search(text):
        raise ValueError("formula target must be LaTeX, not HTML")

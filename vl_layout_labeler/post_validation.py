from __future__ import annotations

from datetime import datetime
import json
from typing import Protocol
from uuid import UUID

import httpx
from pydantic import ValidationError

from .models import Annotation, OCRValidation, Task, ValidationIssue


VALIDATION_TASKS = frozenset({"ocr", "table", "chart"})


class OCRPostValidationError(RuntimeError):
    """A sanitized validation failure safe to return to the labeler client."""


class NoEligibleBlocks(ValueError):
    pass


class OCRPostValidator(Protocol):
    @property
    def model(self) -> str: ...

    def validate_block(
        self, block_id: UUID, task: Task, text: str
    ) -> list[ValidationIssue]: ...

    def close(self) -> None: ...


_ISSUE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "block_id", "start", "end", "text", "category", "reason", "suggestion"
    ],
    "properties": {
        "block_id": {"type": "string", "format": "uuid"},
        "start": {"type": "integer", "minimum": 0},
        "end": {"type": "integer", "minimum": 1},
        "text": {"type": "string", "minLength": 1},
        "category": {"type": "string", "minLength": 1},
        "reason": {"type": "string", "minLength": 1},
        "suggestion": {"type": "string", "minLength": 1},
    },
}

VALIDATION_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["issues"],
    "properties": {"issues": {"type": "array", "items": _ISSUE_SCHEMA}},
}


class OpenAICompatiblePostValidator:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_tokens: int = 2048,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.max_tokens = max_tokens
        self._owns_client = client is None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self.client = client or httpx.Client(timeout=timeout, headers=headers)

    @classmethod
    def from_settings(cls, settings):
        return cls(
            base_url=settings.validation_base_url,
            model=settings.validation_model,
            api_key=settings.validation_api_key,
            timeout=settings.validation_timeout,
            max_tokens=settings.validation_max_tokens,
        )

    @property
    def model(self) -> str:
        return self._model

    def _payload(self, block_id: UUID, task: Task, text: str) -> dict:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Review OCR text for likely recognition errors. Do not rewrite it. "
                        "For table and chart input, ignore OTSL or Markdown syntax and report "
                        "only suspicious natural-language spans. All offsets refer to the raw "
                        "input. Return no issue when uncertain."
                    ),
                },
                {
                    "role": "user",
                    "content": f"block_id={block_id}\ntask={task}\nraw_text:\n{text}",
                },
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ocr_post_validation",
                    "strict": True,
                    "schema": VALIDATION_RESPONSE_SCHEMA,
                },
            },
        }

    @staticmethod
    def _parse(content, block_id: UUID, text: str) -> list[ValidationIssue]:
        if not isinstance(content, str):
            raise OCRPostValidationError("LLM validation returned malformed JSON")
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise OCRPostValidationError("LLM validation returned malformed JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("issues"), list):
            raise OCRPostValidationError("LLM validation violated the response contract")
        parsed = []
        try:
            for raw_issue in payload["issues"]:
                if not isinstance(raw_issue, dict):
                    raise ValueError
                issue_block_id = UUID(str(raw_issue.get("block_id")))
                start = raw_issue.get("start")
                end = raw_issue.get("end")
                excerpt = raw_issue.get("text")
                if (
                    issue_block_id != block_id
                    or not isinstance(start, int)
                    or isinstance(start, bool)
                    or not isinstance(end, int)
                    or isinstance(end, bool)
                    or start < 0
                    or end <= start
                    or end > len(text)
                    or not isinstance(excerpt, str)
                    or text[start:end] != excerpt
                ):
                    raise ValueError
                parsed.append(
                    ValidationIssue.model_validate(
                        {key: value for key, value in raw_issue.items() if key != "text"}
                    )
                )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise OCRPostValidationError(
                "LLM validation violated the semantic contract"
            ) from exc
        return parsed

    def validate_block(
        self, block_id: UUID, task: Task, text: str
    ) -> list[ValidationIssue]:
        payload = self._payload(block_id, task, text)
        for attempt in range(2):
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                raise OCRPostValidationError("LLM validation timed out") from exc
            except httpx.HTTPError as exc:
                raise OCRPostValidationError("LLM validation request failed") from exc
            try:
                content = response.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise OCRPostValidationError(
                    "LLM validation returned malformed JSON"
                ) from exc
            try:
                return self._parse(content, block_id, text)
            except OCRPostValidationError as exc:
                if "semantic contract" not in str(exc) or attempt == 1:
                    raise
        raise OCRPostValidationError("LLM validation failed")

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


class ValidationService:
    def __init__(self, validator: OCRPostValidator):
        self.validator = validator

    @property
    def model(self) -> str:
        return self.validator.model

    def validate_annotation(
        self, annotation: Annotation, block_ids: list[UUID] | None = None
    ) -> Annotation:
        selected = set(block_ids) if block_ids is not None else None
        if selected is not None:
            existing = {block.id for block in annotation.blocks}
            if not selected <= existing:
                raise ValueError("one or more selected blocks do not exist")
        eligible = [
            block
            for block in annotation.blocks
            if (selected is None or block.id in selected)
            and not block.skipped
            and block.task in VALIDATION_TASKS
            and bool(block.text.strip())
        ]
        if not eligible:
            raise NoEligibleBlocks(
                "no eligible OCR, table, or chart blocks with text to validate"
            )
        validations = {}
        for block in eligible:
            issues = self.validator.validate_block(block.id, block.task, block.text)
            validations[block.id] = OCRValidation(
                text_hash=block.current_text_hash(),
                model=self.model,
                checked_at=datetime.now().astimezone(),
                issues=issues,
            )
        return annotation.model_copy(
            update={
                "blocks": [
                    block.model_copy(update={"validation": validations[block.id]})
                    if block.id in validations
                    else block
                    for block in annotation.blocks
                ]
            }
        )

    def close(self) -> None:
        self.validator.close()

from __future__ import annotations

from hashlib import sha256
import math
from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
from paddleocr_vl_tasks import validate_target_for_task

Point = tuple[float, float]
LayoutSource = Literal["layout", "vl", "manual"]
AnnotationStatus = Literal["draft", "detected", "edited", "completed"]
Task = Literal["ocr", "table", "formula", "chart"]


class ValidationIssue(BaseModel):
    block_id: UUID
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    category: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    suggestion: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_span(self) -> ValidationIssue:
        if self.end <= self.start:
            raise ValueError("validation issue end must be greater than start")
        return self


class OCRValidation(BaseModel):
    text_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    checked_at: datetime
    issues: list[ValidationIssue] = Field(default_factory=list)


class ImageInfo(BaseModel):
    path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Block(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    order: int = Field(ge=0)
    polygon: list[Point]
    layout_label: str
    task: Task | None = None
    text: str = ""
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    source: LayoutSource = "layout"
    skipped: bool = False
    validation: OCRValidation | None = None

    def current_text_hash(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def invalidate_stale_validation(self) -> Block:
        if (
            self.validation is not None
            and self.validation.text_hash != self.current_text_hash()
        ):
            self.validation = None
        return self

    @field_validator("layout_label")
    @classmethod
    def validate_layout_label(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("layout_label must not be empty")
        return value

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value: list[Point]) -> list[Point]:
        if len(value) != 4:
            raise ValueError("polygon must contain exactly four points")
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in value):
            raise ValueError("polygon coordinates must be finite")
        return value


class Annotation(BaseModel):
    version: Literal[2] = 2
    image: ImageInfo
    revision: int = Field(default=0, ge=0)
    status: AnnotationStatus = "draft"
    text: str = ""
    blocks: list[Block] = Field(default_factory=list)
    updated_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_v1(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("version") == 1:
            value = dict(value)
            value["version"] = 2
            migrated_blocks = []
            for raw_block in value.get("blocks", []):
                block = dict(raw_block)
                if block.get("layout_label") == "manual":
                    block["layout_label"] = "text"
                migrated_blocks.append(block)
            value["blocks"] = migrated_blocks
            if value.get("status") == "completed" and not any(
                not block.get("skipped", False) for block in migrated_blocks
            ):
                value["status"] = "edited"
        return value

    @model_validator(mode="after")
    def validate_annotation(self) -> Annotation:
        ids = [block.id for block in self.blocks]
        if len(ids) != len(set(ids)):
            raise ValueError("block ids must be unique")
        if self.status == "completed":
            from .task_map import PP_DOCLAYOUTV3_LABEL_SET

            active = [block for block in self.blocks if not block.skipped]
            if not active:
                raise ValueError("completed annotations require at least one active block")
            for block in active:
                if block.layout_label not in PP_DOCLAYOUTV3_LABEL_SET:
                    raise ValueError("completed annotations require taxonomy layout labels")
                polygon = [
                    (
                        min(max(float(x), 0.0), float(max(self.image.width - 1, 0))),
                        min(max(float(y), 0.0), float(max(self.image.height - 1, 0))),
                    )
                    for x, y in block.polygon
                ]
                area = abs(
                    sum(
                        polygon[index][0] * polygon[(index + 1) % 4][1]
                        - polygon[(index + 1) % 4][0] * polygon[index][1]
                        for index in range(4)
                    )
                ) / 2.0
                if area <= 0:
                    raise ValueError("completed annotations require positive-area polygons")
                if block.task is not None and not block.text.strip():
                    raise ValueError("completed VL blocks require text")
                if block.task is not None:
                    try:
                        validate_target_for_task(block.text, block.task)
                    except ValueError as exc:
                        raise ValueError(
                            f"completed {block.task} block has invalid target: {exc}"
                        ) from exc
        return self


class DetectRequest(BaseModel):
    replace_existing: bool = False


class PrelabelRequest(BaseModel):
    block_ids: list[UUID] | None = None
    replace_existing: bool = True
    post_validate: bool = False


class ValidateRequest(BaseModel):
    block_ids: list[UUID] | None = None


class BatchRequest(BaseModel):
    post_validate: bool = False


class ExportRequest(BaseModel):
    output_dir: str

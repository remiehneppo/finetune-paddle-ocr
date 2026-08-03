from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

Point = tuple[float, float]
AnnotationStatus = Literal["ocr", "edited", "completed"]
BlockSource = Literal["ocr", "manual"]


class ImageInfo(BaseModel):
    path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Block(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    order: int = Field(ge=0)
    text: str
    polygon: list[Point]
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    source: BlockSource

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value: list[Point]) -> list[Point]:
        import math

        if len(value) != 4:
            raise ValueError("polygon must contain exactly four points")
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in value):
            raise ValueError("polygon coordinates must be finite")
        return value

    @model_validator(mode="after")
    def validate_source_score(self):
        if self.source == "manual" and self.score is not None:
            raise ValueError("manual blocks cannot have a score")
        if self.source == "ocr" and self.score is None:
            raise ValueError("ocr blocks require a score")
        return self


class OCRMetadata(BaseModel):
    task: Literal["ocr", "detection"] = "ocr"
    det_model: str
    rec_model: str | None = None
    duration_ms: int = Field(ge=0)


class Annotation(BaseModel):
    version: Literal[1] = 1
    image: ImageInfo
    revision: int = Field(default=0, ge=0)
    status: AnnotationStatus = "edited"
    text: str = ""
    blocks: list[Block] = Field(default_factory=list)
    ocr: OCRMetadata | None = None
    updated_at: datetime | None = None

    @model_validator(mode="after")
    def validate_completed_text(self):
        block_ids = [block.id for block in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("block ids must be unique")
        if self.status == "completed" and any(
            not block.text.strip() for block in self.blocks
        ):
            raise ValueError("completed annotations cannot contain empty block text")
        return self

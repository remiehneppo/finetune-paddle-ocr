"""Detect, prelabel, and validation policy adapters for VL batch jobs."""

from __future__ import annotations

from dataclasses import dataclass

from .post_validation import NoEligibleBlocks, OCRPostValidationError


@dataclass(frozen=True)
class BatchOperationResult:
    processed: bool
    validation_error: str | None = None


class DetectBatchOperation:
    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator

    def execute(self, record, store, *, post_validate: bool = False) -> BatchOperationResult:
        del post_validate
        if record.error:
            raise ValueError(record.error)
        existing = store.load(record)
        if existing.status == "completed" or existing.blocks:
            return BatchOperationResult(processed=False)
        result = self.coordinator.detect(record).model_copy(
            update={"revision": existing.revision}
        )
        store.save(record, result)
        return BatchOperationResult(processed=True)


class PrelabelBatchOperation:
    def __init__(self, coordinator, validation_service=None) -> None:
        self.coordinator = coordinator
        self.validation_service = validation_service

    def execute(self, record, store, *, post_validate: bool = False) -> BatchOperationResult:
        if record.error:
            raise ValueError(record.error)
        existing = store.load(record)
        if existing.status == "completed":
            return BatchOperationResult(processed=False)
        active_blocks = [
            block
            for block in existing.blocks
            if not block.skipped and block.task is not None
        ]
        if not active_blocks or all(block.text.strip() for block in active_blocks):
            return BatchOperationResult(processed=False)
        result = self.coordinator.prelabel(record, existing, replace_existing=False)
        saved = store.save(record, result)
        if not post_validate:
            return BatchOperationResult(processed=True)
        try:
            if self.validation_service is None:
                raise RuntimeError("LLM validation is not configured")
            validated = self.validation_service.validate_annotation(saved)
            store.save(record, validated)
        except NoEligibleBlocks:
            pass
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, OCRPostValidationError)
                else "LLM validation failed"
            )
            return BatchOperationResult(processed=True, validation_error=message)
        return BatchOperationResult(processed=True)


__all__ = [
    "BatchOperationResult",
    "DetectBatchOperation",
    "PrelabelBatchOperation",
]

"""Recognition operation policy for OCR labeler batches."""

from __future__ import annotations


class RecognitionBatchOperation:
    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator

    def execute(self, record, store) -> bool:
        if record.error:
            raise ValueError(record.error)
        if store.has_annotation(record):
            store.load(record)
            return False
        annotation = self.coordinator.recognize(record)
        store.save(record, annotation)
        return True


__all__ = ["RecognitionBatchOperation"]

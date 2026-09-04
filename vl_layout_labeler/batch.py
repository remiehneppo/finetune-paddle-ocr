from __future__ import annotations

from dataclasses import asdict, dataclass, field
from threading import Lock

from batch_lifecycle import BatchLifecycleMixin
from .batch_operations import DetectBatchOperation, PrelabelBatchOperation


@dataclass(frozen=True)
class BatchError:
    image: str
    message: str


@dataclass
class BatchSnapshot:
    state: str = "idle"
    operation: str | None = None
    total: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    validation_failed: int = 0
    current_image: str | None = None
    errors: list[BatchError] = field(default_factory=list)
    validation_errors: list[BatchError] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class GPUCoordinator:
    def __init__(self, layout_engine, vl_client):
        self.layout_engine = layout_engine
        self.vl_client = vl_client
        self._lock = Lock()

    def detect(self, record):
        with self._lock:
            return self.layout_engine.detect(record)

    def prelabel(self, record, annotation, block_ids=None, replace_existing=True):
        selected = set(block_ids) if block_ids is not None else None
        if selected is not None:
            matching = [block for block in annotation.blocks if block.id in selected]
            if len(matching) != len(selected):
                raise ValueError("one or more selected blocks do not exist")
            if any(not block.skipped and block.task is None for block in matching):
                raise ValueError("layout-only blocks cannot be prelabelled")
        blocks = []
        changed = False
        with self._lock:
            for block in annotation.blocks:
                if selected is not None and block.id not in selected:
                    blocks.append(block)
                    continue
                if block.skipped:
                    blocks.append(block)
                    continue
                if block.task is None:
                    blocks.append(block)
                    continue
                if not replace_existing and block.text.strip():
                    blocks.append(block)
                    continue
                text = self.vl_client.prelabel(
                    record.path,
                    block.polygon,
                    block.task,
                    annotation.image.width,
                    annotation.image.height,
                )
                blocks.append(block.model_copy(update={"text": text, "source": "vl"}))
                changed = True
        status = "edited" if changed else annotation.status
        return annotation.model_copy(update={"blocks": blocks, "status": status})


class BatchManager(BatchLifecycleMixin):
    def __init__(self, coordinator: GPUCoordinator, validation_service=None):
        self.coordinator = coordinator
        self.validation_service = validation_service
        self.operations = {
            "detect": DetectBatchOperation(coordinator),
            "prelabel": PrelabelBatchOperation(coordinator, validation_service),
        }
        self._init_lifecycle(BatchSnapshot, BatchError)

    def start(
        self, operation: str, catalog, store, *, post_validate: bool = False
    ) -> BatchSnapshot:
        if operation not in {"detect", "prelabel"}:
            raise ValueError("unsupported batch operation")
        records = catalog.list_images()
        self._start_job(
            BatchSnapshot(state="queued", operation=operation, total=len(records)),
            self._run,
            (operation, records, store, post_validate),
        )
        return self.snapshot()

    def _run(self, operation, records, store, post_validate) -> None:
        try:
            if not self._begin():
                return
            operation_adapter = self.operations[operation]
            for record in records:
                if not self._claim(record.name):
                    return
                try:
                    result = operation_adapter.execute(
                        record, store, post_validate=post_validate
                    )
                    self._increment("processed" if result.processed else "skipped")
                    if result.validation_error is not None:
                        with self._state_lock:
                            self._snapshot.validation_failed += 1
                            self._snapshot.validation_errors.append(
                                BatchError(record.name, result.validation_error)
                            )
                except Exception as exc:
                    self._add_error(record.name, str(exc))
            self._finish()
        except Exception as exc:
            self._fail(exc)

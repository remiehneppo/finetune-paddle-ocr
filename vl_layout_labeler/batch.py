from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from threading import Event, Lock, Thread


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
    current_image: str | None = None
    errors: list[BatchError] = field(default_factory=list)

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


class BatchManager:
    def __init__(self, coordinator: GPUCoordinator):
        self.coordinator = coordinator
        self._state_lock = Lock()
        self._cancel = Event()
        self._thread: Thread | None = None
        self._snapshot = BatchSnapshot()

    def start(self, operation: str, catalog, store) -> BatchSnapshot:
        if operation not in {"detect", "prelabel"}:
            raise ValueError("unsupported batch operation")
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("a batch job is already running")
            self._cancel.clear()
            records = catalog.list_images()
            self._snapshot = BatchSnapshot(
                state="queued", operation=operation, total=len(records)
            )
            self._thread = Thread(
                target=self._run, args=(operation, records, store), daemon=True
            )
            self._thread.start()
            return deepcopy(self._snapshot)

    def _run(self, operation, records, store) -> None:
        with self._state_lock:
            self._snapshot.state = "running"
        for record in records:
            with self._state_lock:
                if self._cancel.is_set():
                    self._snapshot.state = "cancelled"
                    self._snapshot.current_image = None
                    return
                self._snapshot.current_image = record.name
            try:
                if record.error:
                    raise ValueError(record.error)
                existing = store.load(record)
                if existing.status == "completed":
                    self._increment("skipped")
                    continue
                if operation == "detect":
                    if existing.blocks:
                        self._increment("skipped")
                        continue
                    result = self.coordinator.detect(record).model_copy(
                        update={"revision": existing.revision}
                    )
                else:
                    active_blocks = [
                        block
                        for block in existing.blocks
                        if not block.skipped and block.task is not None
                    ]
                    if not active_blocks or all(block.text.strip() for block in active_blocks):
                        self._increment("skipped")
                        continue
                    result = self.coordinator.prelabel(
                        record, existing, replace_existing=False
                    )
                store.save(record, result)
                self._increment("processed")
            except Exception as exc:
                with self._state_lock:
                    self._snapshot.failed += 1
                    self._snapshot.errors.append(BatchError(record.name, str(exc)))
        with self._state_lock:
            self._snapshot.state = "cancelled" if self._cancel.is_set() else "completed"
            self._snapshot.current_image = None

    def _increment(self, field_name: str) -> None:
        with self._state_lock:
            setattr(self._snapshot, field_name, getattr(self._snapshot, field_name) + 1)

    def snapshot(self) -> BatchSnapshot:
        with self._state_lock:
            return deepcopy(self._snapshot)

    def cancel(self) -> BatchSnapshot:
        with self._state_lock:
            if self._snapshot.state in {"queued", "running"}:
                self._snapshot.state = "cancelling"
                self._cancel.set()
            return deepcopy(self._snapshot)

    async def shutdown(self) -> None:
        self.cancel()
        thread = self._thread
        if thread is not None:
            await asyncio.to_thread(thread.join)

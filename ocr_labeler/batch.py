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
    total: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    current_image: str | None = None
    errors: list[BatchError] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class InferenceCoordinator:
    def __init__(self, engine):
        self.engine = engine
        self._lock = Lock()

    def recognize(self, record):
        with self._lock:
            return self.engine.recognize(record)


class BatchManager:
    def __init__(self, coordinator: InferenceCoordinator):
        self.coordinator = coordinator
        self._state_lock = Lock()
        self._cancel = Event()
        self._thread: Thread | None = None
        self._snapshot = BatchSnapshot()

    def start(self, catalog, store) -> BatchSnapshot:
        records = catalog.list_images()
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("a batch job is already running")
            self._cancel.clear()
            self._snapshot = BatchSnapshot(state="queued", total=len(records))
            self._thread = Thread(target=self._run, args=(records, store), daemon=True)
            self._thread.start()
        return self.snapshot()

    def cancel(self) -> BatchSnapshot:
        with self._state_lock:
            if self._snapshot.state in {"queued", "running"}:
                self._snapshot.state = "cancelling"
                self._cancel.set()
            return deepcopy(self._snapshot)

    def snapshot(self) -> BatchSnapshot:
        with self._state_lock:
            return deepcopy(self._snapshot)

    def wait(self, timeout=None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    async def shutdown(self) -> None:
        self.cancel()
        await asyncio.to_thread(self.wait)

    def _run(self, records, store) -> None:
        try:
            if not self._begin():
                return
            for record in records:
                if not self._claim(record.name):
                    return
                try:
                    if record.error:
                        self._add_error(record.name, record.error)
                        continue
                    if store.has_annotation(record):
                        store.load(record)
                        self._increment("skipped")
                        continue
                    annotation = self.coordinator.recognize(record)
                    store.save(record, annotation)
                    self._increment("processed")
                except Exception as exc:
                    self._add_error(record.name, str(exc))
            self._finish()
        except Exception as exc:
            self._fail(exc)

    def _begin(self) -> bool:
        with self._state_lock:
            if self._cancel.is_set():
                self._snapshot.state = "cancelled"
                self._snapshot.current_image = None
                return False
            self._snapshot.state = "running"
            return True

    def _claim(self, image: str) -> bool:
        with self._state_lock:
            if self._cancel.is_set():
                self._snapshot.state = "cancelled"
                self._snapshot.current_image = None
                return False
            self._snapshot.current_image = image
            return True

    def _finish(self) -> None:
        with self._state_lock:
            self._snapshot.state = (
                "cancelled" if self._cancel.is_set() else "completed"
            )
            self._snapshot.current_image = None

    def _fail(self, exc: Exception) -> None:
        with self._state_lock:
            image = self._snapshot.current_image or "<batch>"
            self._snapshot.state = "failed"
            self._snapshot.current_image = None
            self._snapshot.failed += 1
            self._snapshot.errors.append(BatchError(image=image, message=str(exc)))

    def _increment(self, name: str) -> None:
        with self._state_lock:
            setattr(self._snapshot, name, getattr(self._snapshot, name) + 1)

    def _add_error(self, image: str, message: str) -> None:
        with self._state_lock:
            self._snapshot.failed += 1
            self._snapshot.errors.append(BatchError(image=image, message=message))

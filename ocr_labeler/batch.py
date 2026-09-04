from dataclasses import asdict, dataclass, field
from threading import Lock

from batch_lifecycle import BatchLifecycleMixin
from .batch_operations import RecognitionBatchOperation


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


class BatchManager(BatchLifecycleMixin):
    def __init__(self, coordinator: InferenceCoordinator):
        self.coordinator = coordinator
        self.operation = RecognitionBatchOperation(coordinator)
        self._init_lifecycle(BatchSnapshot, BatchError)

    def start(self, catalog, store) -> BatchSnapshot:
        records = catalog.list_images()
        self._start_job(
            BatchSnapshot(state="queued", total=len(records)),
            self._run,
            (records, store),
        )
        return self.snapshot()

    def _run(self, records, store) -> None:
        try:
            if not self._begin():
                return
            for record in records:
                if not self._claim(record.name):
                    return
                try:
                    processed = self.operation.execute(record, store)
                    self._increment("processed" if processed else "skipped")
                except Exception as exc:
                    self._add_error(record.name, str(exc))
            self._finish()
        except Exception as exc:
            self._fail(exc)

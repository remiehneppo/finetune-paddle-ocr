import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path

from PIL import Image

from ocr_labeler.batch import BatchManager, InferenceCoordinator
from ocr_labeler.catalog import WorkspaceCatalog
from ocr_labeler.storage import AnnotationStore


def make_image(path: Path, size=(20, 10)) -> None:
    Image.new("RGB", size, "white").save(path)


class FakeEngine:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = []

    def recognize(self, record):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append(record.name)
        try:
            time.sleep(0.01)
            annotation = AnnotationStore(record.path.parent).create_draft(record)
            annotation.status = "ocr"
            return annotation
        finally:
            self.active -= 1


class BatchTests(unittest.TestCase):
    def test_coordinator_serializes_concurrent_requests(self):
        engine = FakeEngine()
        coordinator = InferenceCoordinator(engine)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.png", "b.png"):
                make_image(root / name)
            records = WorkspaceCatalog.open(root).list_images()
            threads = [
                threading.Thread(target=coordinator.recognize, args=(record,))
                for record in records
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(engine.max_active, 1)

    def test_batch_saves_each_result_in_catalog_order_and_completes(self):
        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("page-10.png", "page-2.png"):
                make_image(root / name)
            catalog = WorkspaceCatalog.open(root)
            store = AnnotationStore(root)
            manager = BatchManager(InferenceCoordinator(engine))

            manager.start(catalog, store)
            manager.wait(timeout=2)
            snapshot = manager.snapshot()

            saved = [store.load(record) for record in catalog.list_images()]

        self.assertEqual(snapshot.state, "completed")
        self.assertEqual(snapshot.processed, 2)
        self.assertEqual(snapshot.failed, 0)
        self.assertEqual(engine.calls, ["page-2.png", "page-10.png"])
        self.assertEqual([annotation.status for annotation in saved], ["ocr", "ocr"])

    def test_batch_skips_only_valid_saved_annotations_and_continues_past_bad_image(self):
        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "saved.png")
            make_image(root / "fresh.png")
            (root / "broken.png").write_bytes(b"not an image")
            catalog = WorkspaceCatalog.open(root)
            store = AnnotationStore(root)
            saved_record = next(
                record for record in catalog.list_images() if record.name == "saved.png"
            )
            store.save(saved_record, store.create_draft(saved_record))
            manager = BatchManager(InferenceCoordinator(engine))

            manager.start(catalog, store)
            manager.wait(timeout=2)
            snapshot = manager.snapshot()

        self.assertEqual(snapshot.state, "completed")
        self.assertEqual(snapshot.skipped, 1)
        self.assertEqual(snapshot.processed, 1)
        self.assertEqual(snapshot.failed, 1)
        self.assertEqual(engine.calls, ["fresh.png"])
        self.assertEqual([error.image for error in snapshot.errors], ["broken.png"])

    def test_batch_reports_invalid_sidecar_without_overwriting_it(self):
        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            catalog = WorkspaceCatalog.open(root)
            record = catalog.list_images()[0]
            store = AnnotationStore(root)
            sidecar = store._path(record)
            sidecar.parent.mkdir(parents=True)
            sidecar.write_text("not json", encoding="utf-8")
            manager = BatchManager(InferenceCoordinator(engine))

            manager.start(catalog, store)
            manager.wait(timeout=2)
            snapshot = manager.snapshot()

            sidecar_contents = sidecar.read_text(encoding="utf-8")

        self.assertEqual(snapshot.state, "completed")
        self.assertEqual(snapshot.skipped, 0)
        self.assertEqual(snapshot.processed, 0)
        self.assertEqual(snapshot.failed, 1)
        self.assertEqual(engine.calls, [])
        self.assertEqual(sidecar_contents, "not json")

    def test_batch_reports_stale_parseable_sidecar_without_overwriting_it(self):
        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            make_image(image_path, size=(21, 10))
            catalog = WorkspaceCatalog.open(root)
            record = catalog.list_images()[0]
            store = AnnotationStore(root)
            store.save(record, store.create_draft(record))
            sidecar = store._path(record)
            before = sidecar.read_text(encoding="utf-8")
            make_image(image_path)
            manager = BatchManager(InferenceCoordinator(engine))

            manager.start(catalog, store)
            manager.wait(timeout=2)
            snapshot = manager.snapshot()

            after = sidecar.read_text(encoding="utf-8")

        self.assertEqual(snapshot.state, "completed")
        self.assertEqual(snapshot.skipped, 0)
        self.assertEqual(snapshot.processed, 0)
        self.assertEqual(snapshot.failed, 1)
        self.assertEqual(engine.calls, [])
        self.assertEqual(before, after)

    def test_cancel_finishes_current_image_then_stops(self):
        class BlockingEngine(FakeEngine):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.release = threading.Event()

            def recognize(self, record):
                self.started.set()
                self.release.wait(timeout=1)
                return super().recognize(record)

        engine = BlockingEngine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.png", "b.png", "c.png"):
                make_image(root / name)
            catalog = WorkspaceCatalog.open(root)
            manager = BatchManager(InferenceCoordinator(engine))
            manager.start(catalog, AnnotationStore(root))
            self.assertTrue(engine.started.wait(timeout=1))

            manager.cancel()
            engine.release.set()
            manager.wait(timeout=2)
            snapshot = manager.snapshot()

        self.assertEqual(snapshot.state, "cancelled")
        self.assertEqual(snapshot.processed, 1)
        self.assertEqual(engine.calls, ["a.png"])

    def test_cancel_winning_before_next_claim_prevents_that_image_from_starting(self):
        class GateLock:
            def __init__(self):
                self.lock = threading.Lock()
                self.arm = threading.Event()
                self.worker_waiting = threading.Event()
                self.release = threading.Event()

            def __enter__(self):
                if self.arm.is_set() and threading.current_thread() is not threading.main_thread():
                    self.worker_waiting.set()
                    self.release.wait(timeout=1)
                self.lock.acquire()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.lock.release()

        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.png", "b.png"):
                make_image(root / name)
            catalog = WorkspaceCatalog.open(root)
            manager = BatchManager(InferenceCoordinator(engine))
            gate = GateLock()
            manager._state_lock = gate
            original_increment = manager._increment

            def arm_boundary(name):
                original_increment(name)
                gate.arm.set()

            manager._increment = arm_boundary
            manager.start(catalog, AnnotationStore(root))
            self.assertTrue(gate.worker_waiting.wait(timeout=1))

            manager.cancel()
            gate.release.set()
            manager.wait(timeout=2)
            snapshot = manager.snapshot()

        self.assertEqual(snapshot.state, "cancelled")
        self.assertEqual(engine.calls, ["a.png"])

    def test_start_while_a_batch_is_live_raises(self):
        class BlockingEngine(FakeEngine):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.release = threading.Event()

            def recognize(self, record):
                self.started.set()
                self.release.wait(timeout=1)
                return super().recognize(record)

        engine = BlockingEngine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            catalog = WorkspaceCatalog.open(root)
            manager = BatchManager(InferenceCoordinator(engine))
            manager.start(catalog, AnnotationStore(root))
            self.assertTrue(engine.started.wait(timeout=1))

            with self.assertRaisesRegex(RuntimeError, "already running"):
                manager.start(catalog, AnnotationStore(root))

            manager.cancel()
            engine.release.set()
            manager.wait(timeout=2)

    def test_restart_after_cancel_skips_saved_work_and_processes_remaining_images(self):
        class BlockingEngine(FakeEngine):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.release = threading.Event()
                self.block_once = True

            def recognize(self, record):
                if self.block_once:
                    self.block_once = False
                    self.started.set()
                    self.release.wait(timeout=1)
                return super().recognize(record)

        engine = BlockingEngine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.png", "b.png", "c.png"):
                make_image(root / name)
            catalog = WorkspaceCatalog.open(root)
            store = AnnotationStore(root)
            manager = BatchManager(InferenceCoordinator(engine))
            manager.start(catalog, store)
            self.assertTrue(engine.started.wait(timeout=1))
            manager.cancel()
            engine.release.set()
            manager.wait(timeout=2)

            manager.start(catalog, store)
            manager.wait(timeout=2)
            snapshot = manager.snapshot()

        self.assertEqual(snapshot.state, "completed")
        self.assertEqual(snapshot.processed, 2)
        self.assertEqual(snapshot.skipped, 1)
        self.assertEqual(snapshot.failed, 0)
        self.assertEqual(engine.calls, ["a.png", "b.png", "c.png"])

    def test_shutdown_cancels_and_joins_a_blocked_batch(self):
        class BlockingEngine(FakeEngine):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.release = threading.Event()

            def recognize(self, record):
                self.started.set()
                self.release.wait(timeout=2)
                return super().recognize(record)

        engine = BlockingEngine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "a.png")
            make_image(root / "b.png")
            manager = BatchManager(InferenceCoordinator(engine))
            manager.start(WorkspaceCatalog.open(root), AnnotationStore(root))
            self.assertTrue(engine.started.wait(timeout=1))
            finished = threading.Event()

            def shutdown():
                asyncio.run(manager.shutdown())
                finished.set()

            worker = threading.Thread(target=shutdown)
            worker.start()
            self.assertFalse(finished.wait(timeout=0.1))
            engine.release.set()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(manager.snapshot().state, "cancelled")
        self.assertEqual(engine.calls, ["a.png"])

    def test_unsafe_persistence_is_recorded_and_batch_reaches_terminal_state(self):
        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            outside = root / "outside"
            outside.mkdir()
            (root / ".paddleocr-labeler").symlink_to(
                outside, target_is_directory=True
            )
            manager = BatchManager(InferenceCoordinator(engine))

            manager.start(WorkspaceCatalog.open(root), AnnotationStore(root))
            manager.wait(timeout=2)
            snapshot = manager.snapshot()
            asyncio.run(manager.shutdown())

        self.assertEqual(snapshot.state, "completed")
        self.assertEqual(snapshot.failed, 1)
        self.assertEqual(snapshot.errors[0].image, "page.png")
        self.assertIn("persistence", snapshot.errors[0].message)
        self.assertEqual(engine.calls, [])

    def test_unexpected_worker_error_sets_failed_terminal_state(self):
        class ExplodingBatchManager(BatchManager):
            def _claim(self, image):
                raise RuntimeError("unexpected worker failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            manager = ExplodingBatchManager(InferenceCoordinator(FakeEngine()))

            manager.start(WorkspaceCatalog.open(root), AnnotationStore(root))
            manager.wait(timeout=2)
            snapshot = manager.snapshot()

        self.assertEqual(snapshot.state, "failed")
        self.assertIsNone(snapshot.current_image)
        self.assertEqual(snapshot.failed, 1)
        self.assertIn("unexpected worker failure", snapshot.errors[0].message)


if __name__ == "__main__":
    unittest.main()

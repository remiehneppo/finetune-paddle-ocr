import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from starlette.exceptions import StarletteDeprecationWarning

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Using `httpx` with `starlette\.testclient` is deprecated; install `httpx2` instead\.",
    category=StarletteDeprecationWarning,
    module=r"fastapi\.testclient$",
)

from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from ocr_labeler.app import (
    AppState,
    OpenFileStreamingResponse,
    OpenWorkspaceRequest,
    create_app,
)
from ocr_labeler.batch import BatchSnapshot
from ocr_labeler.catalog import WorkspaceCatalog
from ocr_labeler.models import Block
from ocr_labeler.settings import LabelerSettings
from ocr_labeler.storage import AnnotationStore


class FakeEngine:
    def __init__(self):
        self.closed = False
        self.recognize_calls = 0

    def recognize(self, record):
        self.recognize_calls += 1
        annotation = AnnotationStore(record.path.parent).create_draft(record)
        annotation.status = "ocr"
        annotation.blocks = [
            Block(
                order=0,
                text="Việt Nam",
                polygon=[(1, 1), (30, 1), (30, 10), (1, 10)],
                score=0.95,
                source="ocr",
            )
        ]
        return annotation

    def close(self):
        self.closed = True


class LabelerAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.new("RGB", (40, 20), "white").save(self.root / "page.png")
        settings = LabelerSettings(rec_model_dir=Path("unused-in-injected-test"))
        self.client_context = TestClient(
            create_app(settings=settings, engine=FakeEngine())
        )
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temp.cleanup()

    def open_workspace(self):
        response = self.client.post("/api/workspace/open", json={"path": str(self.root)})
        self.assertEqual(response.status_code, 200)
        return self.client.get("/api/images").json()["images"][0]["image_id"]

    def test_open_list_ocr_save_reload_and_export(self):
        image_id = self.open_workspace()

        images = self.client.get("/api/images").json()["images"]
        self.assertEqual(images[0]["name"], "page.png")
        self.assertEqual(images[0]["status"], "not_ocr")
        content = self.client.get(f"/api/images/{image_id}/content")
        self.assertEqual(content.status_code, 200)
        self.assertEqual(content.headers["content-type"], "image/png")

        ocr = self.client.post(
            f"/api/images/{image_id}/ocr", json={"replace_existing": False}
        )
        self.assertEqual(ocr.status_code, 200)
        self.assertEqual(ocr.json()["revision"], 0)
        saved = self.client.put(f"/api/images/{image_id}/annotation", json=ocr.json())
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["revision"], 1)
        loaded = self.client.get(f"/api/images/{image_id}/annotation")
        self.assertEqual(loaded.status_code, 200)
        self.assertEqual(loaded.json()["revision"], 1)
        self.assertEqual(loaded.json()["blocks"][0]["text"], "Việt Nam")
        self.assertEqual(self.client.get("/api/images").json()["images"][0]["status"], "ocr")
        exported = self.client.post("/api/export")
        self.assertEqual(exported.status_code, 200)
        self.assertTrue(Path(exported.json()["path"]).is_file())
        self.assertEqual(exported.json()["records"], 1)

    def test_unknown_image_does_not_accept_a_path(self):
        self.open_workspace()
        response = self.client.get("/api/images/../../etc/passwd/content")
        self.assertIn(response.status_code, {404, 422})

    def test_revision_conflict_and_ocr_replacement_protection(self):
        image_id = self.open_workspace()
        draft = self.client.post(
            f"/api/images/{image_id}/ocr", json={"replace_existing": False}
        ).json()
        saved = self.client.put(f"/api/images/{image_id}/annotation", json=draft)
        self.assertEqual(saved.status_code, 200)
        conflict = self.client.put(f"/api/images/{image_id}/annotation", json=draft)
        self.assertEqual(conflict.status_code, 409)
        protected = self.client.post(
            f"/api/images/{image_id}/ocr", json={"replace_existing": False}
        )
        self.assertEqual(protected.status_code, 409)
        replacement = self.client.post(
            f"/api/images/{image_id}/ocr", json={"replace_existing": True}
        )
        self.assertEqual(replacement.status_code, 200)
        self.assertEqual(replacement.json()["revision"], 1)
        replaced = self.client.put(
            f"/api/images/{image_id}/annotation", json=replacement.json()
        )
        self.assertEqual(replaced.status_code, 200)
        self.assertEqual(replaced.json()["revision"], 2)

    def test_source_image_conflict_is_reported(self):
        image_id = self.open_workspace()
        draft = self.client.post(
            f"/api/images/{image_id}/ocr", json={"replace_existing": False}
        ).json()
        Image.new("RGB", (40, 20), "black").save(self.root / "page.png")

        conflict = self.client.put(f"/api/images/{image_id}/annotation", json=draft)
        self.assertEqual(conflict.status_code, 409)
        self.assertIn("source image changed", conflict.json()["detail"])

    def test_image_content_rejects_a_symlink_swap_without_serving_external_bytes(self):
        image_id = self.open_workspace()
        outside = self.root.parent / f"{self.root.name}-outside.png"
        Image.new("RGB", (40, 20), "black").save(outside)
        external_bytes = outside.read_bytes()
        (self.root / "page.png").unlink()
        (self.root / "page.png").symlink_to(outside)
        try:
            response = self.client.get(f"/api/images/{image_id}/content")
        finally:
            outside.unlink(missing_ok=True)

        self.assertEqual(response.status_code, 409)
        self.assertIn("source image changed", response.json()["detail"])
        self.assertNotEqual(response.content, external_bytes)

    def test_image_content_reports_deleted_source_as_a_conflict(self):
        image_id = self.open_workspace()
        (self.root / "page.png").unlink()

        response = self.client.get(f"/api/images/{image_id}/content")

        self.assertEqual(response.status_code, 409)
        self.assertIn("source image changed", response.json()["detail"])

    def test_image_content_serves_a_symlink_to_a_direct_child_target(self):
        target = self.root / "target.png"
        Image.new("RGB", (32, 16), "blue").save(target)
        alias = self.root / "alias.png"
        alias.symlink_to(target.name)
        response = self.client.post(
            "/api/workspace/open", json={"path": str(self.root)}
        )
        self.assertEqual(response.status_code, 200)
        images = self.client.get("/api/images").json()["images"]
        image_id = next(item["image_id"] for item in images if item["name"] == alias.name)

        content = self.client.get(f"/api/images/{image_id}/content")

        self.assertEqual(content.status_code, 200)
        self.assertEqual(content.content, target.read_bytes())
        self.assertEqual(content.headers["content-length"], str(target.stat().st_size))
        self.assertEqual(content.headers["content-type"], "image/png")

    def test_unsafe_persistence_routes_return_json_conflicts(self):
        image_id = self.open_workspace()
        draft = self.client.get(
            f"/api/images/{image_id}/annotation"
        ).json()
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / ".paddleocr-labeler").symlink_to(
            outside, target_is_directory=True
        )
        requests = [
            ("GET", "/api/images", None),
            ("GET", f"/api/images/{image_id}/annotation", None),
            ("PUT", f"/api/images/{image_id}/annotation", draft),
            ("POST", "/api/export", None),
        ]

        for method, path, payload in requests:
            with self.subTest(method=method, path=path):
                response = self.client.request(method, path, json=payload)
                self.assertEqual(response.status_code, 409)
                self.assertIn("unsafe persistence path", response.json()["detail"])

    def test_invalid_existing_sidecar_is_not_replaced_by_ocr(self):
        image_id = self.open_workspace()
        sidecar = self.root / ".paddleocr-labeler" / "annotations" / "page.json"
        sidecar.parent.mkdir(parents=True)
        invalid = "{not valid json"
        sidecar.write_text(invalid, encoding="utf-8")

        response = self.client.post(
            f"/api/images/{image_id}/ocr", json={"replace_existing": True}
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(sidecar.read_text(encoding="utf-8"), invalid)

    def test_batch_endpoint_reaches_terminal_state(self):
        self.open_workspace()
        started = self.client.post("/api/batch")
        self.assertEqual(started.status_code, 200)
        for _ in range(50):
            snapshot = self.client.get("/api/batch").json()
            if snapshot["state"] in {"completed", "cancelled", "failed"}:
                break
            time.sleep(0.01)
        self.assertEqual(snapshot["state"], "completed")
        self.assertEqual(snapshot["processed"], 1)

    def test_health_and_batch_cancel_are_available_before_workspace_opens(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["ready"])
        cancelled = self.client.post("/api/batch/cancel")
        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(cancelled.json()["state"], "idle")


class FalseyFakeEngine(FakeEngine):
    def __bool__(self):
        return False


class OpenFileStreamingResponseTests(unittest.TestCase):
    @staticmethod
    def make_response(payload: bytes):
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / "payload.bin"
        path.write_bytes(payload)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        response = OpenFileStreamingResponse(
            fd,
            content_length=len(payload),
            media_type="application/octet-stream",
        )
        return temp, fd, response

    def assert_fd_closed(self, fd, response):
        with self.assertRaises(OSError):
            os.fstat(fd)
        response.close()
        response.close()

    @staticmethod
    def scope(spec_version):
        return {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": spec_version},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/content",
            "raw_path": b"/content",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8010),
        }

    def test_normal_asgi_completion_closes_fd_while_response_remains_alive(self):
        temp, fd, response = self.make_response(b"normal body")
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        try:
            asyncio.run(response(self.scope("2.4"), receive, send))
            self.assertEqual(
                b"".join(
                    item.get("body", b"")
                    for item in sent
                    if item["type"] == "http.response.body"
                ),
                b"normal body",
            )
            self.assert_fd_closed(fd, response)
        finally:
            temp.cleanup()

    def test_disconnect_before_first_chunk_closes_fd(self):
        temp, fd, response = self.make_response(b"body")
        body_chunks = []
        disconnect_seen = None

        async def receive():
            disconnect_seen.set()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                await disconnect_seen.wait()
                await asyncio.sleep(0)
            if message["type"] == "http.response.body" and message.get("body"):
                body_chunks.append(message["body"])

        try:
            async def run_response():
                nonlocal disconnect_seen
                disconnect_seen = asyncio.Event()
                await response(self.scope("2.3"), receive, send)

            asyncio.run(run_response())
            self.assertEqual(body_chunks, [])
            self.assert_fd_closed(fd, response)
        finally:
            temp.cleanup()

    def test_mid_stream_disconnect_closes_fd(self):
        temp, fd, response = self.make_response(b"x" * (3 * 1024 * 1024))

        async def run_response():
            first_chunk = asyncio.Event()
            body_chunks = []

            async def receive():
                await first_chunk.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.body" and message.get("body"):
                    body_chunks.append(message["body"])
                    first_chunk.set()
                    await asyncio.sleep(0)

            await response(self.scope("2.3"), receive, send)
            return body_chunks

        try:
            body_chunks = asyncio.run(run_response())
            self.assertEqual(len(body_chunks), 1)
            self.assert_fd_closed(fd, response)
        finally:
            temp.cleanup()

    def test_asgi_24_send_oserror_closes_fd(self):
        temp, fd, response = self.make_response(b"body")

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("client disconnected")

        try:
            with self.assertRaises(ClientDisconnect):
                asyncio.run(response(self.scope("2.4"), receive, send))
            self.assert_fd_closed(fd, response)
        finally:
            temp.cleanup()


class BlockingBatch:
    def __init__(self):
        self.started = Event()
        self.release = Event()
        self.arguments = None

    def snapshot(self):
        return SimpleNamespace(state="idle")

    def start(self, catalog, store):
        self.arguments = (catalog, store)
        self.started.set()
        self.release.wait(timeout=1)
        return BatchSnapshot(state="queued", total=1)


class AppStateConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root_a = Path(self.temp.name) / "a"
        self.root_b = Path(self.temp.name) / "b"
        self.root_a.mkdir()
        self.root_b.mkdir()
        Image.new("RGB", (40, 20), "white").save(self.root_a / "page.png")
        Image.new("RGB", (40, 20), "white").save(self.root_b / "page.png")
        self.settings = LabelerSettings(rec_model_dir=Path("unused-in-injected-test"))

    def tearDown(self):
        self.temp.cleanup()

    def test_require_workspace_snapshots_one_pair_while_swap_is_locked(self):
        state = AppState(self.settings, FakeEngine())
        catalog_a = WorkspaceCatalog.open(self.root_a)
        store_a = AnnotationStore(catalog_a.root)
        catalog_b = WorkspaceCatalog.open(self.root_b)
        store_b = AnnotationStore(catalog_b.root)
        state.catalog = catalog_a
        state.store = store_a
        state._workspace = SimpleNamespace(catalog=catalog_a, store=store_a)
        result = []
        finished = Event()

        def read_workspace():
            result.append(state.require_workspace())
            finished.set()

        with state.workspace_lock:
            worker = Thread(target=read_workspace)
            worker.start()
            self.assertFalse(finished.wait(timeout=0.2))
            state.catalog = catalog_b
            state.store = store_b
            state._workspace = SimpleNamespace(catalog=catalog_b, store=store_b)
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result[0][0].root, self.root_b.resolve())
        self.assertEqual(result[0][1].root, self.root_b.resolve())

    def test_open_and_batch_start_have_one_ordering_and_keep_one_pair(self):
        app = create_app(self.settings, engine=FakeEngine())
        open_workspace = next(
            route.endpoint
            for route in app.routes
            if route.path == "/api/workspace/open" and "POST" in route.methods
        )
        start_batch = next(
            route.endpoint
            for route in app.routes
            if route.path == "/api/batch" and "POST" in route.methods
        )
        open_workspace(OpenWorkspaceRequest(path=str(self.root_a)))
        state = app.state.labeler
        batch = BlockingBatch()
        state.batch = batch
        start_result = []
        open_result = []
        open_finished = Event()

        start_worker = Thread(target=lambda: start_result.append(start_batch()))
        start_worker.start()
        self.assertTrue(batch.started.wait(timeout=1))

        def open_second_workspace():
            open_result.append(open_workspace(OpenWorkspaceRequest(path=str(self.root_b))))
            open_finished.set()

        open_worker = Thread(target=open_second_workspace)
        open_worker.start()
        self.assertFalse(open_finished.wait(timeout=0.2))
        batch.release.set()
        start_worker.join(timeout=1)
        open_worker.join(timeout=1)

        self.assertFalse(start_worker.is_alive())
        self.assertFalse(open_worker.is_alive())
        self.assertEqual(batch.arguments[0].root, self.root_a.resolve())
        self.assertEqual(batch.arguments[1].root, self.root_a.resolve())
        self.assertEqual(open_result[0]["root"], str(self.root_b.resolve()))
        self.assertEqual(start_result[0]["state"], "queued")


class AppLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.new("RGB", (40, 20), "white").save(self.root / "page.png")
        self.settings = LabelerSettings(rec_model_dir=Path("unused-in-injected-test"))

    def tearDown(self):
        self.temp.cleanup()

    def test_falsey_injected_engine_is_used_without_creating_paddle_engine(self):
        injected = FalseyFakeEngine()
        with patch("ocr_labeler.app.PaddleOCREngine.create") as create_engine:
            app = create_app(self.settings, engine=injected)
            with TestClient(app) as client:
                opened = client.post(
                    "/api/workspace/open", json={"path": str(self.root)}
                )
                self.assertEqual(opened.status_code, 200)
                image_id = client.get("/api/images").json()["images"][0]["image_id"]
                self.assertEqual(
                    client.post(
                        f"/api/images/{image_id}/ocr",
                        json={"replace_existing": False},
                    ).status_code,
                    200,
                )
        create_engine.assert_not_called()
        self.assertEqual(injected.recognize_calls, 1)
        self.assertFalse(injected.closed)

    def test_owned_engine_is_created_once_and_closed_after_normal_shutdown(self):
        created = FakeEngine()
        with patch(
            "ocr_labeler.app.PaddleOCREngine.create", return_value=created
        ) as create_engine:
            app = create_app(self.settings)
            with TestClient(app) as client:
                self.assertEqual(client.get("/api/health").status_code, 200)
        create_engine.assert_called_once_with(self.settings)
        self.assertTrue(created.closed)

    def test_owned_engine_closes_when_initial_workspace_startup_fails(self):
        created = FakeEngine()
        missing = self.root / "missing"
        with patch(
            "ocr_labeler.app.PaddleOCREngine.create", return_value=created
        ) as create_engine:
            app = create_app(self.settings, initial_workspace=missing)
            with self.assertRaises(FileNotFoundError):
                with TestClient(app):
                    pass
        create_engine.assert_called_once_with(self.settings)
        self.assertTrue(created.closed)

    def test_owned_engine_closes_only_after_blocked_batch_is_cancelled_and_joined(self):
        class BlockingEngine(FakeEngine):
            def __init__(self):
                super().__init__()
                self.started = Event()
                self.release = Event()
                self.active = False
                self.closed_while_active = False

            def recognize(self, record):
                self.active = True
                self.started.set()
                self.release.wait(timeout=2)
                try:
                    return super().recognize(record)
                finally:
                    self.active = False

            def close(self):
                self.closed_while_active = self.active
                super().close()

        created = BlockingEngine()
        with patch("ocr_labeler.app.PaddleOCREngine.create", return_value=created):
            context = TestClient(create_app(self.settings, initial_workspace=self.root))
            client = context.__enter__()
            self.assertEqual(client.post("/api/batch").status_code, 200)
            self.assertTrue(created.started.wait(timeout=1))
            stopped = Event()

            def stop_client():
                context.__exit__(None, None, None)
                stopped.set()

            worker = Thread(target=stop_client)
            worker.start()
            self.assertFalse(stopped.wait(timeout=0.1))
            self.assertFalse(created.closed)
            created.release.set()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertTrue(created.closed)
        self.assertFalse(created.closed_while_active)
        self.assertEqual(
            context.app.state.labeler.batch.snapshot().state,
            "cancelled",
        )

    def test_injected_engine_batch_is_cancelled_and_joined_on_shutdown(self):
        class BlockingInjectedEngine(FakeEngine):
            def __init__(self):
                super().__init__()
                self.started = Event()
                self.release = Event()

            def recognize(self, record):
                self.started.set()
                self.release.wait(timeout=2)
                return super().recognize(record)

        injected = BlockingInjectedEngine()
        context = TestClient(
            create_app(
                self.settings,
                engine=injected,
                initial_workspace=self.root,
            )
        )
        client = context.__enter__()
        self.assertEqual(client.post("/api/batch").status_code, 200)
        self.assertTrue(injected.started.wait(timeout=1))
        stopped = Event()

        def stop_client():
            context.__exit__(None, None, None)
            stopped.set()

        worker = Thread(target=stop_client)
        worker.start()
        self.assertFalse(stopped.wait(timeout=0.1))
        injected.release.set()
        worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertFalse(injected.closed)
        self.assertEqual(
            context.app.state.labeler.batch.snapshot().state,
            "cancelled",
        )

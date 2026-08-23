from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
import time
import unittest
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from vl_layout_labeler.app import create_app
from vl_layout_labeler.batch import BatchManager
from vl_layout_labeler.catalog import WorkspaceCatalog
from vl_layout_labeler.models import (
    Annotation,
    Block,
    ImageInfo,
    OCRValidation,
    ValidationIssue,
)
from vl_layout_labeler.post_validation import (
    OCRPostValidationError,
    OpenAICompatiblePostValidator,
    ValidationService,
)
from vl_layout_labeler.settings import LabelerSettings
from vl_layout_labeler.storage import AnnotationStore


def response(content, status=200):
    return httpx.Response(
        status,
        json={"choices": [{"message": {"content": content}}]},
    )


class PostValidatorAdapterTests(unittest.TestCase):
    def test_client_configuration_keeps_auth_out_of_payload(self):
        client = Mock()
        with patch(
            "vl_layout_labeler.post_validation.httpx.Client", return_value=client
        ) as constructor:
            validator = OpenAICompatiblePostValidator(
                base_url="http://validator.test/v1/",
                model="reviewer",
                api_key="secret-key",
                timeout=4.5,
                max_tokens=321,
            )
        constructor.assert_called_once_with(
            timeout=4.5, headers={"Authorization": "Bearer secret-key"}
        )
        payload = validator._payload(uuid4(), "ocr", "secret OCR text")
        self.assertEqual(payload["model"], "reviewer")
        self.assertEqual(payload["max_tokens"], 321)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        schema = payload["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertNotIn("secret-key", json.dumps(payload))

    def test_valid_response_is_parsed_and_sent_to_chat_completions(self):
        block_id = uuid4()
        calls = []

        def handler(request):
            calls.append(request)
            payload = json.loads(request.content)
            self.assertEqual(
                payload["response_format"]["json_schema"]["name"],
                "ocr_post_validation",
            )
            return response(
                json.dumps(
                    {
                        "issues": [
                            {
                                "block_id": str(block_id),
                                "start": 0,
                                "end": 3,
                                "text": "Vỉe",
                                "category": "character",
                                "reason": "Ký tự có vẻ sai",
                                "suggestion": "Việ",
                            }
                        ]
                    }
                )
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        validator = OpenAICompatiblePostValidator(
            base_url="http://validator.test/v1",
            model="reviewer",
            client=client,
        )
        issues = validator.validate_block(block_id, "ocr", "Vỉet Nam")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].block_id, block_id)
        self.assertEqual(calls[0].url.path, "/v1/chat/completions")

    def test_semantic_contract_retries_once(self):
        block_id = uuid4()
        contents = [
            json.dumps(
                {
                    "issues": [
                        {
                            "block_id": str(block_id),
                            "start": 0,
                            "end": 4,
                            "text": "wrong",
                            "category": "word",
                            "reason": "bad",
                            "suggestion": "good",
                        }
                    ]
                }
            ),
            json.dumps({"issues": []}),
        ]
        calls = []

        def handler(_request):
            calls.append(1)
            return response(contents.pop(0))

        validator = OpenAICompatiblePostValidator(
            base_url="http://validator.test/v1",
            model="reviewer",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        self.assertEqual(validator.validate_block(block_id, "ocr", "text"), [])
        self.assertEqual(len(calls), 2)

    def test_out_of_bounds_and_mismatched_text_fail_after_one_retry(self):
        block_id = uuid4()
        invalid_issues = (
            {
                "block_id": str(block_id),
                "start": 0,
                "end": 99,
                "text": "text",
                "category": "word",
                "reason": "bad",
                "suggestion": "good",
            },
            {
                "block_id": str(block_id),
                "start": 0,
                "end": 4,
                "text": "nope",
                "category": "word",
                "reason": "bad",
                "suggestion": "good",
            },
        )
        for raw_issue in invalid_issues:
            with self.subTest(raw_issue=raw_issue):
                calls = []

                def handler(_request):
                    calls.append(1)
                    return response(json.dumps({"issues": [raw_issue]}))

                validator = OpenAICompatiblePostValidator(
                    base_url="http://validator.test/v1",
                    model="reviewer",
                    client=httpx.Client(transport=httpx.MockTransport(handler)),
                )
                with self.assertRaisesRegex(
                    OCRPostValidationError, "semantic contract"
                ):
                    validator.validate_block(block_id, "ocr", "text")
                self.assertEqual(len(calls), 2)

    def test_timeout_and_malformed_json_are_sanitized(self):
        block_id = uuid4()

        def timeout_handler(request):
            raise httpx.ReadTimeout("contains secret OCR text", request=request)

        timed = OpenAICompatiblePostValidator(
            base_url="http://validator.test/v1",
            model="reviewer",
            client=httpx.Client(transport=httpx.MockTransport(timeout_handler)),
        )
        with self.assertRaisesRegex(OCRPostValidationError, "^LLM validation timed out$"):
            timed.validate_block(block_id, "ocr", "secret OCR text")

        malformed = OpenAICompatiblePostValidator(
            base_url="http://validator.test/v1",
            model="reviewer",
            client=httpx.Client(
                transport=httpx.MockTransport(lambda _request: response("{bad"))
            ),
        )
        with self.assertRaisesRegex(OCRPostValidationError, "malformed JSON"):
            malformed.validate_block(block_id, "ocr", "secret OCR text")


class FakeLayout:
    def detect(self, record):
        return Annotation(
            image=ImageInfo(
                path=record.relative_path,
                width=record.width,
                height=record.height,
                sha256=record.sha256,
            ),
            status="detected",
            blocks=[
                Block(
                    order=0,
                    polygon=[(1, 1), (8, 1), (8, 8), (1, 8)],
                    layout_label="text",
                    task="ocr",
                ),
                Block(
                    order=1,
                    polygon=[(9, 1), (14, 1), (14, 8), (9, 8)],
                    layout_label="display_formula",
                    task="formula",
                ),
                Block(
                    order=2,
                    polygon=[(15, 1), (19, 1), (19, 8), (15, 8)],
                    layout_label="image",
                    task=None,
                ),
            ],
        )


class FakeVL:
    def prelabel(self, _path, _polygon, task, _width, _height):
        return {"ocr": "Vỉet Nam", "formula": "x"}[task]


class FakeValidator:
    model = "mock-reviewer"

    def __init__(self, error=None, started=None, release=None):
        self.error = error
        self.calls = []
        self.started = started
        self.release = release

    def validate_block(self, block_id, task, text):
        self.calls.append((block_id, task, text))
        if self.started:
            self.started.set()
        if self.release:
            self.release.wait(timeout=2)
        if self.error:
            raise self.error
        return [
            ValidationIssue(
                block_id=block_id,
                start=0,
                end=3,
                category="character",
                reason="Có thể sai ký tự",
                suggestion="Việ",
            )
        ]

    def close(self):
        pass


class PostValidationAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.new("RGB", (20, 10), "white").save(self.root / "page.png")

    def tearDown(self):
        self.temp.cleanup()

    def open_and_detect(self, client):
        client.post("/api/workspace/open", json={"path": str(self.root)})
        image_id = client.get("/api/images").json()["images"][0]["image_id"]
        client.post(
            f"/api/images/{image_id}/detect", json={"replace_existing": True}
        )
        return image_id

    def make_app(self, validator=None, configured=False):
        settings = LabelerSettings(
            validation_base_url="http://validator.test/v1" if configured else None,
            validation_model="mock-reviewer" if configured else None,
            validation_api_key="must-not-leak" if configured else None,
        ).validate(require_runtime_models=False)
        return create_app(
            settings,
            layout_engine=FakeLayout(),
            vl_client=FakeVL(),
            post_validator=validator,
        )

    def test_disabled_capability_and_prelabel_fail_open(self):
        with TestClient(self.make_app()) as client:
            health = client.get("/api/health").json()
            self.assertEqual(
                health["post_validation"], {"configured": False, "model": None}
            )
            self.assertNotIn("api_key", json.dumps(health))
            image_id = self.open_and_detect(client)
            result = client.post(
                f"/api/images/{image_id}/prelabel",
                json={
                    "block_ids": None,
                    "replace_existing": True,
                    "post_validate": True,
                },
            )
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["annotation"]["blocks"][0]["text"], "Vỉet Nam")
            self.assertIn("not configured", result.json()["validation_error"])
            self.assertEqual(
                client.post(
                    f"/api/images/{image_id}/validate", json={"block_ids": None}
                ).status_code,
                503,
            )

    def test_validate_success_skips_formula_and_layout_and_complete_stays_allowed(self):
        validator = FakeValidator()
        with TestClient(self.make_app(validator, configured=True)) as client:
            image_id = self.open_and_detect(client)
            prelabelled = client.post(
                f"/api/images/{image_id}/prelabel",
                json={
                    "block_ids": None,
                    "replace_existing": True,
                    "post_validate": True,
                },
            ).json()
            self.assertIsNone(prelabelled["validation_error"])
            annotation = prelabelled["annotation"]
            self.assertEqual(len(validator.calls), 1)
            self.assertEqual(validator.calls[0][1], "ocr")
            self.assertEqual(len(annotation["blocks"][0]["validation"]["issues"]), 1)
            self.assertIsNone(annotation["blocks"][1]["validation"])
            self.assertIsNone(annotation["blocks"][2]["validation"])
            completed = client.post(f"/api/images/{image_id}/complete")
            self.assertEqual(completed.status_code, 200, completed.text)
            self.assertEqual(
                client.post(
                    f"/api/images/{image_id}/validate", json={"block_ids": None}
                ).status_code,
                409,
            )

    def test_revision_conflict_does_not_overwrite_concurrent_text_edit(self):
        started = Event()
        release = Event()
        validator = FakeValidator(started=started, release=release)
        app = self.make_app(validator, configured=True)
        with TestClient(app) as client:
            image_id = self.open_and_detect(client)
            current = client.post(
                f"/api/images/{image_id}/prelabel",
                json={"block_ids": None, "replace_existing": True},
            ).json()
            result = {}

            def validate():
                result["response"] = client.post(
                    f"/api/images/{image_id}/validate", json={"block_ids": None}
                )

            thread = Thread(target=validate)
            thread.start()
            self.assertTrue(started.wait(timeout=1))
            current["blocks"][0]["text"] = "updated concurrently"
            edited = client.put(
                f"/api/images/{image_id}/annotation", json=current
            )
            self.assertEqual(edited.status_code, 200)
            release.set()
            thread.join(timeout=2)
            self.assertEqual(result["response"].status_code, 409)
            latest = client.get(f"/api/images/{image_id}/annotation").json()
            self.assertEqual(latest["blocks"][0]["text"], "updated concurrently")
            self.assertIsNone(latest["blocks"][0]["validation"])

    def test_completed_image_and_no_eligible_blocks_are_clear_errors(self):
        validator = FakeValidator()
        with TestClient(self.make_app(validator, configured=True)) as client:
            image_id = self.open_and_detect(client)
            detected = client.get(f"/api/images/{image_id}/annotation").json()
            formula_id = detected["blocks"][1]["id"]
            response = client.post(
                f"/api/images/{image_id}/validate",
                json={"block_ids": [formula_id]},
            )
            self.assertEqual(response.status_code, 422)
            self.assertIn("no eligible", response.json()["detail"])

    def test_validate_accepts_an_omitted_optional_body(self):
        validator = FakeValidator()
        with TestClient(self.make_app(validator, configured=True)) as client:
            image_id = self.open_and_detect(client)
            client.post(
                f"/api/images/{image_id}/prelabel",
                json={"block_ids": None, "replace_existing": True},
            )
            response = client.post(f"/api/images/{image_id}/validate")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()["blocks"][0]["validation"]["issues"]), 1)


class PostValidationStorageAndBatchTests(unittest.TestCase):
    def test_old_v2_sidecar_without_validation_loads_unchanged(self):
        annotation = Annotation.model_validate(
            {
                "version": 2,
                "image": {
                    "path": "page.png",
                    "width": 20,
                    "height": 10,
                    "sha256": "a" * 64,
                },
                "blocks": [
                    {
                        "id": "12345678-1234-5678-1234-567812345678",
                        "order": 0,
                        "polygon": [(1, 1), (10, 1), (10, 8), (1, 8)],
                        "layout_label": "text",
                        "task": "ocr",
                        "text": "hello",
                    }
                ],
            }
        )
        self.assertIsNone(annotation.blocks[0].validation)
        self.assertEqual(annotation.blocks[0].text, "hello")

    def test_text_edit_clears_matching_sidecar_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 10), "white").save(root / "page.png")
            record = WorkspaceCatalog.open(root).list_images()[0]
            store = AnnotationStore(root)
            block = Block(
                order=0,
                polygon=[(1, 1), (10, 1), (10, 8), (1, 8)],
                layout_label="text",
                task="ocr",
                text="old",
            )
            issue = ValidationIssue(
                block_id=block.id,
                start=0,
                end=3,
                category="word",
                reason="bad",
                suggestion="new",
            )
            block.validation = OCRValidation(
                text_hash=block.current_text_hash(),
                model="reviewer",
                checked_at=datetime.now().astimezone(),
                issues=[issue],
            )
            saved = store.save(
                record,
                store.load(record).model_copy(
                    update={"status": "edited", "blocks": [block]}
                ),
            )
            edited = saved.model_copy(
                update={
                    "blocks": [
                        saved.blocks[0].model_copy(update={"text": "new text"})
                    ]
                }
            )
            self.assertIsNone(store.save(record, edited).blocks[0].validation)

    def test_batch_saves_prelabel_when_validation_fails(self):
        class Coordinator:
            def prelabel(self, _record, annotation, **_kwargs):
                block = annotation.blocks[0].model_copy(update={"text": "saved OCR"})
                return annotation.model_copy(
                    update={"status": "edited", "blocks": [block]}
                )

        validator = FakeValidator(
            error=OCRPostValidationError("LLM validation request failed")
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 10), "white").save(root / "page.png")
            catalog = WorkspaceCatalog.open(root)
            record = catalog.list_images()[0]
            store = AnnotationStore(root)
            draft = store.load(record)
            store.save(
                record,
                draft.model_copy(
                    update={
                        "status": "detected",
                        "blocks": [
                            Block(
                                order=0,
                                polygon=[(1, 1), (10, 1), (10, 8), (1, 8)],
                                layout_label="text",
                                task="ocr",
                            )
                        ],
                    }
                ),
            )
            manager = BatchManager(
                Coordinator(), ValidationService(validator)
            )
            manager.start("prelabel", catalog, store, post_validate=True)
            for _ in range(100):
                if manager.snapshot().state == "completed":
                    break
                time.sleep(0.01)
            snapshot = manager.snapshot()
            saved = store.load(record)
        self.assertEqual(snapshot.processed, 1)
        self.assertEqual(snapshot.failed, 0)
        self.assertEqual(snapshot.validation_failed, 1)
        self.assertEqual(len(snapshot.validation_errors), 1)
        self.assertEqual(saved.blocks[0].text, "saved OCR")

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("datasets"), "datasets not installed"
    )
    def test_validation_metadata_does_not_change_hf_or_layout_exports(self):
        def prepare(root, with_validation):
            for name, color in (("page1.png", "white"), ("page2.png", "gray")):
                Image.new("RGB", (20, 10), color).save(root / name)
            catalog = WorkspaceCatalog.open(root)
            store = AnnotationStore(root)
            for index, record in enumerate(catalog.list_images()):
                block = Block(
                    id=UUID(f"12345678-1234-5678-1234-{index + 1:012d}"),
                    order=0,
                    polygon=[(1, 1), (10, 1), (10, 8), (1, 8)],
                    layout_label="text",
                    task="ocr",
                    text=f"hello {index}",
                )
                if with_validation:
                    block.validation = OCRValidation(
                        text_hash=block.current_text_hash(),
                        model="reviewer",
                        checked_at=datetime.now().astimezone(),
                        issues=[],
                    )
                saved = store.save(
                    record,
                    store.load(record).model_copy(
                        update={"status": "edited", "blocks": [block]}
                    ),
                )
                store.save(
                    record,
                    Annotation.model_validate(
                        {**saved.model_dump(mode="python"), "status": "completed"}
                    ),
                )
            return catalog, store

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            plain_root = base / "plain"
            validated_root = base / "validated"
            plain_root.mkdir()
            validated_root.mkdir()
            plain_catalog, plain_store = prepare(plain_root, False)
            validated_catalog, validated_store = prepare(validated_root, True)
            plain_store.export_layout(plain_catalog, base / "plain-layout")
            validated_store.export_layout(validated_catalog, base / "validated-layout")
            plain_store.export_hf(plain_catalog, base / "plain-hf")
            validated_store.export_hf(validated_catalog, base / "validated-hf")

            def relative_bytes(root):
                return {
                    path.relative_to(root): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file()
                }

            self.assertEqual(
                relative_bytes(base / "plain-layout"),
                relative_bytes(base / "validated-layout"),
            )

            from datasets import load_from_disk

            def hf_records(root):
                dataset = load_from_disk(root)
                return {
                    split: [
                        {
                            "text": row["text"],
                            "task": row["task"],
                            "source_page_id": row["source_page_id"],
                            "image": row["image"].tobytes(),
                        }
                        for row in dataset[split]
                    ]
                    for split in dataset
                }

            self.assertEqual(
                hf_records(base / "plain-hf"),
                hf_records(base / "validated-hf"),
            )


if __name__ == "__main__":
    unittest.main()

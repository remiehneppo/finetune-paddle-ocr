import unittest
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r"Using `httpx` with `starlette\.testclient` is deprecated; install `httpx2` instead\.",
    category=UserWarning,
    module=r"fastapi\.testclient",
)

from fastapi.testclient import TestClient

from ocr_labeler.app import create_app
from ocr_labeler.settings import LabelerSettings


class NoOpEngine:
    pass


class StaticContractTests(unittest.TestCase):
    def test_application_shell_has_required_regions_and_module_entry(self):
        html = Path("ocr_labeler/static/index.html").read_text(encoding="utf-8")
        for element_id in (
            "folder-path",
            "image-list",
            "page-stage",
            "overlay",
            "block-list",
            "text-editor",
            "batch-progress",
            "save-status",
        ):
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn('src="/static/app.mjs"', html)
        self.assertIn('type="module"', html)
        self.assertIn(
            'id="page-stage" role="region" tabindex="0" aria-label="Trang và vùng văn bản"',
            html,
        )
        self.assertIn(
            'id="overlay" role="group" aria-label="Các vùng văn bản và tay nắm chỉnh sửa"',
            html,
        )
        self.assertNotIn('role="img"', html)

    def test_shell_and_stylesheet_are_served_over_http(self):
        settings = LabelerSettings(rec_model_dir=Path("unused-static-test"))
        with TestClient(create_app(settings=settings, engine=NoOpEngine())) as client:
            shell = client.get("/")
            stylesheet = client.get("/static/styles.css")

        self.assertEqual(shell.status_code, 200)
        self.assertIn("PaddleOCR Labeler", shell.text)
        self.assertIn('id="page-stage"', shell.text)
        self.assertIn('id="overlay"', shell.text)
        self.assertEqual(stylesheet.status_code, 200)
        self.assertIn("grid-template-columns: 260px minmax(0, 1fr) 380px", stylesheet.text)

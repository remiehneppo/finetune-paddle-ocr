"""Local browser-QA fixture with a mock OpenAI-compatible validator."""

from __future__ import annotations

import json
import os
from threading import Thread

from fastapi import FastAPI
import uvicorn

from vl_layout_labeler.app import create_app
from vl_layout_labeler.models import Annotation, Block, ImageInfo
from vl_layout_labeler.settings import LabelerSettings


mock_validator = FastAPI()


@mock_validator.post("/v1/chat/completions")
def chat_completions(payload: dict):
    user_content = payload["messages"][1]["content"]
    block_id = user_content.splitlines()[0].split("=", 1)[1]
    text = user_content.split("raw_text:\n", 1)[1]
    excerpt = text[:3]
    content = json.dumps(
        {
            "issues": [
                {
                    "block_id": block_id,
                    "start": 0,
                    "end": len(excerpt),
                    "text": excerpt,
                    "category": "character",
                    "reason": "<img src=x onerror=alert(1)> có thể là OCR sai.",
                    "suggestion": "<script>alert(1)</script> Việt",
                }
            ]
        },
        ensure_ascii=False,
    )
    return {"choices": [{"message": {"content": content}}]}


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
                    polygon=[(10, 10), (310, 10), (310, 90), (10, 90)],
                    layout_label="text",
                    task="ocr",
                )
            ],
        )


class FakeVL:
    def prelabel(self, *_args):
        return "Vỉet Nam"


def _start_mock_validator():
    server = uvicorn.Server(
        uvicorn.Config(mock_validator, host="127.0.0.1", port=8766, log_level="error")
    )
    Thread(target=server.run, daemon=True).start()


_start_mock_validator()
settings = LabelerSettings(
    validation_base_url="http://127.0.0.1:8766/v1",
    validation_model="mock-reviewer",
).validate(require_runtime_models=False)
app = create_app(
    settings,
    layout_engine=FakeLayout(),
    vl_client=FakeVL(),
    initial_workspace=os.environ["VL_QA_WORKSPACE"],
)

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import uvicorn

from .app import create_app
from .settings import LabelerSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PaddleOCR-VL layout crop labeler")
    parser.add_argument("--layout-model-dir", type=Path, default=LabelerSettings.layout_model_dir)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--vl-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vl-model", default="paddleocr-vl")
    parser.add_argument("--vl-api-key")
    parser.add_argument("--vl-timeout", type=float, default=120.0)
    parser.add_argument("--vl-max-tokens", type=int, default=4096)
    parser.add_argument("--validation-base-url")
    parser.add_argument("--validation-model")
    parser.add_argument("--validation-api-key")
    parser.add_argument("--validation-timeout", type=float, default=30.0)
    parser.add_argument("--validation-max-tokens", type=int, default=2048)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--images", type=Path)
    return parser


def parse_args(argv: Sequence[str] | None = None):
    return build_parser().parse_args(argv)


def build_settings(args) -> LabelerSettings:
    return LabelerSettings(
        layout_model_dir=args.layout_model_dir,
        device=args.device,
        vl_base_url=args.vl_base_url,
        vl_model=args.vl_model,
        vl_api_key=args.vl_api_key,
        vl_timeout=args.vl_timeout,
        vl_max_tokens=args.vl_max_tokens,
        validation_base_url=args.validation_base_url,
        validation_model=args.validation_model,
        validation_api_key=args.validation_api_key,
        validation_timeout=args.validation_timeout,
        validation_max_tokens=args.validation_max_tokens,
        host=args.host,
        port=args.port,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = build_settings(args).validate()
    app = create_app(settings, initial_workspace=args.images)
    uvicorn.run(app, host=settings.host, port=settings.port, workers=1)
    return 0

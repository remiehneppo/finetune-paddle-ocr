import argparse
from pathlib import Path
from typing import Sequence

import uvicorn

from .app import create_app
from .settings import LabelerSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PaddleOCR Vietnamese labeler")
    parser.add_argument(
        "--rec-model-dir",
        type=Path,
        default=Path("runs/vi_rec_3datasets_v1/inference/best_accuracy"),
        help="PaddleOCR recognition inference model directory",
    )
    parser.add_argument(
        "--det-model-dir",
        type=Path,
        help="optional PaddleOCR detection inference model directory",
    )
    parser.add_argument("--device", default="gpu:0", help="gpu:<index> or cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--images",
        type=Path,
        help="optional directory of direct-child images to open at startup",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_settings(args: argparse.Namespace) -> LabelerSettings:
    return LabelerSettings(
        rec_model_dir=args.rec_model_dir,
        det_model_dir=args.det_model_dir,
        device=args.device,
        host=args.host,
        port=args.port,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = build_settings(args).validate()
    app = create_app(settings=settings, initial_workspace=args.images)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=1,
    )
    return 0

import argparse
from pathlib import Path
from typing import Sequence

import uvicorn

from .app import create_app
from .settings import LabelerSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lightweight PP-OCRv6 detection labeling tool"
    )
    parser.add_argument(
        "--det-model-dir",
        type=Path,
        help="optional local PaddleOCR detection inference model directory",
    )
    parser.add_argument(
        "--det-model-name",
        default="PP-OCRv6_medium_det",
        help="PaddleOCR detection model name used when --det-model-dir is absent",
    )
    parser.add_argument("--device", default="gpu:0", help="gpu:<index> or cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument(
        "--images",
        type=Path,
        help="optional directory of direct-child images to open at startup",
    )
    parser.add_argument("--det-limit-side-len", type=int, default=1600)
    parser.add_argument(
        "--det-limit-type", choices=("min", "max"), default="max"
    )
    parser.add_argument("--det-thresh", type=float, default=0.3)
    parser.add_argument("--det-box-thresh", type=float, default=0.6)
    parser.add_argument("--det-unclip-ratio", type=float, default=1.5)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_settings(args: argparse.Namespace) -> LabelerSettings:
    return LabelerSettings(
        task="detection",
        det_model_dir=args.det_model_dir,
        det_model_name=args.det_model_name,
        device=args.device,
        host=args.host,
        port=args.port,
        text_det_limit_side_len=args.det_limit_side_len,
        text_det_limit_type=args.det_limit_type,
        text_det_thresh=args.det_thresh,
        text_det_box_thresh=args.det_box_thresh,
        text_det_unclip_ratio=args.det_unclip_ratio,
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

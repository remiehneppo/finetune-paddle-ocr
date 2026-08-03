#!/usr/bin/env python3
"""Prepare labeler exports and fine-tune PP-OCRv6 text detection.

The script deliberately starts from PaddleOCR's native PP-OCRv6 detection
configuration.  It validates and stages labeler annotations, changes only
training/runtime settings, and verifies that the model architecture remains
byte-for-byte equivalent as a Python object before launching training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from PIL import Image, UnidentifiedImageError


LOGGER = logging.getLogger("paddleocr_det_finetune")
DEFAULT_CONFIG = "configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml"
DEFAULT_PRETRAINED = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
    "official_pretrained_model/PP-OCRv6_medium_det_pretrained.pdparams"
)
LABELER_DIR_NAME = ".paddleocr-det-labeler"
LABEL_FILE_NAME = "det_labels.txt"
SUPPORTED_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetSource:
    root: Path
    labels: Path
    index: int


@dataclass(frozen=True)
class PreparedSample:
    image_path: str
    labels: tuple[dict[str, Any], ...]
    dataset_index: int
    source_image: str
    sha256: str


class RejectionReport:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("w", encoding="utf-8", newline="\n")
        self.counts: Counter[str] = Counter()

    def add(
        self,
        source: DatasetSource,
        line_number: int,
        reason: str,
        detail: str = "",
        box_index: int | None = None,
    ) -> None:
        self.counts[reason] += 1
        row: dict[str, Any] = {
            "dataset": str(source.root),
            "label_file": str(source.labels),
            "line_number": line_number,
            "reason": reason,
            "detail": detail[:500],
        }
        if box_index is not None:
            row["box_index"] = box_index
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "RejectionReport":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and mix PaddleOCR detection-labeler exports, then fine-tune "
            "the native PP-OCRv6 medium detector."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more labeler workspace directories, .paddleocr-det-labeler "
            "directories, or det_labels.txt files."
        ),
    )
    parser.add_argument(
        "--paddleocr-dir",
        type=Path,
        default=Path("./PaddleOCR"),
        help="PaddleOCR source checkout (default: ./PaddleOCR).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="New run directory (default: runs/vi_det_YYYYmmdd_HHMMSS).",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--pretrained-model", default=DEFAULT_PRETRAINED)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-step", type=int, default=200)
    parser.add_argument("--save-epoch-step", type=int, default=5)
    parser.add_argument("--max-image-pixels", type=int, default=50_000_000)
    parser.add_argument("--min-polygon-area", type=float, default=4.0)
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable mixed precision (AMP is enabled by default for RTX GPUs).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate/stage data and write the resolved config without training.",
    )
    parser.add_argument(
        "--export-after-train",
        action="store_true",
        help="Export output/best_accuracy to work-dir/inference/best_accuracy.",
    )
    return parser.parse_args(argv)


def resolve_config_path(paddle_dir: Path, config: str) -> Path:
    path = Path(config).expanduser()
    return path.resolve() if path.is_absolute() else (paddle_dir / path).resolve()


def resolve_dataset_source(raw_path: Path, index: int) -> DatasetSource:
    path = raw_path.expanduser().resolve()
    if path.is_file():
        if path.name != LABEL_FILE_NAME:
            raise ValueError(f"Expected {LABEL_FILE_NAME}, got: {path}")
        root = path.parent.parent if path.parent.name == LABELER_DIR_NAME else path.parent
        return DatasetSource(root=root.resolve(), labels=path, index=index)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    candidates = [path / LABELER_DIR_NAME / LABEL_FILE_NAME, path / LABEL_FILE_NAME]
    labels = next((candidate for candidate in candidates if candidate.is_file()), None)
    if labels is None:
        raise FileNotFoundError(
            f"No {LABEL_FILE_NAME} found under {path}. Open the detection labeler "
            "and click Export before fine-tuning."
        )
    root = labels.parent.parent if labels.parent.name == LABELER_DIR_NAME else labels.parent
    return DatasetSource(root=root.resolve(), labels=labels.resolve(), index=index)


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, list[DatasetSource]]:
    if not 0.0 < args.validation_ratio < 0.5:
        raise ValueError("--validation-ratio must be between 0 and 0.5")
    for name in (
        "epochs",
        "batch_size",
        "num_workers",
        "eval_batch_step",
        "save_epoch_step",
        "max_image_pixels",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.min_polygon_area <= 0:
        raise ValueError("learning rate and minimum polygon area must be positive")

    paddle_dir = args.paddleocr_dir.expanduser().resolve()
    if not (paddle_dir / "tools" / "train.py").is_file():
        raise FileNotFoundError(f"Not a PaddleOCR source checkout: {paddle_dir}")
    config_path = resolve_config_path(paddle_dir, args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"PaddleOCR config not found: {config_path}")
    sources = [
        resolve_dataset_source(path, index) for index, path in enumerate(args.dataset_dir)
    ]
    return paddle_dir, config_path, sources


def make_work_dir(path: Path | None) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("runs") / f"vi_det_{stamp}"
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Work directory is not empty: {resolved}. Use a new directory."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_image(path: Path, max_pixels: int) -> tuple[int, int, bytes, str]:
    data = path.read_bytes()
    if not data:
        raise ValueError("empty image file")
    try:
        with Image.open(path) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError("image has invalid dimensions")
            if width * height > max_pixels:
                raise ValueError(
                    f"image has {width * height} pixels, limit is {max_pixels}"
                )
            image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"cannot decode image: {exc}") from exc
    return width, height, data, _sha256(data)


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )


def _orientation(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else 2


def _segments_intersect(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    d: Sequence[float],
) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return o1 != o2 and o3 != o4 and 0 not in (o1, o2, o3, o4)


def normalize_polygon(
    raw_points: Any,
    width: int,
    height: int,
    min_area: float,
) -> list[list[int]]:
    if not isinstance(raw_points, list) or len(raw_points) < 4:
        raise ValueError("points must contain at least four vertices")
    points: list[list[int]] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
            raise ValueError("each point must be [x, y]")
        x, y = raw_point
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
            or not math.isfinite(float(x))
            or not math.isfinite(float(y))
        ):
            raise ValueError("coordinates must be finite numbers")
        points.append(
            [
                min(max(round(float(x)), 0), width - 1),
                min(max(round(float(y)), 0), height - 1),
            ]
        )
    if len({tuple(point) for point in points}) < 3:
        raise ValueError("polygon has fewer than three unique points")
    if len(points) == 4 and (
        _segments_intersect(points[0], points[1], points[2], points[3])
        or _segments_intersect(points[1], points[2], points[3], points[0])
    ):
        raise ValueError("polygon is self-intersecting")
    area = polygon_area(points)
    if area < min_area:
        raise ValueError(f"polygon area {area:.3f} is below {min_area}")
    return points


def normalize_label(
    raw_label: Any,
    width: int,
    height: int,
    min_area: float,
) -> dict[str, Any]:
    if not isinstance(raw_label, dict):
        raise ValueError("label entry must be an object")
    points = normalize_polygon(raw_label.get("points"), width, height, min_area)
    transcription = raw_label.get("transcription")
    ignored = isinstance(transcription, str) and transcription.strip() in {"*", "###"}
    return {"transcription": "###" if ignored else "text", "points": points}


def stage_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def process_source(
    source: DatasetSource,
    prepared_dir: Path,
    args: argparse.Namespace,
    report: RejectionReport,
    seen_hashes: set[str],
) -> list[PreparedSample]:
    samples: list[PreparedSample] = []
    with source.labels.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                report.add(source, line_number, "empty_line")
                continue
            try:
                relative_text, payload_text = line.split("\t", 1)
            except ValueError:
                report.add(source, line_number, "invalid_line", "missing tab separator")
                continue
            relative_path = Path(relative_text)
            if not relative_text or relative_path.is_absolute() or ".." in relative_path.parts:
                report.add(source, line_number, "unsafe_image_path", relative_text)
                continue
            try:
                image_path = (source.root / relative_path).resolve(strict=True)
            except OSError as exc:
                report.add(source, line_number, "missing_image", str(exc))
                continue
            if not _is_within(image_path, source.root) or not image_path.is_file():
                report.add(source, line_number, "unsafe_image_path", relative_text)
                continue
            try:
                width, height, _image_bytes, digest = inspect_image(
                    image_path, args.max_image_pixels
                )
            except (OSError, ValueError) as exc:
                report.add(source, line_number, "invalid_image", str(exc))
                continue
            if digest in seen_hashes:
                report.add(source, line_number, "duplicate_image", digest)
                continue
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                report.add(source, line_number, "invalid_json", str(exc))
                continue
            if not isinstance(payload, list) or not payload:
                report.add(source, line_number, "empty_boxes", "at least one box is required")
                continue

            normalized: list[dict[str, Any]] = []
            for box_index, raw_label in enumerate(payload):
                try:
                    normalized.append(
                        normalize_label(
                            raw_label,
                            width,
                            height,
                            args.min_polygon_area,
                        )
                    )
                except ValueError as exc:
                    report.add(
                        source,
                        line_number,
                        "invalid_box",
                        str(exc),
                        box_index,
                    )
            if not normalized:
                report.add(source, line_number, "no_valid_boxes")
                continue

            suffix = image_path.suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                suffix = ".img"
            staged_relative = Path("images") / f"d{source.index:02d}_{digest[:20]}{suffix}"
            stage_image(image_path, prepared_dir / staged_relative)
            seen_hashes.add(digest)
            samples.append(
                PreparedSample(
                    image_path=staged_relative.as_posix(),
                    labels=tuple(normalized),
                    dataset_index=source.index,
                    source_image=str(image_path),
                    sha256=digest,
                )
            )
    return samples


def split_train_validation(
    samples: Sequence[PreparedSample], validation_ratio: float, seed: int
) -> tuple[list[PreparedSample], list[PreparedSample]]:
    if len(samples) < 2:
        raise RuntimeError("At least two valid, unique labeled images are required")
    grouped: dict[int, list[PreparedSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.dataset_index].append(sample)

    train: list[PreparedSample] = []
    validation: list[PreparedSample] = []
    for dataset_index, group in sorted(grouped.items()):
        random.Random(seed + dataset_index).shuffle(group)
        if len(group) < 2:
            train.extend(group)
            continue
        validation_count = max(1, round(len(group) * validation_ratio))
        validation_count = min(validation_count, len(group) - 1)
        validation.extend(group[:validation_count])
        train.extend(group[validation_count:])

    if not validation:
        random.Random(seed).shuffle(train)
        validation.append(train.pop())
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(validation)
    return train, validation


def write_label_file(path: Path, samples: Iterable[PreparedSample]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            payload = json.dumps(
                list(sample.labels), ensure_ascii=False, separators=(",", ":")
            )
            handle.write(f"{sample.image_path}\t{payload}\n")


def prepare_datasets(
    sources: Sequence[DatasetSource], args: argparse.Namespace, work_dir: Path
) -> dict[str, Any]:
    prepared_dir = work_dir / "prepared"
    prepared_dir.mkdir()
    samples: list[PreparedSample] = []
    per_dataset: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    with RejectionReport(prepared_dir / "rejected.jsonl") as report:
        for source in sources:
            LOGGER.info("Loading detection dataset %d: %s", source.index, source.root)
            dataset_samples = process_source(
                source, prepared_dir, args, report, seen_hashes
            )
            samples.extend(dataset_samples)
            per_dataset.append(
                {
                    "dataset": str(source.root),
                    "label_file": str(source.labels),
                    "valid_samples": len(dataset_samples),
                }
            )
        rejection_counts = dict(sorted(report.counts.items()))

    train, validation = split_train_validation(
        samples, args.validation_ratio, args.seed
    )
    write_label_file(prepared_dir / "train.txt", train)
    write_label_file(prepared_dir / "validation.txt", validation)
    summary = {
        "task": "text_detection",
        "model": "PP-OCRv6_medium_det",
        "train_samples": len(train),
        "validation_samples": len(validation),
        "unique_images": len(samples),
        "rejection_events": sum(rejection_counts.values()),
        "rejection_counts": rejection_counts,
        "datasets": per_dataset,
        "split_seed": args.seed,
        "validation_ratio": args.validation_ratio,
    }
    with (prepared_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def _transform_names(transforms: list[dict[str, Any]]) -> list[str]:
    return [next(iter(transform)) for transform in transforms]


def create_resolved_config(
    source_path: Path,
    target_path: Path,
    prepared_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    with source_path.open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle)
    config = copy.deepcopy(source)
    original_architecture = copy.deepcopy(source["Architecture"])
    original_loss = copy.deepcopy(source["Loss"])
    original_train_transforms = _transform_names(source["Train"]["dataset"]["transforms"])
    original_eval_transforms = _transform_names(source["Eval"]["dataset"]["transforms"])

    config["Global"].update(
        {
            "use_gpu": True,
            "epoch_num": args.epochs,
            "save_model_dir": str(output_dir),
            "save_epoch_step": args.save_epoch_step,
            "eval_batch_step": [0, args.eval_batch_step],
            "cal_metric_during_train": False,
            "checkpoints": None,
            "pretrained_model": args.pretrained_model,
            "use_visualdl": True,
            "use_amp": not args.disable_amp,
            "scale_loss": 512.0,
            "use_dynamic_loss_scaling": True,
            "seed": args.seed,
        }
    )
    config["Optimizer"]["lr"]["learning_rate"] = args.learning_rate
    config["Optimizer"]["lr"]["warmup_epoch"] = min(2, args.epochs)

    train = config["Train"]
    train["dataset"]["data_dir"] = str(prepared_dir)
    train["dataset"]["label_file_list"] = [str(prepared_dir / "train.txt")]
    train["dataset"]["ratio_list"] = [1.0]
    train["loader"]["batch_size_per_card"] = args.batch_size
    train["loader"]["num_workers"] = args.num_workers
    for transform in train["dataset"]["transforms"]:
        for name in ("MakeBorderMap", "MakeShrinkMap"):
            if name in transform and isinstance(transform[name], dict):
                transform[name]["total_epoch"] = args.epochs

    evaluation = config["Eval"]
    evaluation["dataset"]["data_dir"] = str(prepared_dir)
    evaluation["dataset"]["label_file_list"] = [
        str(prepared_dir / "validation.txt")
    ]
    evaluation["loader"]["batch_size_per_card"] = 1
    evaluation["loader"]["num_workers"] = max(1, args.num_workers // 2)

    if config["Architecture"] != original_architecture:
        raise RuntimeError("Refusing to write config: model Architecture changed")
    if config["Loss"] != original_loss:
        raise RuntimeError("Refusing to write config: detection Loss changed")
    if _transform_names(train["dataset"]["transforms"]) != original_train_transforms:
        raise RuntimeError("Refusing to write config: training transform chain changed")
    if _transform_names(evaluation["dataset"]["transforms"]) != original_eval_transforms:
        raise RuntimeError("Refusing to write config: evaluation transform chain changed")

    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def download_pretrained(value: str, work_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in ("http", "https"):
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            raise ValueError(
                "--pretrained-model must be a training .pdparams file, not an "
                "inference model directory"
            )
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained model not found: {path}")
        if path.suffix != ".pdparams":
            raise ValueError("--pretrained-model must point to a .pdparams file")
        return path

    destination = work_dir / "pretrained" / Path(parsed.path).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    LOGGER.info("Downloading official pretrained weights to %s", destination)
    try:
        urllib.request.urlretrieve(value, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def verify_paddle_gpu() -> None:
    try:
        import paddle
    except ImportError as exc:
        raise RuntimeError("PaddlePaddle is not installed in this Python environment") from exc
    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise RuntimeError("PaddlePaddle cannot see a CUDA GPU")
    try:
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=512.0, use_dynamic_loss_scaling=True
        )
        float(scaler._scale)
    except TypeError as exc:
        try:
            import numpy

            numpy_version = numpy.__version__
        except ImportError:
            numpy_version = "unknown"
        raise RuntimeError(
            "Paddle GradScaler is incompatible with NumPy "
            f"{numpy_version}; install the pinned dependencies first"
        ) from exc


def verify_weight_compatibility(
    paddle_dir: Path, config_path: Path, weights_path: Path
) -> None:
    import paddle

    sys.path.insert(0, str(paddle_dir))
    try:
        from ppocr.modeling.architectures import build_model

        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        paddle.set_device("cpu")
        model_state = build_model(config["Architecture"]).state_dict()
        loaded = paddle.load(str(weights_path))
    finally:
        if sys.path and sys.path[0] == str(paddle_dir):
            sys.path.pop(0)

    if isinstance(loaded, dict):
        for key in ("state_dict", "model"):
            if key in loaded and isinstance(loaded[key], dict):
                loaded = loaded[key]
                break
    if not isinstance(loaded, dict):
        raise RuntimeError("Pretrained checkpoint is not a Paddle state dictionary")

    loaded_shapes = {
        key: tuple(value.shape)
        for key, value in loaded.items()
        if hasattr(value, "shape")
    }
    model_shapes = {key: tuple(value.shape) for key, value in model_state.items()}
    missing = sorted(set(model_shapes) - set(loaded_shapes))
    mismatched = sorted(
        key
        for key in set(model_shapes) & set(loaded_shapes)
        if model_shapes[key] != loaded_shapes[key]
    )
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:8]}")
        if mismatched:
            details.append(
                "mismatched="
                + repr(
                    [
                        (key, model_shapes[key], loaded_shapes[key])
                        for key in mismatched[:8]
                    ]
                )
            )
        raise RuntimeError(
            "Pretrained weights do not exactly match PP-OCRv6 detector architecture: "
            + "; ".join(details)
        )
    LOGGER.info("Verified %d pretrained tensors against the model", len(model_shapes))


def export_best_model(paddle_dir: Path, config_path: Path, work_dir: Path) -> None:
    output_dir = work_dir / "output"
    candidates = [output_dir / "best_accuracy", output_dir / "best_model" / "model"]
    checkpoint = next(
        (candidate for candidate in candidates if candidate.with_suffix(".pdparams").is_file()),
        None,
    )
    if checkpoint is None:
        raise RuntimeError("Training finished but no best checkpoint was found")
    inference_dir = work_dir / "inference" / "best_accuracy"
    command = [
        sys.executable,
        "tools/export_model.py",
        "-c",
        str(config_path),
        "-o",
        f"Global.pretrained_model={checkpoint}",
        f"Global.save_inference_dir={inference_dir}",
    ]
    LOGGER.info("Exporting inference model: %s", " ".join(command))
    subprocess.run(command, cwd=paddle_dir, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paddle_dir, config_source, sources = validate_args(args)
    work_dir = make_work_dir(args.work_dir)
    summary = prepare_datasets(sources, args, work_dir)
    config_target = work_dir / "resolved_config.yml"
    create_resolved_config(
        config_source,
        config_target,
        work_dir / "prepared",
        work_dir / "output",
        args,
    )
    LOGGER.info("Prepared summary: %s", json.dumps(summary, ensure_ascii=False))
    LOGGER.info("Resolved config: %s", config_target)
    if args.prepare_only:
        return 0

    verify_paddle_gpu()
    pretrained = download_pretrained(args.pretrained_model, work_dir)
    with config_target.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["Global"]["pretrained_model"] = str(pretrained)
    with config_target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    verify_weight_compatibility(paddle_dir, config_target, pretrained)

    command = [sys.executable, "tools/train.py", "-c", str(config_target)]
    LOGGER.info("Starting training: %s", " ".join(command))
    subprocess.run(command, cwd=paddle_dir, check=True)
    if args.export_after_train:
        export_best_model(paddle_dir, config_target, work_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(2)

#!/usr/bin/env python3
"""Prepare Vietnamese OCR data and run PaddleOCR-VL-1.6 ERNIEKit LoRA SFT."""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Self

import yaml
from PIL import Image, UnidentifiedImageError

import finetune as rec_loader
from paddleocr_vl_tasks import (
    TASK_PROMPTS,
    prompt_for_task,
    resolve_row_task,
    task_for_prompt,
    validate_target_for_task,
)

LOGGER = logging.getLogger("paddleocr_vl_vi_finetune")
PROMPT = TASK_PROMPTS["ocr"]
DEFAULT_MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
DEFAULT_MIN_PIXELS = 64 * 28 * 28
DEFAULT_MAX_PIXELS = 576 * 28 * 28
ERNIEKIT_COMPAT_DIR = Path(__file__).with_name("erniekit_compat").resolve()
LORA_SCOPE_MARKER = "PADDLEOCR_VL_LORA_SCOPE=text_decoder_only"
ERNIEKIT_REVISION = "790a50b045d1aca2753d5395d8bec0806b2e6925"
ERNIEKIT_RUNTIME_VERSIONS = {
    "paddlepaddle-gpu": "3.2.1",
    "paddleformers": "0.4.0",
    "safetensors": "0.7.0",
    "transformers": "4.55.4",
    "ml_dtypes": "0.5.4",
}
TRAINABLE_LINE = re.compile(
    r"trainable\s+parameters?\s*[:=]\s*([0-9,.+e-]+).*?"
    r"(?:all|total)\s+parameters?\s*[:=]\s*([0-9,.+e-]+)",
    re.IGNORECASE | re.DOTALL,
)

TRAINABLE_NAMES_LINE = re.compile(
    r"^PADDLEOCR_VL_TRAINABLE_PARAMETER_NAMES=(\[.*\])",
    re.MULTILINE,
)

load_hf_dataset = rec_loader.load_hf_dataset
select_splits = rec_loader.select_splits
normalize_text = rec_loader.normalize_text


def normalize_target(row: Mapping[str, Any], task: str = "ocr") -> str:
    value: str | None = None
    for column in ("label", "text"):
        candidate = row.get(column)
        if isinstance(candidate, str) and candidate.strip():
            value = candidate
            break
    if value is None:
        return ""
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return value if value.strip() else ""


@dataclass(frozen=True)
class PreparedSample:
    image_path: str
    text: str
    dataset_index: int
    prompt: str = PROMPT


@dataclass(frozen=True)
class TrainableParameterReport:
    trainable: int
    total: int
    names: tuple[str, ...]

    @property
    def fraction(self) -> float:
        return self.trainable / self.total if self.total else 0.0


class PixelLimitExceeded(ValueError):
    pass


class RejectionReport:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("w", encoding="utf-8", newline="\n")
        self.counts: Counter[str] = Counter()

    def reject(
        self,
        dataset: Path,
        split: str,
        row_index: int,
        reason: str,
        detail: str = "",
    ) -> None:
        self.counts[reason] += 1
        record = {
            "dataset": str(dataset),
            "split": split,
            "row_index": row_index,
            "reason": reason,
            "detail": detail[:500],
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare OCR/layout datasets and LoRA fine-tune PaddleOCR-VL-1.6 with "
            "ERNIEKit. Mix ocr/table/formula/chart in one run via a per-sample "
            "'task' column; targets stay in the dataset text/label fields."
        )
    )
    parser.add_argument(
        "--task",
        choices=tuple(TASK_PROMPTS),
        default="ocr",
        help=(
            "Default task prompt for rows without a 'task' column. "
            "Use a dataset 'task' column to mix layout types in one run."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, nargs="+")
    parser.add_argument(
        "--dataset-task",
        choices=tuple(TASK_PROMPTS),
        nargs="+",
        help=(
            "Default task for each --dataset-dir, in the same order. Required "
            "when multiple sources omit a per-row 'task' column."
        ),
    )
    parser.add_argument(
        "--prepared-from",
        type=Path,
        nargs="+",
        help=(
            "Reuse JSONL and images from one or more existing --prepare-only "
            "runs. Multiple runs require matching --prepared-weight values."
        ),
    )
    parser.add_argument(
        "--prepared-weight",
        type=float,
        nargs="+",
        metavar="WEIGHT",
        help=(
            "Positive relative weight for each --prepared-from run. Required "
            "for multiple runs and normalized automatically."
        ),
    )
    parser.add_argument(
        "--erniekit-dir",
        type=Path,
        help="ERNIEKit release/v1.5 source checkout (required unless --prepare-only).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--inspect-model",
        action="store_true",
        help="Load and inspect LoRA trainable parameters, then exit before training.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--max-image-pixels", type=int, default=50_000_000)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-samples-per-dataset", type=int, default=32)
    parser.add_argument("--eval-max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--eval-task-max-new-tokens",
        action="append",
        default=[],
        metavar="TASK=COUNT",
        help="Override generation limit per task; may be repeated.",
    )
    parser.add_argument("--eval-max-checkpoints", type=int, default=3)
    parser.add_argument("--min-normalized-edit-distance", type=float, default=0.5)
    parser.add_argument("--max-cer", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        help="Comma-separated CUDA devices. Defaults to one GPU (0).",
    )
    parser.add_argument(
        "--no-flash-attention", dest="flash_attention", action="store_false"
    )
    parser.set_defaults(flash_attention=True)
    args = parser.parse_args(argv)
    if args.prepared_from is not None and len(args.prepared_from) == 1:
        args.prepared_from = args.prepared_from[0]
    return args


def parse_task_token_limits(values: Sequence[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for value in values:
        task, separator, raw_count = value.partition("=")
        if not separator:
            raise ValueError("task token limits must use TASK=COUNT")
        task = task.strip()
        prompt_for_task(task)
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(f"invalid token limit for task {task!r}") from exc
        if count <= 0:
            raise ValueError(f"token limit for task {task!r} must be positive")
        if task in limits:
            raise ValueError(f"duplicate token limit for task {task!r}")
        limits[task] = count
    return limits


def normalize_prepared_weights(
    prepared_runs: Sequence[Path], prepared_weights: Sequence[float] | None
) -> list[float]:
    if not prepared_runs:
        raise ValueError("At least one --prepared-from run is required")
    if prepared_weights is None:
        if len(prepared_runs) > 1:
            raise ValueError(
                "Multiple --prepared-from runs require one --prepared-weight "
                "value per run"
            )
        return [1.0]
    weights = list(prepared_weights)
    if len(weights) != len(prepared_runs):
        raise ValueError(
            "--prepared-weight must contain one value per --prepared-from run"
        )
    if not all(math.isfinite(weight) and weight > 0 for weight in weights):
        raise ValueError("--prepared-weight values must be finite and positive")
    total_weight = sum(weights)
    return [weight / total_weight for weight in weights]


def validate_args(args: argparse.Namespace) -> None:
    prompt_for_task(args.task)
    has_datasets = bool(args.dataset_dir)
    has_prepared = args.prepared_from is not None
    prepared_runs = (
        [args.prepared_from]
        if isinstance(args.prepared_from, Path)
        else list(args.prepared_from or ())
    )
    if has_datasets and has_prepared:
        raise ValueError("--dataset-dir and --prepared-from cannot be used together")
    if args.resume_from is None and not has_datasets and not has_prepared:
        raise ValueError("A fresh run requires --dataset-dir or --prepared-from")
    if has_prepared and args.prepare_only:
        raise ValueError("--prepared-from cannot be used with --prepare-only")
    if has_prepared and args.resume_from is not None:
        raise ValueError("--prepared-from cannot be used with --resume-from")
    if args.prepared_weight is not None and not has_prepared:
        raise ValueError("--prepared-weight requires --prepared-from")
    if has_prepared:
        args.prepared_weight = normalize_prepared_weights(
            prepared_runs, args.prepared_weight
        )
    if args.dataset_task and not has_datasets:
        raise ValueError("--dataset-task requires --dataset-dir")
    if args.dataset_task and len(args.dataset_task) != len(args.dataset_dir):
        raise ValueError("--dataset-task must contain one task per --dataset-dir")

    missing = [str(path) for path in (args.dataset_dir or ()) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Dataset directories not found: {missing}")
    missing_prepared = [str(path) for path in prepared_runs if not path.is_dir()]
    if missing_prepared:
        raise FileNotFoundError(
            f"Prepared run directories not found: {missing_prepared}"
        )
    if not 0.0 < args.validation_ratio < 0.5:
        raise ValueError("--validation-ratio must be between 0 and 0.5")
    positive = (
        "epochs",
        "learning_rate",
        "lora_rank",
        "min_pixels",
        "max_pixels",
        "max_image_pixels",
        "max_seq_len",
        "gradient_accumulation_steps",
        "num_workers",
        "prefetch_factor",
        "eval_samples_per_dataset",
        "eval_max_new_tokens",
        "eval_max_checkpoints",
        "save_steps",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.min_pixels > args.max_pixels:
        raise ValueError("--min-pixels cannot exceed --max-pixels")
    if not 0.0 <= args.min_normalized_edit_distance <= 1.0:
        raise ValueError("--min-normalized-edit-distance must be between 0 and 1")
    if args.max_cer < 0.0:
        raise ValueError("--max-cer must be non-negative")
    parse_task_token_limits(args.eval_task_max_new_tokens)
    if args.smoke_steps is not None and args.smoke_steps <= 0:
        raise ValueError("--smoke-steps must be positive")
    selected_devices(args.devices)
    if not args.prepare_only:
        if args.erniekit_dir is None:
            raise ValueError("--erniekit-dir is required for model inspection/training")
        validate_erniekit_source(args.erniekit_dir)
        args.model = str(require_local_model_snapshot(args.model))


def make_work_dir(path: Path | None, resume_from: Path | None) -> Path:
    if path is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        path = Path("runs") / f"paddleocr_vl_1_6_vi_lora_{stamp}"
    resolved = path.expanduser().resolve()
    if resume_from is None:
        if resolved.exists() and any(resolved.iterdir()):
            raise FileExistsError(
                f"Work directory is not empty: {resolved}. Use a new run or --resume-from."
            )
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    checkpoint = resume_from.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Resume work directory not found: {resolved}")
    try:
        checkpoint.relative_to(resolved)
    except ValueError as exc:
        raise ValueError("--resume-from must be inside --work-dir") from exc
    return resolved


def load_tokenizer(model: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing preparation dependency 'transformers'. Install requirements-vl-prepare.txt."
        ) from exc
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def require_local_model_snapshot(model: str) -> Path:
    path = Path(model).expanduser().resolve()
    required = (path / "config.json", path / "preprocessor_config.json")
    weights = tuple(path.glob("model*.safetensors")) if path.is_dir() else ()
    tokenizers = tuple(path.glob("tokenizer.*")) if path.is_dir() else ()
    if (
        not path.is_dir()
        or any(not candidate.is_file() for candidate in required)
        or not weights
        or not tokenizers
    ):
        raise FileNotFoundError(
            "Training requires a complete local PaddleOCR-VL-1.6 snapshot with "
            "config, preprocessor, weights, and tokenizer; run "
            "./download_pretrained_models.sh vl ./models and pass "
            "--model ./models/PaddleOCR-VL-1.6"
        )
    return path


def count_tokens(tokenizer: Any, text: str, prompt: str = PROMPT) -> int:
    def encode(value: str) -> Sequence[int]:
        if hasattr(tokenizer, "encode"):
            encoded = tokenizer.encode(value, add_special_tokens=False)
        else:
            encoded = tokenizer(value, add_special_tokens=False)
        if isinstance(encoded, Mapping):
            return encoded["input_ids"]
        return encoded.input_ids if hasattr(encoded, "input_ids") else encoded

    return len(encode(prompt)) + len(encode(text)) + 1


def smart_resize_dimensions(
    height: int,
    width: int,
    *,
    min_pixels: int,
    max_pixels: int,
    factor: int = 28,
) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if height < factor:
        width = round(width * factor / height)
        height = factor
    if width < factor:
        height = round(height * factor / width)
        width = factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute image aspect ratio must be smaller than 200")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        resized_height = math.floor(height / scale / factor) * factor
        resized_width = math.floor(width / scale / factor) * factor
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    if resized_height <= 0 or resized_width <= 0:
        raise ValueError("image becomes empty after PaddleOCR-VL smart resize")
    return resized_height, resized_width


def visual_token_count(
    height: int,
    width: int,
    *,
    min_pixels: int,
    max_pixels: int,
    patch_size: int = 14,
    merge_size: int = 2,
) -> int:
    resized_height, resized_width = smart_resize_dimensions(
        height,
        width,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        factor=patch_size * merge_size,
    )
    grid_height = resized_height // patch_size
    grid_width = resized_width // patch_size
    return grid_height * grid_width // (merge_size**2)


def total_multimodal_tokens(
    tokenizer: Any, text: str, *, visual_tokens: int, prompt: str = PROMPT
) -> int:
    return count_tokens(tokenizer, text, prompt) + visual_tokens


def selected_devices(value: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in value.split(",") if part.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--devices must be a non-empty comma-separated unique list")
    return devices


def _checked_rgb(image: Image.Image, max_image_pixels: int) -> Image.Image:
    width, height = image.size
    if width <= 1 or height <= 1:
        raise ValueError(f"invalid dimensions {width}x{height}")
    pixels = width * height
    if pixels > max_image_pixels:
        raise PixelLimitExceeded(
            f"image has {pixels} pixels; limit is {max_image_pixels}"
        )
    image.load()
    return image.convert("RGB")


def open_image_rgb(value: Any, dataset_dir: Path, max_image_pixels: int) -> Image.Image:
    if isinstance(value, Image.Image):
        return _checked_rgb(value.copy(), max_image_pixels)
    if isinstance(value, Mapping):
        raw_bytes = value.get("bytes")
        if raw_bytes is not None:
            with Image.open(io.BytesIO(raw_bytes)) as image:
                return _checked_rgb(image, max_image_pixels)
        value = value.get("path")
    if isinstance(value, (str, os.PathLike)):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = dataset_dir / path
        with Image.open(path) as image:
            return _checked_rgb(image, max_image_pixels)
    if hasattr(value, "__array_interface__"):
        return _checked_rgb(Image.fromarray(value), max_image_pixels)
    raise TypeError(f"Unsupported image value type: {type(value).__name__}")


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG", compress_level=3)
    os.replace(temporary, path)


def process_split(
    split: Any,
    dataset_dir: Path,
    dataset_index: int,
    split_name: str,
    prepared_dir: Path,
    max_image_pixels: int,
    max_seq_len: int,
    tokenizer: Any,
    report: RejectionReport,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    default_task: str = "ocr",
) -> list[PreparedSample]:
    columns = set(getattr(split, "column_names", []))
    if columns and "image" not in columns:
        raise ValueError(f"{dataset_dir}/{split_name} has no 'image' column")
    if columns and not columns.intersection({"label", "text"}):
        raise ValueError(f"{dataset_dir}/{split_name} has no 'label' or 'text' column")

    samples: list[PreparedSample] = []
    for row_index in range(len(split)):
        try:
            row = split[row_index]
        except Exception as exc:  # noqa: BLE001 - decoders may raise arbitrary errors
            report.reject(
                dataset_dir, split_name, row_index, "row_load_error", str(exc)
            )
            continue
        try:
            task = resolve_row_task(row, default_task)
        except (TypeError, ValueError) as exc:
            report.reject(dataset_dir, split_name, row_index, "invalid_task", str(exc))
            continue
        prompt = prompt_for_task(task)
        text = normalize_target(row, task)
        if not text:
            report.reject(dataset_dir, split_name, row_index, "empty_text")
            continue
        try:
            validate_target_for_task(text, task)
        except ValueError as exc:
            report.reject(
                dataset_dir,
                split_name,
                row_index,
                "invalid_target_schema",
                str(exc),
            )
            continue
        if any(
            unicodedata.category(character) == "Cc"
            and character != "\n"
            for character in text
        ):
            report.reject(dataset_dir, split_name, row_index, "control_character")
            continue
        image: Image.Image | None = None
        try:
            image = open_image_rgb(row.get("image"), dataset_dir, max_image_pixels)
            image_visual_tokens = visual_token_count(
                image.height,
                image.width,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            token_count = total_multimodal_tokens(
                tokenizer,
                text,
                visual_tokens=image_visual_tokens,
                prompt=prompt,
            )
            if token_count > max_seq_len:
                report.reject(
                    dataset_dir,
                    split_name,
                    row_index,
                    "token_budget_exceeded",
                    f"{token_count} > {max_seq_len}",
                )
                continue
            relative_path = (
                Path("images")
                / f"source-{dataset_index:03d}"
                / (f"{split_name}-{row_index:09d}.png")
            )
            _save_png(image, prepared_dir / relative_path)
        except PixelLimitExceeded as exc:
            report.reject(
                dataset_dir,
                split_name,
                row_index,
                "pixel_limit_exceeded",
                str(exc),
            )
            continue
        except (OSError, ValueError, TypeError, UnidentifiedImageError) as exc:
            report.reject(dataset_dir, split_name, row_index, "invalid_image", str(exc))
            continue
        finally:
            if image is not None:
                image.close()
        samples.append(
            PreparedSample(relative_path.as_posix(), text, dataset_index, prompt)
        )
    return samples


def split_train_validation(
    samples: Sequence[PreparedSample], validation_ratio: float, seed: int
) -> tuple[list[PreparedSample], list[PreparedSample]]:
    return rec_loader.split_train_validation(samples, validation_ratio, seed)


def dataset_default_task(
    args: argparse.Namespace,
    dataset_index: int,
    train_split: Any,
    validation_split: Any | None,
) -> str:
    dataset_tasks = getattr(args, "dataset_task", None)
    if dataset_tasks:
        return dataset_tasks[dataset_index]
    splits = [train_split]
    if validation_split is not None:
        splits.append(validation_split)
    missing_task_column = any(
        "task" not in set(getattr(split, "column_names", [])) for split in splits
    )
    if len(args.dataset_dir) > 1 and missing_task_column:
        raise ValueError(
            "Multiple dataset sources without a per-row 'task' column require "
            "one --dataset-task value per --dataset-dir"
        )
    return getattr(args, "task", "ocr")


def sqrt_probabilities(sample_counts: Sequence[int]) -> list[float]:
    if not sample_counts or any(count <= 0 for count in sample_counts):
        raise ValueError("All source sample counts must be positive")
    roots = [math.sqrt(count) for count in sample_counts]
    total = sum(roots)
    return [value / total for value in roots]


def erniekit_payload(sample: PreparedSample) -> dict[str, Any]:
    return {
        "image_info": [{"image_url": sample.image_path, "matched_text_index": 0}],
        "text_info": [
            {"text": sample.prompt, "tag": "mask"},
            {"text": sample.text, "tag": "no_mask"},
        ],
    }


def write_erniekit_jsonl(path: Path, samples: Iterable[PreparedSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(
                json.dumps(erniekit_payload(sample), ensure_ascii=False) + "\n"
            )


def prepare_datasets(
    args: argparse.Namespace, work_dir: Path, tokenizer: Any
) -> dict[str, Any]:
    observed_tasks: set[str] = set()
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = work_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    source_summaries: list[dict[str, Any]] = []
    with RejectionReport(work_dir / "rejected.jsonl") as report:
        for dataset_index, dataset_dir in enumerate(args.dataset_dir):
            dataset_dir = dataset_dir.expanduser().resolve()
            dataset = load_hf_dataset(dataset_dir)
            train_split, validation_split, train_name, validation_name = select_splits(
                dataset
            )
            default_task = dataset_default_task(
                args, dataset_index, train_split, validation_split
            )
            train_samples = process_split(
                train_split,
                dataset_dir,
                dataset_index,
                train_name,
                prepared_dir,
                args.max_image_pixels,
                args.max_seq_len,
                tokenizer,
                report,
                args.min_pixels,
                args.max_pixels,
                default_task,
            )
            observed_tasks.update(task_for_prompt(sample.prompt) for sample in train_samples)
            if validation_split is None:
                train_samples, validation_samples = split_train_validation(
                    train_samples,
                    args.validation_ratio,
                    args.seed + dataset_index,
                )
                validation_name = "holdout"
            else:
                validation_samples = process_split(
                    validation_split,
                    dataset_dir,
                    dataset_index,
                    validation_name or "validation",
                    prepared_dir,
                    args.max_image_pixels,
                    args.max_seq_len,
                    tokenizer,
                    report,
                    args.min_pixels,
                    args.max_pixels,
                    default_task,
                )
            observed_tasks.update(
                task_for_prompt(sample.prompt) for sample in validation_samples
            )
            if not train_samples:
                raise ValueError(f"No valid training samples remain for {dataset_dir}")
            if not validation_samples:
                raise ValueError(
                    f"No valid validation samples remain for {dataset_dir}"
                )

            train_path = prepared_dir / f"train-source-{dataset_index:03d}.jsonl"
            validation_path = (
                prepared_dir / f"validation-source-{dataset_index:03d}.jsonl"
            )
            write_erniekit_jsonl(train_path, train_samples)
            write_erniekit_jsonl(validation_path, validation_samples)
            source_summaries.append(
                {
                    "dataset": str(dataset_dir),
                    "default_task": default_task,
                    "train_split": train_name,
                    "validation_split": validation_name,
                    "train_samples": len(train_samples),
                    "validation_samples": len(validation_samples),
                    "train_jsonl": str(train_path.resolve()),
                    "validation_jsonl": str(validation_path.resolve()),
                }
            )

        train_probabilities = sqrt_probabilities(
            [source["train_samples"] for source in source_summaries]
        )
        validation_probabilities = sqrt_probabilities(
            [source["validation_samples"] for source in source_summaries]
        )
        summary = {
            "tasks": sorted(observed_tasks),
            "prompts": [prompt_for_task(task) for task in sorted(observed_tasks)],
            "model": getattr(args, "model", DEFAULT_MODEL),
            "sources": source_summaries,
            "train_samples": sum(
                source["train_samples"] for source in source_summaries
            ),
            "validation_samples": sum(
                source["validation_samples"] for source in source_summaries
            ),
            "train_probabilities": train_probabilities,
            "validation_probabilities": validation_probabilities,
            "rejected": dict(sorted(report.counts.items())),
        }
        if len(observed_tasks) == 1:
            only_task = next(iter(observed_tasks))
            summary["task"] = only_task
            summary["prompt"] = prompt_for_task(only_task)
        else:
            summary["task"] = "mixed"
    (work_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _resolve_reused_path(value: Any, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Prepared summary has invalid {field}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _validate_prepared_jsonl(
    path: Path, allowed_prompts: Sequence[str] | None = None
) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared JSONL not found: {path}")

    allowed = (
        set(TASK_PROMPTS.values())
        if allowed_prompts is None
        else set(allowed_prompts)
    )
    if not allowed:
        raise ValueError(
            "Prepared JSONL validation requires at least one allowed prompt"
        )

    count = 0
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise TypeError(f"Invalid sample in {path}:{line_number}")

            image_info = payload.get("image_info")
            text_info = payload.get("text_info")
            if not isinstance(image_info, list) or len(image_info) != 1:
                raise ValueError(f"Invalid image_info contract in {path}:{line_number}")
            if not isinstance(text_info, list) or len(text_info) != 2:
                raise ValueError(f"Invalid text_info contract in {path}:{line_number}")
            image = image_info[0]
            prompt_row, target = text_info
            if (
                not isinstance(image, Mapping)
                or image.get("matched_text_index") != 0
                or not isinstance(prompt_row, Mapping)
                or prompt_row.get("tag") != "mask"
                or not isinstance(prompt_row.get("text"), str)
                or prompt_row["text"] not in allowed
                or not isinstance(target, Mapping)
                or target.get("tag") != "no_mask"
                or not isinstance(target.get("text"), str)
                or not target["text"]
            ):
                raise ValueError(f"Invalid task mask contract in {path}:{line_number}")
            task = task_for_prompt(prompt_row["text"])
            try:
                validate_target_for_task(target["text"], task)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {task} target schema in {path}:{line_number}: {exc}"
                ) from exc

            image_path = _resolve_reused_path(
                image.get("image_url"), path.parent, "image_url"
            )
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Prepared image not found at {path}:{line_number}: {image_path}"
                )
            count += 1
    return count


def _normalize_summary_tasks(summary: dict[str, Any]) -> list[str]:
    raw_tasks = summary.get("tasks")
    if isinstance(raw_tasks, list) and raw_tasks:
        normalized: list[str] = []
        for task in raw_tasks:
            if not isinstance(task, str):
                raise TypeError("Prepared summary tasks must be strings")
            prompt_for_task(task)
            normalized.append(task)
        tasks = sorted(set(normalized))
    else:
        legacy_task = summary.get("task", "ocr")
        if legacy_task == "mixed":
            raise ValueError("Prepared summary task='mixed' requires a non-empty tasks list")
        if not isinstance(legacy_task, str):
            raise TypeError("Prepared summary task must be a string")
        prompt_for_task(legacy_task)
        tasks = [legacy_task]

    prompts = [prompt_for_task(task) for task in tasks]
    if "prompt" in summary and summary["prompt"] not in prompts:
        raise ValueError(
            f"Prepared summary prompt {summary['prompt']!r} does not match tasks {tasks}"
        )
    if "prompts" in summary:
        recorded = summary["prompts"]
        if not isinstance(recorded, list) or set(recorded) != set(prompts):
            raise ValueError("Prepared summary prompts do not match tasks")
    summary["tasks"] = tasks
    summary["prompts"] = prompts
    if len(tasks) == 1:
        summary["task"] = tasks[0]
        summary["prompt"] = prompts[0]
    else:
        summary["task"] = "mixed"
        summary.pop("prompt", None)
    return tasks


def read_prepared_run(prepared_from: Path) -> dict[str, Any]:
    prepared_run = prepared_from.expanduser().resolve()
    summary_path = prepared_run / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Prepared run is missing summary.json: {prepared_run}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid prepared summary: {summary_path}") from exc
    if not isinstance(summary, dict):
        raise TypeError(f"Prepared summary must be a JSON object: {summary_path}")
    tasks = _normalize_summary_tasks(summary)
    allowed_prompts = [prompt_for_task(task) for task in tasks]

    sources = summary.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Prepared summary must contain at least one source")
    for field in ("train_probabilities", "validation_probabilities"):
        probabilities = summary.get(field)
        if not isinstance(probabilities, list) or len(probabilities) != len(sources):
            raise ValueError(f"Prepared summary has invalid {field}")
        if not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0
            for value in probabilities
        ) or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Prepared summary has invalid {field}")

    totals = {"train": 0, "validation": 0}
    resolved_sources: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise TypeError(f"Prepared source {source_index} must be an object")
        resolved_source = dict(source)
        for split in ("train", "validation"):
            path_field = f"{split}_jsonl"
            count_field = f"{split}_samples"
            jsonl_path = _resolve_reused_path(
                resolved_source.get(path_field), prepared_run, path_field
            )
            actual_count = _validate_prepared_jsonl(jsonl_path, allowed_prompts)
            expected_count = resolved_source.get(count_field)
            if not isinstance(expected_count, int) or expected_count <= 0:
                raise ValueError(
                    f"Prepared source {source_index} has invalid {count_field}"
                )
            if actual_count != expected_count:
                raise ValueError(
                    f"Prepared source {source_index} {count_field} is "
                    f"{expected_count}, but {jsonl_path} contains {actual_count}"
                )
            resolved_source[path_field] = str(jsonl_path)
            totals[split] += actual_count
        resolved_sources.append(resolved_source)

    for split, actual_count in totals.items():
        count_field = f"{split}_samples"
        if summary.get(count_field) != actual_count:
            raise ValueError(
                f"Prepared summary {count_field} is {summary.get(count_field)}, "
                f"but sources contain {actual_count}"
            )

    summary["sources"] = resolved_sources
    summary["prepared_from"] = str(prepared_run)
    return summary


def aggregate_prepared_runs(
    prepared_from: Sequence[Path],
    prepared_weights: Sequence[float] | None,
    work_dir: Path,
) -> dict[str, Any]:
    normalized_weights = normalize_prepared_weights(prepared_from, prepared_weights)
    run_summaries = [read_prepared_run(path) for path in prepared_from]
    models = [summary.get("model") for summary in run_summaries]
    if any(not isinstance(model, str) or not model for model in models):
        raise ValueError("Every prepared summary must record a non-empty model")
    if len(set(models)) != 1:
        raise ValueError(f"Prepared runs use different base models: {models}")

    sources: list[dict[str, Any]] = []
    train_probabilities: list[float] = []
    validation_probabilities: list[float] = []
    tasks: set[str] = set()
    source_runs: list[str] = []
    rejected: Counter[str] = Counter()
    for run_summary, run_weight in zip(
        run_summaries, normalized_weights, strict=True
    ):
        run_path = run_summary["prepared_from"]
        run_sources = run_summary["sources"]
        sources.extend(run_sources)
        source_runs.extend(run_path for _ in run_sources)
        train_probabilities.extend(
            run_weight * value for value in run_summary["train_probabilities"]
        )
        validation_probabilities.extend(
            run_weight * value for value in run_summary["validation_probabilities"]
        )
        tasks.update(run_summary["tasks"])
        run_rejected = run_summary.get("rejected", {})
        if isinstance(run_rejected, Mapping):
            rejected.update(
                {
                    str(reason): count
                    for reason, count in run_rejected.items()
                    if isinstance(count, int)
                }
            )

    for field, probabilities in (
        ("train_probabilities", train_probabilities),
        ("validation_probabilities", validation_probabilities),
    ):
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Aggregated {field} does not sum to 1.0")

    ordered_tasks = sorted(tasks)
    summary: dict[str, Any] = {
        "task": ordered_tasks[0] if len(ordered_tasks) == 1 else "mixed",
        "tasks": ordered_tasks,
        "prompts": [prompt_for_task(task) for task in ordered_tasks],
        "model": models[0],
        "sources": sources,
        "source_runs": source_runs,
        "train_samples": sum(summary["train_samples"] for summary in run_summaries),
        "validation_samples": sum(
            summary["validation_samples"] for summary in run_summaries
        ),
        "train_probabilities": train_probabilities,
        "validation_probabilities": validation_probabilities,
        "prepared_from_runs": [summary["prepared_from"] for summary in run_summaries],
        "prepared_weights": normalized_weights,
        "prepared_weight_policy": "relative_normalized",
        "rejected": dict(sorted(rejected.items())),
    }
    if len(ordered_tasks) == 1:
        summary["prompt"] = prompt_for_task(ordered_tasks[0])

    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def load_prepared_runs(
    prepared_from: Path | Sequence[Path],
    prepared_weights: Sequence[float] | None,
    work_dir: Path,
) -> dict[str, Any]:
    prepared_runs = (
        [prepared_from] if isinstance(prepared_from, Path) else list(prepared_from)
    )
    weights = normalize_prepared_weights(prepared_runs, prepared_weights)
    if len(prepared_runs) == 1:
        summary = read_prepared_run(prepared_runs[0])
        summary["prepared_weights"] = weights
        summary["prepared_weight_policy"] = "relative_normalized"
        summary["source_runs"] = [summary["prepared_from"]] * len(summary["sources"])
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary
    return aggregate_prepared_runs(prepared_runs, prepared_weights, work_dir)


def load_prepared_run(prepared_from: Path, work_dir: Path) -> dict[str, Any]:
    return load_prepared_runs(prepared_from, [1.0], work_dir)


def _csv(values: Iterable[Any]) -> str:
    return ",".join(str(value) for value in values)


def create_resolved_config(
    target_path: Path,
    work_dir: Path,
    summary: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sources = summary["sources"]
    output_dir = work_dir / "adapter"
    batch_size = 1
    packing_size = 1
    num_samples_each_epoch = summary.get("train_samples", 0) or sum(
        source.get("train_samples", 1) for source in sources
    )
    data_parallel_size = len(selected_devices(getattr(args, "devices", "0")))
    effective_batch_size = (
        batch_size
        * packing_size
        * args.gradient_accumulation_steps
        * data_parallel_size
    )
    max_steps = math.ceil(num_samples_each_epoch * args.epochs / effective_batch_size)
    config: dict[str, Any] = {
        "train_dataset_type": _csv("erniekit" for _ in sources),
        "eval_dataset_type": _csv("erniekit" for _ in sources),
        "train_dataset_path": _csv(source["train_jsonl"] for source in sources),
        "train_dataset_prob": _csv(summary["train_probabilities"]),
        "eval_dataset_path": _csv(source["validation_jsonl"] for source in sources),
        "eval_dataset_prob": _csv(summary["validation_probabilities"]),
        "max_seq_len": args.max_seq_len,
        "num_samples_each_epoch": num_samples_each_epoch,
        "use_pic_id": False,
        "sft_replace_ids": True,
        "sft_image_normalize": True,
        "sft_image_rescale": True,
        "image_dtype": "float32",
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "model_name_or_path": args.model,
        "fine_tuning": "LoRA",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_rank * 2,
        "multimodal": True,
        "use_flash_attention": args.flash_attention,
        "use_sparse_flash_attn": False,
        "stage": "OCR-VL-SFT",
        "seed": args.seed,
        "do_train": True,
        # ERNIEKit release/v1.5 leaves eval_dataset=None in OCR-VL-SFT.
        # Validation JSONL is retained for deterministic OCR evaluation instead.
        "do_eval": False,
        "distributed_dataloader": False,
        "dataloader_num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "batch_size": batch_size,
        "packing_size": packing_size,
        "packing": True,
        "padding": False,
        "num_train_epochs": args.epochs,
        # OCR-VL uses an IterableDataset, so ERNIEKit requires a positive
        # max_steps instead of deriving it from len(train_dataloader).
        "max_steps": max_steps,
        "save_strategy": "steps",
        "save_steps": getattr(args, "save_steps", 100),
        "save_total_limit": 3,
        "logging_steps": 1,
        "release_grads": True,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_dir": str((work_dir / "tensorboard_logs").resolve()),
        "output_dir": str(output_dir.resolve()),
        "disable_tqdm": False,
        "overwrite_output_dir": False,
        "warmup_ratio": 0.03,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "min_lr": args.learning_rate * 0.1,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-8,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "tensor_parallel_degree": 1,
        "pipeline_parallel_degree": 1,
        "sharding_parallel_degree": 1,
        "sharding": "stage1",
        "sequence_parallel": False,
        "recompute": True,
        "recompute_granularity": "full",
        "recompute_use_reentrant": True,
        "compute_type": "bf16",
        "bf16": True,
        "fp16_opt_level": "O2",
        "disable_ckpt_quant": True,
        "unified_checkpoint": True,
        # The official PaddleOCR-VL-1.6 snapshot stores Hugging Face-shaped
        # weights. ERNIEKit also uses this flag to select q/k/v/up/gate LoRA
        # targets instead of the fused ERNIE projection names.
        "use_huggingface_model": True,
        "convert_from_hf": True,
        "save_to_hf": True,
        "freeze_config": "freeze_vision",
        "pre_alloc_memory": 0,
        "from_scratch": 0,
    }
    if args.smoke_steps is not None:
        config["max_steps"] = args.smoke_steps
    if args.resume_from is not None:
        config["resume_from_checkpoint"] = str(args.resume_from.resolve())
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return config


def runtime_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.smoke_steps is not None:
        overrides["max_steps"] = args.smoke_steps
    if args.resume_from is not None:
        checkpoint = args.resume_from.expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        overrides["resume_from_checkpoint"] = str(checkpoint)
    return overrides


def create_resume_config(
    resolved_config: Path, work_dir: Path, args: argparse.Namespace
) -> Path:
    config = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    config.update(runtime_overrides(args))
    config["overwrite_output_dir"] = False
    candidate = work_dir / "resolved-resume.yaml"
    suffix = 1
    while candidate.exists():
        candidate = work_dir / f"resolved-resume-{suffix}.yaml"
        suffix += 1
    candidate.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return candidate


def validate_erniekit_source(erniekit_dir: Path) -> None:
    root = erniekit_dir.expanduser().resolve()
    workflow = root / "erniekit" / "train" / "ocr_vl_sft" / "workflow.py"
    peft = root / "ernie" / "utils" / "peft_utils.py"
    if not workflow.is_file() or not peft.is_file():
        raise FileNotFoundError(
            f"{root} is not a compatible ERNIEKit checkout with OCR-VL-SFT"
        )
    workflow_text = workflow.read_text(encoding="utf-8")
    peft_text = peft.read_text(encoding="utf-8")
    required = {
        "freeze_vision": workflow_text,
        "initialize_lora_model": workflow_text,
        "mark_only_lora_as_trainable": peft_text,
    }
    missing = [marker for marker, text in required.items() if marker not in text]
    if missing:
        raise RuntimeError(f"Incompatible ERNIEKit checkout; missing: {missing}")


def validate_dependency_versions(installed: Mapping[str, str]) -> None:
    mismatches = [
        f"{name}={installed.get(name, '<missing>')} (expected {expected})"
        for name, expected in ERNIEKIT_RUNTIME_VERSIONS.items()
        if installed.get(name) != expected
    ]
    if mismatches:
        raise RuntimeError("Incompatible ERNIEKit runtime: " + ", ".join(mismatches))


def validate_erniekit_runtime(erniekit_dir: Path) -> None:
    root = erniekit_dir.expanduser().resolve()
    if (root / ".git").exists():
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != ERNIEKIT_REVISION:
            raise RuntimeError(
                f"ERNIEKit revision {revision} does not match pinned {ERNIEKIT_REVISION}"
            )
    python = Path(_erniekit_python(root))
    if not (python.parent.parent / "pyvenv.cfg").is_file():
        return
    package_names = list(ERNIEKIT_RUNTIME_VERSIONS)
    code = (
        "import json; from importlib.metadata import version; "
        f"names={package_names!r}; "
        "print(json.dumps({name: version(name) for name in names}))"
    )
    result = subprocess.run(
        [str(python), "-c", code],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    installed = json.loads(result.stdout.strip().splitlines()[-1])
    validate_dependency_versions(installed)


def parse_trainable_parameter_output(output: str) -> TrainableParameterReport:
    match = TRAINABLE_LINE.search(output)
    if not match:
        raise RuntimeError("ERNIEKit did not emit a trainable-parameter report")
    trainable = int(Decimal(match.group(1).replace(",", "")))
    total = int(Decimal(match.group(2).replace(",", "")))
    names_match = TRAINABLE_NAMES_LINE.search(output)
    if not names_match:
        raise RuntimeError("ERNIEKit did not emit exact trainable parameter names")
    try:
        parsed_names = json.loads(names_match.group(1))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "ERNIEKit emitted invalid trainable parameter names"
        ) from exc
    if not isinstance(parsed_names, list) or not all(
        isinstance(name, str) and name for name in parsed_names
    ):
        raise RuntimeError("ERNIEKit emitted invalid trainable parameter names")
    return TrainableParameterReport(trainable, total, tuple(parsed_names))


def resolve_run_model(config_path: Path, requested_model: str) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or not isinstance(
        config.get("model_name_or_path"), str
    ):
        raise TypeError(f"Resolved config has no base model: {config_path}")
    resolved = Path(config["model_name_or_path"]).expanduser().resolve()
    requested = Path(requested_model).expanduser().resolve()
    if resolved != requested:
        raise ValueError(
            f"Requested base model {requested} does not match run base model {resolved}"
        )
    return resolved


def validate_work_dir_isolation(work_dir: Path, model: Path) -> None:
    resolved_work_dir = work_dir.expanduser().resolve()
    resolved_model = model.expanduser().resolve()
    if resolved_work_dir == resolved_model or resolved_model in resolved_work_dir.parents:
        raise ValueError("--work-dir must be outside the immutable base model snapshot")


def validate_adapter_base_model(adapter_dir: Path, model: Path) -> None:
    config_path = adapter_dir / "lora_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"LoRA config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured = config.get("base_model_name_or_path")
    if not isinstance(configured, str):
        raise TypeError("LoRA config has no base model")
    if Path(configured).expanduser().resolve() != model.expanduser().resolve():
        raise ValueError("LoRA adapter base model does not match run base model")


def validate_trainable_parameters(report: TrainableParameterReport) -> None:
    if report.total <= 0 or report.trainable <= 0:
        raise RuntimeError("Trainable parameter counts must be positive")
    if not any("lora" in name.lower() for name in report.names):
        raise RuntimeError("No LoRA trainable parameters were found")
    allowed = re.compile(
        r"^(?:model\.)?model\.layers\.\d+\."
        r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
        r"mlp\.(?:up_proj|gate_proj|down_proj))\.lora_[AB]\Z"
    )
    invalid = [name for name in report.names if not allowed.fullmatch(name)]
    if invalid:
        raise RuntimeError(
            "Trainable parameters outside text-decoder LoRA scope: "
            + ", ".join(invalid[:5])
        )
    if report.trainable >= report.total or report.fraction > 0.20:
        raise RuntimeError(
            "Too much of the base model is trainable; refusing to start full SFT"
        )


def validate_lora_parameter_names(names: Sequence[str]) -> None:
    if not names:
        raise RuntimeError("LoRA adapter contains no parameters")
    vision = [
        name for name in names if name.startswith("visual.") or ".vision_model." in name
    ]
    if vision:
        raise RuntimeError(
            "LoRA adapter contains vision-encoder parameters; refusing export"
        )
    non_decoder = [name for name in names if not name.startswith("model.layers.")]
    if non_decoder:
        raise RuntimeError(
            "LoRA adapter contains parameters outside the text decoder; refusing export"
        )
    required = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    }
    missing = sorted(
        projection
        for projection in required
        if not any(f".{projection}.lora_" in name for name in names)
    )
    if missing:
        raise RuntimeError(
            f"LoRA adapter is missing decoder projections: {', '.join(missing)}"
        )


def validate_adapter_scope(adapter_dir: Path) -> int:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to validate the LoRA adapter"
        ) from exc

    files = sorted(adapter_dir.glob("peft_model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"LoRA safetensors not found in: {adapter_dir}")
    names: list[str] = []
    for path in files:
        with safe_open(path, framework="numpy") as handle:
            names.extend(handle.keys())
    validate_lora_parameter_names(names)
    return len(names)


def _erniekit_python(erniekit_dir: Path) -> str:
    candidates = (
        erniekit_dir / ".venv" / "bin" / "python",
        erniekit_dir / "venv" / "bin" / "python",
    )
    return str(
        next((path for path in candidates if path.is_file()), Path(sys.executable))
    )


def build_erniekit_command(
    erniekit_dir: Path,
    action: str,
    config_path: Path,
    overrides: Mapping[str, Any] | None = None,
) -> list[str]:
    command = [
        _erniekit_python(erniekit_dir),
        "-m",
        "erniekit.cli",
        action,
        str(config_path.resolve()),
    ]
    for key, value in (overrides or {}).items():
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        command.append(f"{key}={rendered}")
    return command


def _query_vram_mib() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return max(
            int(line.strip()) for line in result.stdout.splitlines() if line.strip()
        )
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError):
        return None


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_logged_command(
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    *,
    capture_output: bool = True,
    erniekit_compat: bool = False,
    env_overrides: Mapping[str, str] | None = None,
) -> tuple[str, int | None]:
    env = os.environ.copy()
    env.update(env_overrides or {})
    executable_dir = Path(command[0]).expanduser().parent
    if executable_dir != Path("."):
        env["PATH"] = os.pathsep.join(
            [str(executable_dir), env.get("PATH", "")]
        ).rstrip(os.pathsep)
    python_paths = [str(cwd.resolve())]
    if erniekit_compat:
        env["PADDLEOCR_VL_TEXT_ONLY_LORA"] = "1"
        python_paths.insert(0, str(ERNIEKIT_COMPAT_DIR))
    python_paths.append(env.get("PYTHONPATH", ""))
    env["PYTHONPATH"] = os.pathsep.join(python_paths).rstrip(os.pathsep)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    peak_vram = _query_vram_mib()
    stop = threading.Event()

    def monitor() -> None:
        nonlocal peak_vram
        while not stop.wait(1.0):
            current = _query_vram_mib()
            if current is not None:
                peak_vram = current if peak_vram is None else max(peak_vram, current)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    output_parts: list[str] = []
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
                if capture_output:
                    output_parts.append(line)
        return_code = process.wait()
    except BaseException:
        terminate_process_group(process)
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        if process.stdout is not None:
            process.stdout.close()
    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code, command, output="".join(output_parts)
        )
    return "".join(output_parts), peak_vram


def inspect_model(
    erniekit_dir: Path, config_path: Path, work_dir: Path, devices: str = "0"
) -> TrainableParameterReport:
    validate_erniekit_source(erniekit_dir)
    validate_erniekit_runtime(erniekit_dir)
    command = build_erniekit_command(
        erniekit_dir,
        "train",
        config_path,
        {"do_train": False, "do_eval": False, "max_steps": 0},
    )
    try:
        output, peak_vram = run_logged_command(
            command,
            erniekit_dir.resolve(),
            work_dir / "logs" / "inspect-model.log",
            erniekit_compat=True,
            env_overrides={"CUDA_VISIBLE_DEVICES": devices},
        )
    except subprocess.CalledProcessError as exc:
        output = exc.output or ""
        known_dry_run_stop = (
            "AttributeError: 'FinetuningArguments' object has no attribute "
            "'is_train_mm'"
        )
        if known_dry_run_stop not in output:
            raise
        peak_vram = _query_vram_mib()
        LOGGER.warning(
            "ERNIEKit release/v1.5 stopped after LoRA inspection at its known "
            "do_train=false is_train_mm bug"
        )
    if LORA_SCOPE_MARKER not in output:
        raise RuntimeError(
            "ERNIEKit LoRA scope patch did not load; refusing to risk training "
            "vision-encoder adapters"
        )
    report = parse_trainable_parameter_output(output)
    validate_trainable_parameters(report)
    metrics_dir = work_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trainable": report.trainable,
        "total": report.total,
        "fraction": report.fraction,
        "peak_vram_mib": peak_vram,
    }
    (metrics_dir / "trainable_parameters.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return report


def create_export_config(work_dir: Path, model: str) -> Path:
    config = {
        "model_name_or_path": model,
        "fine_tuning": "LoRA",
        "lora": True,
        "copy_tokenizer": True,
        "output_dir": str((work_dir / "adapter").resolve()),
        "max_shard_size": 5,
        "tensor_parallel_degree": 1,
        "pipeline_parallel_degree": 1,
        "sharding_parallel_degree": 1,
        "sharding": "stage1",
        "sequence_parallel": False,
        "compute_type": "bf16",
        "fp16_opt_level": "O2",
    }
    path = work_dir / "export.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def build_export_command(
    erniekit_dir: Path,
    model: str,
    work_dir: Path,
    *,
    fixture_jsonl: Path,
    min_pixels: int,
    max_pixels: int,
    adapter_dir: Path | None = None,
    output_dir: Path | None = None,
) -> list[str]:
    """Build the PaddleOCR-VL-aware merge command."""
    entrypoint = Path(__file__).with_name("merge_paddleocr_vl_lora.py").resolve()
    selected_adapter = (adapter_dir or (work_dir / "adapter")).resolve()
    export_dir = (output_dir or (work_dir / "adapter" / "export")).resolve()
    return [
        _erniekit_python(erniekit_dir),
        str(entrypoint),
        "--base-model",
        model,
        "--adapter-dir",
        str(selected_adapter),
        "--output-dir",
        str(export_dir),
        "--fixture-jsonl",
        str(fixture_jsonl.resolve()),
        "--min-pixels",
        str(min_pixels),
        "--max-pixels",
        str(max_pixels),
    ]


def build_evaluation_command(
    evaluation_venv: Path,
    model: str,
    work_dir: Path,
    *,
    merged_model: Path,
    validation_jsonls: Sequence[Path],
    samples_per_dataset: int,
    max_new_tokens: int,
    task_max_new_tokens: Sequence[str] = (),
    min_normalized_edit_distance: float = 0.5,
    max_cer: float = 1.0,
    base_predictions_jsonl: Path | None = None,
    report_only: bool = False,
    output_dir: Path | None = None,
) -> list[str]:
    entrypoint = Path(__file__).with_name("evaluate_paddleocr_vl.py").resolve()
    metrics_dir = (output_dir or (work_dir / "metrics")).resolve()
    command = [
        str((evaluation_venv / "bin" / "python").resolve()),
        str(entrypoint),
        "--base-model",
        model,
        "--merged-model",
        str(merged_model.resolve()),
        "--validation-jsonl",
        *(str(path.resolve()) for path in validation_jsonls),
        "--output-dir",
        str(metrics_dir),
        "--samples-per-dataset",
        str(samples_per_dataset),
        "--max-new-tokens",
        str(max_new_tokens),
        "--min-normalized-edit-distance",
        str(min_normalized_edit_distance),
        "--max-cer",
        str(max_cer),
    ]
    for value in task_max_new_tokens:
        command.extend(("--task-max-new-tokens", value))
    if base_predictions_jsonl is not None:
        command.extend(
            ("--base-predictions-jsonl", str(base_predictions_jsonl.resolve()))
        )
    if report_only:
        command.append("--report-only")
    return command


def copy_inference_assets(model: str, export_dir: Path) -> list[str]:
    model_path = Path(model).expanduser()
    if not model_path.is_dir():
        return []
    copied: list[str] = []
    candidates = (
        "config.json",
        "tokenizer.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "inference.yml",
        "generation.json",
        "generation_config.json",
        "configuration_paddleocr_vl.py",
        "modeling_paddleocr_vl.py",
        "image_processing_paddleocr_vl.py",
        "processing_paddleocr_vl.py",
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    for name in candidates:
        source = model_path / name
        target = export_dir / name
        if source.is_file() and (name == "config.json" or not target.exists()):
            shutil.copy2(source, target)
            copied.append(name)
    return copied


def promote_export_directory(candidate: Path, destination: Path) -> None:
    """Replace an export only after its candidate has been fully verified."""
    candidate = candidate.resolve()
    destination = destination.expanduser().absolute()
    if candidate.parent != destination.parent:
        raise ValueError("Export candidate and destination must share a parent")
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError("Existing export destination must be a real directory")
    backup: Path | None = None
    if destination.exists():
        backup = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}-previous-", dir=destination.parent)
        )
        backup.rmdir()
        os.replace(destination, backup)
    try:
        os.replace(candidate, destination)
    except BaseException:
        if backup is not None:
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _metric_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    edits = sum(_edit_distance(row["target"], row["prediction"]) for row in rows)
    characters = sum(len(row["target"]) for row in rows)
    exact = sum(row["target"] == row["prediction"] for row in rows)
    normalized = sum(
        1.0
        - _edit_distance(row["target"], row["prediction"])
        / max(len(row["target"]), len(row["prediction"]), 1)
        for row in rows
    )
    return {
        "samples": len(rows),
        "cer": edits / characters if characters else 0.0,
        "exact_match": exact / len(rows) if rows else 0.0,
        "normalized_edit_distance": normalized / len(rows) if rows else 0.0,
    }


def compute_ocr_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dataset_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    task_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset_groups[row["dataset"]].append(row)
        task = row.get("task")
        if isinstance(task, str):
            task_groups[task].append(row)
    return {
        "overall": _metric_group(rows),
        "datasets": {
            name: _metric_group(group)
            for name, group in sorted(dataset_groups.items())
        },
        "tasks": {
            name: _metric_group(group) for name, group in sorted(task_groups.items())
        },
    }


def select_best_checkpoint(
    reports: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not reports:
        raise ValueError("No checkpoint metric reports were provided")
    return min(
        reports,
        key=lambda report: (
            float(report["cer"]),
            -float(report["exact_match"]),
            -float(report.get("normalized_edit_distance", 0.0)),
            str(report["checkpoint"]),
        ),
    )


def resolve_selected_adapter(work_dir: Path, selection: Mapping[str, Any]) -> Path:
    name = str(selection.get("checkpoint", ""))
    if name == "adapter":
        selected = work_dir / "adapter"
    elif re.fullmatch(r"checkpoint-\d+", name):
        selected = work_dir / "adapter" / name
    else:
        raise ValueError(f"Invalid selected checkpoint: {name!r}")
    selected = selected.resolve()
    adapter_root = (work_dir / "adapter").resolve()
    if selected != adapter_root and adapter_root not in selected.parents:
        raise ValueError("Selected checkpoint escapes adapter directory")
    if not (selected / "lora_config.json").is_file():
        raise FileNotFoundError(f"Selected checkpoint is incomplete: {selected}")
    return selected


def adapter_candidates(work_dir: Path, max_checkpoints: int) -> list[Path]:
    adapter_root = (work_dir / "adapter").resolve()
    checkpoints = sorted(
        (
            path
            for path in adapter_root.glob("checkpoint-*")
            if path.is_dir()
            and re.fullmatch(r"checkpoint-\d+", path.name)
            and (path / "lora_config.json").is_file()
        ),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
        reverse=True,
    )[:max_checkpoints]
    return [adapter_root, *checkpoints]


def evaluate_adapter_candidates(
    args: argparse.Namespace,
    work_dir: Path,
    run_model: Path,
    validation_jsonls: Sequence[Path],
    fixture_jsonl: Path,
    evaluation_samples: int,
) -> tuple[Path, dict[str, Any]]:
    evaluation_venv = Path(__file__).with_name(".venv-vl-eval")
    scratch_root = work_dir / ".checkpoint-evaluation"
    metrics_root = work_dir / "metrics" / "checkpoints"
    reports: list[dict[str, Any]] = []
    base_predictions_jsonl: Path | None = None
    try:
        for adapter_dir in adapter_candidates(work_dir, args.eval_max_checkpoints):
            validate_adapter_scope(adapter_dir)
            validate_adapter_base_model(adapter_dir, run_model)
            checkpoint = (
                "adapter"
                if adapter_dir == (work_dir / "adapter").resolve()
                else adapter_dir.name
            )
            candidate_export = scratch_root / checkpoint
            candidate_metrics = metrics_root / checkpoint
            shutil.rmtree(candidate_export, ignore_errors=True)
            shutil.rmtree(candidate_metrics, ignore_errors=True)
            run_logged_command(
                build_export_command(
                    args.erniekit_dir,
                    str(run_model),
                    work_dir,
                    fixture_jsonl=fixture_jsonl,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                    adapter_dir=adapter_dir,
                    output_dir=candidate_export,
                ),
                args.erniekit_dir.resolve(),
                work_dir / "logs" / f"checkpoint-export-{checkpoint}.log",
                capture_output=False,
                env_overrides={"CUDA_VISIBLE_DEVICES": args.devices},
            )
            copy_inference_assets(str(run_model), candidate_export)
            run_logged_command(
                build_evaluation_command(
                    evaluation_venv,
                    str(run_model),
                    work_dir,
                    merged_model=candidate_export,
                    validation_jsonls=validation_jsonls,
                    samples_per_dataset=evaluation_samples,
                    max_new_tokens=args.eval_max_new_tokens,
                    task_max_new_tokens=args.eval_task_max_new_tokens,
                    min_normalized_edit_distance=args.min_normalized_edit_distance,
                    max_cer=args.max_cer,
                    base_predictions_jsonl=base_predictions_jsonl,
                    report_only=True,
                    output_dir=candidate_metrics,
                ),
                Path(__file__).resolve().parent,
                work_dir / "logs" / f"checkpoint-evaluation-{checkpoint}.log",
                capture_output=False,
                env_overrides={"CUDA_VISIBLE_DEVICES": args.devices},
            )
            report = json.loads(
                (candidate_metrics / "ocr_metrics.json").read_text(encoding="utf-8")
            )
            if base_predictions_jsonl is None:
                base_predictions_jsonl = candidate_metrics / "ocr_predictions.jsonl"
                if not base_predictions_jsonl.is_file():
                    raise FileNotFoundError(
                        f"Candidate evaluation omitted predictions: {base_predictions_jsonl}"
                    )
            overall = report["candidates"]["merged"]["overall"]
            reports.append(
                {
                    "checkpoint": checkpoint,
                    "status": report.get("status"),
                    "failures": report.get("failures", []),
                    **overall,
                }
            )
            shutil.rmtree(candidate_export, ignore_errors=True)
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    passing = [report for report in reports if report["status"] == "passed"]
    eligible = passing or (reports if args.smoke_steps is not None else [])
    if not eligible:
        selection_report = {"status": "failed", "candidates": reports}
        metrics_root.mkdir(parents=True, exist_ok=True)
        (work_dir / "metrics" / "checkpoint_selection.json").write_text(
            json.dumps(selection_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("No adapter checkpoint passed the native OCR quality gate")
    selected = dict(select_best_checkpoint(eligible))
    selection_report = {
        "status": "passed" if selected["status"] == "passed" else "failed",
        "selected": selected,
        "candidates": reports,
        "base_predictions_jsonl": str(base_predictions_jsonl.resolve()),
    }
    (work_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (work_dir / "metrics" / "checkpoint_selection.json").write_text(
        json.dumps(selection_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolve_selected_adapter(work_dir, selected), selection_report


def export_verification_status(
    skip_evaluation: bool, merge_consistency: Mapping[str, Any] | None
) -> str:
    if skip_evaluation:
        return "unverified"
    if merge_consistency and merge_consistency.get("status") == "passed":
        return "passed"
    return "failed"


def _load_existing_run(work_dir: Path) -> tuple[dict[str, Any], Path]:
    summary_path = work_dir / "summary.json"
    config_path = work_dir / "resolved.yaml"
    if not summary_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            "Resume run is missing summary.json or resolved.yaml; refusing to overwrite it"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise TypeError(f"Run summary must be a JSON object: {summary_path}")
    _normalize_summary_tasks(summary)
    return summary, config_path


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)
    validate_args(args)
    work_dir = make_work_dir(args.work_dir, args.resume_from)
    if args.resume_from is not None:
        summary, config_path = _load_existing_run(work_dir)
        config_path = create_resume_config(config_path, work_dir, args)
    elif args.prepared_from is not None:
        prepared_runs = (
            [args.prepared_from]
            if isinstance(args.prepared_from, Path)
            else args.prepared_from
        )
        summary = load_prepared_runs(
            prepared_runs,
            args.prepared_weight,
            work_dir,
        )
        config_path = work_dir / "resolved.yaml"
        create_resolved_config(config_path, work_dir, summary, args)
    else:
        tokenizer = load_tokenizer(args.model)
        summary = prepare_datasets(args, work_dir, tokenizer)
        config_path = work_dir / "resolved.yaml"
        create_resolved_config(config_path, work_dir, summary, args)

    LOGGER.info(
        "Prepared %d train and %d validation samples",
        summary["train_samples"],
        summary["validation_samples"],
    )
    if args.prepare_only:
        return 0

    assert args.erniekit_dir is not None
    run_model = resolve_run_model(config_path, args.model)
    validate_work_dir_isolation(work_dir, run_model)
    inspect_model(args.erniekit_dir, config_path, work_dir, args.devices)
    if args.inspect_model:
        return 0

    train_command = build_erniekit_command(
        args.erniekit_dir, "train", config_path, runtime_overrides(args)
    )
    _, peak_vram = run_logged_command(
        train_command,
        args.erniekit_dir.resolve(),
        work_dir / "logs" / "train.log",
        capture_output=False,
        erniekit_compat=True,
        env_overrides={"CUDA_VISIBLE_DEVICES": args.devices},
    )
    metrics_path = work_dir / "metrics" / "runtime.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({"peak_vram_mib": peak_vram}, indent=2) + "\n", encoding="utf-8"
    )
    validate_adapter_scope(work_dir / "adapter")
    validate_adapter_base_model(work_dir / "adapter", run_model)

    validation_jsonls = [
        Path(source["validation_jsonl"]) for source in summary["sources"]
    ]
    evaluation_samples = (
        1 if args.smoke_steps is not None else args.eval_samples_per_dataset
    )
    create_export_config(work_dir, str(run_model))
    fixture_jsonl = Path(summary["sources"][0]["validation_jsonl"])
    checkpoint_selection = None
    if args.skip_evaluation:
        selected_adapter = (work_dir / "adapter").resolve()
    else:
        selected_adapter, checkpoint_selection = evaluate_adapter_candidates(
            args,
            work_dir,
            run_model,
            validation_jsonls,
            fixture_jsonl,
            evaluation_samples,
        )
    export_dir = work_dir / "adapter" / "export"
    candidate_export = Path(
        tempfile.mkdtemp(prefix=".export-build-", dir=export_dir.parent)
    )
    try:
        export_command = build_export_command(
            args.erniekit_dir,
            str(run_model),
            work_dir,
            fixture_jsonl=fixture_jsonl,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            adapter_dir=selected_adapter,
            output_dir=candidate_export,
        )
        run_logged_command(
            export_command,
            args.erniekit_dir.resolve(),
            work_dir / "logs" / "export.log",
            capture_output=False,
            env_overrides={"CUDA_VISIBLE_DEVICES": args.devices},
        )
        copied = copy_inference_assets(str(run_model), candidate_export)
        weight_verification = json.loads(
            (candidate_export / "merge_verification.json").read_text(encoding="utf-8")
        )
        logits_verification = json.loads(
            (candidate_export / "logits_verification.json").read_text(encoding="utf-8")
        )
        if weight_verification.get("status") != "passed":
            raise RuntimeError("Merged model verification did not pass")
        if logits_verification.get("status") != "passed":
            raise RuntimeError("Merged model logits verification did not pass")

        evaluation_report = None
        if not args.skip_evaluation:
            evaluation_venv = Path(__file__).with_name(".venv-vl-eval")
            base_predictions_jsonl = Path(
                checkpoint_selection["base_predictions_jsonl"]
            )
            evaluation_command = build_evaluation_command(
                evaluation_venv,
                str(run_model),
                work_dir,
                merged_model=candidate_export,
                validation_jsonls=validation_jsonls,
                samples_per_dataset=evaluation_samples,
                max_new_tokens=args.eval_max_new_tokens,
                task_max_new_tokens=args.eval_task_max_new_tokens,
                min_normalized_edit_distance=args.min_normalized_edit_distance,
                max_cer=args.max_cer,
                base_predictions_jsonl=base_predictions_jsonl,
                report_only=args.smoke_steps is not None,
                output_dir=work_dir / "metrics",
            )
            run_logged_command(
                evaluation_command,
                Path(__file__).resolve().parent,
                work_dir / "logs" / "evaluation.log",
                capture_output=False,
                env_overrides={"CUDA_VISIBLE_DEVICES": args.devices},
            )
            evaluation_report = json.loads(
                (work_dir / "metrics" / "ocr_metrics.json").read_text(encoding="utf-8")
            )
        if (
            evaluation_report is not None
            and evaluation_report.get("status") != "passed"
            and args.smoke_steps is None
        ):
            raise RuntimeError("Native OCR evaluation did not pass")
        promote_export_directory(candidate_export, export_dir)
    finally:
        shutil.rmtree(candidate_export, ignore_errors=True)

    manifest = {
        "status": export_verification_status(args.skip_evaluation, evaluation_report),
        "adapter_dir": str((work_dir / "adapter").resolve()),
        "selected_adapter_dir": str(selected_adapter),
        "checkpoint_selection": checkpoint_selection,
        "merged_model_dir": str(export_dir.resolve()),
        "copied_assets": copied,
        "verification": {
            "weights": weight_verification,
            "logits": logits_verification,
            "native_evaluation": (
                None
                if evaluation_report is None
                else {
                    "status": evaluation_report.get("status"),
                    "fixture_count": evaluation_report.get("fixture_count"),
                }
            ),
        },
        "evaluation": evaluation_report,
    }
    (work_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

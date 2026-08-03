#!/usr/bin/env python3
"""Prepare Hugging Face OCR datasets and fine-tune PaddleOCR recognition."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import re
import subprocess
import sys
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from PIL import Image, UnidentifiedImageError


LOGGER = logging.getLogger("paddleocr_vi_finetune")
DEFAULT_CONFIG = "configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml"
DEFAULT_PRETRAINED = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
    "official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams"
)
PARQUET_SHARD_PATTERN = re.compile(
    r"^(?P<split>.+)-\d{5}-of-\d{5}(?:-[^.]+)?\.parquet$"
)


@dataclass(frozen=True)
class PreparedSample:
    image_path: str
    text: str
    dataset_index: int


class RejectionReport:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("w", encoding="utf-8")
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

    def __enter__(self) -> "RejectionReport":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mix Hugging Face datasets saved to disk or stored as local Parquet "
            "snapshots, filter invalid OCR samples, and fine-tune PP-OCRv6 recognition."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more dataset directories created by save_to_disk() or containing "
            "data/*.parquet files with image + label/text columns."
        ),
    )
    parser.add_argument(
        "--paddleocr-dir",
        type=Path,
        required=True,
        help="PaddleOCR source checkout containing tools/train.py.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="New run directory (default: runs/vi_rec_YYYYmmdd_HHMMSS).",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--pretrained-model", default=DEFAULT_PRETRAINED)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--max-text-length", type=int, default=80)
    parser.add_argument("--max-image-pixels", type=int, default=50_000_000)
    parser.add_argument(
        "--character-dict",
        type=Path,
        help="One-character-per-line dictionary (default: bundled Vietnamese dictionary).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare/filter data and config without downloading weights or training.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.validation_ratio < 0.5:
        raise ValueError("--validation-ratio must be between 0 and 0.5")
    for name in ("epochs", "batch_size", "num_workers", "image_width"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_text_length <= 0 or args.max_image_pixels <= 0:
        raise ValueError("text length and image pixel limits must be positive")
    missing = [str(path) for path in args.dataset_dir if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Dataset directories not found: {missing}")

    paddle_dir = args.paddleocr_dir.resolve()
    if not (paddle_dir / "tools" / "train.py").is_file():
        raise FileNotFoundError(f"Not a PaddleOCR source checkout: {paddle_dir}")
    config_path = resolve_config_path(paddle_dir, args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"PaddleOCR config not found: {config_path}")


def resolve_config_path(paddle_dir: Path, config: str) -> Path:
    path = Path(config).expanduser()
    return path.resolve() if path.is_absolute() else (paddle_dir / path).resolve()


def make_work_dir(path: Path | None) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("runs") / f"vi_rec_{stamp}"
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Work directory is not empty: {path}. Use a new directory to avoid overwriting a run."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_parquet_splits(path: Path) -> dict[str, list[str]]:
    candidates = sorted((path / "data").glob("*.parquet"))
    candidates.extend(sorted(path.glob("*.parquet")))
    splits: dict[str, list[str]] = {}
    for parquet_path in dict.fromkeys(candidate.resolve() for candidate in candidates):
        match = PARQUET_SHARD_PATTERN.match(parquet_path.name)
        split = match.group("split") if match else "train"
        splits.setdefault(split, []).append(str(parquet_path))
    return splits


def load_hf_dataset(path: Path) -> Any:
    try:
        from datasets import Image as HFImage
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Run: python -m pip install -r requirements.txt"
        ) from exc

    if (path / "state.json").is_file() or (path / "dataset_dict.json").is_file():
        dataset = load_from_disk(str(path))
    else:
        parquet_splits = discover_parquet_splits(path)
        if not parquet_splits:
            raise ValueError(
                f"Dataset directory {path} is neither a save_to_disk() directory "
                "nor a local Parquet snapshot containing data/*.parquet files"
            )
        dataset = load_dataset("parquet", data_files=parquet_splits)

    def disable_image_decode(split: Any) -> Any:
        columns = getattr(split, "column_names", [])
        if "image" not in columns:
            return split
        try:
            return split.cast_column("image", HFImage(decode=False))
        except Exception:
            return split

    if isinstance(dataset, Mapping):
        return {name: disable_image_decode(split) for name, split in dataset.items()}
    return disable_image_decode(dataset)


def select_splits(dataset: Any) -> tuple[Any, Any | None, str, str | None]:
    if not isinstance(dataset, Mapping):
        return dataset, None, "train", None

    names = list(dataset)
    if "train" in dataset:
        train_name = "train"
    elif len(names) == 1:
        train_name = names[0]
    else:
        raise ValueError(f"DatasetDict has no train split; available splits: {names}")

    validation_name = next(
        (name for name in ("validation", "valid", "dev") if name in dataset), None
    )
    validation = dataset[validation_name] if validation_name else None
    return dataset[train_name], validation, train_name, validation_name


def normalize_text(row: Mapping[str, Any]) -> str:
    value: str | None = None
    for column in ("label", "text"):
        candidate = row.get(column)
        if isinstance(candidate, str) and candidate.strip():
            value = candidate
            break
    if value is None:
        return ""
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    value = value.replace("\t", " ").strip()
    return value


def load_character_set(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig") as file:
        characters = {line.rstrip("\r\n") for line in file}
    characters.discard("")
    characters.add(" ")
    return characters


def unsupported_characters(text: str, characters: set[str]) -> list[str]:
    return sorted({character for character in text if character not in characters})


def checked_image_copy(image: Image.Image, max_image_pixels: int) -> Image.Image:
    width, height = image.size
    if width <= 1 or height <= 1:
        raise ValueError(f"invalid dimensions {width}x{height}")
    if width * height > max_image_pixels:
        raise ValueError(f"image has {width * height} pixels; limit is {max_image_pixels}")
    image.load()
    return image.copy()


def open_image(value: Any, dataset_dir: Path, max_image_pixels: int) -> Image.Image:
    if isinstance(value, Image.Image):
        return checked_image_copy(value, max_image_pixels)

    if isinstance(value, Mapping):
        raw_bytes = value.get("bytes")
        if raw_bytes is not None:
            with Image.open(io.BytesIO(raw_bytes)) as image:
                return checked_image_copy(image, max_image_pixels)
        value = value.get("path")

    if isinstance(value, (str, os.PathLike)):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = dataset_dir / path
        with Image.open(path) as image:
            return checked_image_copy(image, max_image_pixels)

    if hasattr(value, "__array_interface__"):
        image = Image.fromarray(value)
        width, height = image.size
        if width <= 1 or height <= 1:
            raise ValueError(f"invalid dimensions {width}x{height}")
        if width * height > max_image_pixels:
            raise ValueError(
                f"image has {width * height} pixels; limit is {max_image_pixels}"
            )
        return image

    raise TypeError(f"Unsupported image value type: {type(value).__name__}")


def save_lossless_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(temp_path, format="PNG", compress_level=3)
    os.replace(temp_path, path)


def process_split(
    split: Any,
    dataset_dir: Path,
    dataset_index: int,
    split_name: str,
    prepared_dir: Path,
    characters: set[str],
    max_text_length: int,
    max_image_pixels: int,
    report: RejectionReport,
) -> list[PreparedSample]:
    columns = set(getattr(split, "column_names", []))
    if columns and "image" not in columns:
        raise ValueError(f"{dataset_dir}/{split_name} has no 'image' column")
    if columns and not columns.intersection({"label", "text"}):
        raise ValueError(f"{dataset_dir}/{split_name} has no 'label' or 'text' column")

    samples: list[PreparedSample] = []
    total = len(split)
    for row_index in range(total):
        try:
            row = split[row_index]
        except Exception as exc:
            report.reject(dataset_dir, split_name, row_index, "row_load_error", str(exc))
            continue

        text = normalize_text(row)
        if not text:
            report.reject(dataset_dir, split_name, row_index, "empty_text")
            continue
        if any(unicodedata.category(char) == "Cc" for char in text):
            report.reject(dataset_dir, split_name, row_index, "control_character")
            continue
        if len(text) > max_text_length:
            report.reject(
                dataset_dir,
                split_name,
                row_index,
                "text_too_long",
                f"{len(text)} > {max_text_length}",
            )
            continue
        unknown = unsupported_characters(text, characters)
        if unknown:
            report.reject(
                dataset_dir,
                split_name,
                row_index,
                "unsupported_characters",
                "".join(unknown),
            )
            continue

        try:
            image = open_image(row.get("image"), dataset_dir, max_image_pixels)
            relative_path = Path("images") / f"dataset_{dataset_index:03d}" / (
                f"{split_name}_{row_index:09d}.png"
            )
            save_lossless_image(image, prepared_dir / relative_path)
        except (OSError, ValueError, TypeError, UnidentifiedImageError) as exc:
            report.reject(dataset_dir, split_name, row_index, "invalid_image", str(exc))
            continue
        finally:
            if "image" in locals():
                image.close()
                del image

        samples.append(PreparedSample(relative_path.as_posix(), text, dataset_index))
        if (row_index + 1) % 1000 == 0:
            LOGGER.info(
                "%s/%s: checked %d/%d rows",
                dataset_dir.name,
                split_name,
                row_index + 1,
                total,
            )
    return samples


def split_train_validation(
    samples: Sequence[PreparedSample], validation_ratio: float, seed: int
) -> tuple[list[PreparedSample], list[PreparedSample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, []
    validation_count = max(1, round(len(shuffled) * validation_ratio))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def write_label_file(path: Path, samples: Iterable[PreparedSample]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(f"{sample.image_path}\t{sample.text}\n")


def find_transform(transforms: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for transform in transforms:
        if name in transform:
            return transform[name]
    return None


def create_resolved_config(
    source_path: Path,
    target_path: Path,
    prepared_dir: Path,
    output_dir: Path,
    character_dict: Path,
    args: argparse.Namespace,
) -> None:
    with source_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    global_config = config["Global"]
    global_config.update(
        {
            "use_gpu": True,
            "epoch_num": args.epochs,
            "save_model_dir": str(output_dir),
            "save_epoch_step": 1,
            "eval_batch_step": [0, 1000],
            "cal_metric_during_train": True,
            "pretrained_model": args.pretrained_model,
            "character_dict_path": str(character_dict),
            "max_text_length": args.max_text_length,
            "use_space_char": True,
            "use_visualdl": True,
            "use_amp": True,
            "scale_loss": 512.0,
            "use_dynamic_loss_scaling": True,
            "seed": args.seed,
            "d2s_train_image_shape": [3, 48, args.image_width],
        }
    )
    config["Optimizer"]["lr"]["learning_rate"] = args.learning_rate
    config["Optimizer"]["lr"]["warmup_epoch"] = min(2, args.epochs)

    for head in config["Architecture"]["Head"]["head_list"]:
        if "NRTRHead" in head:
            head["NRTRHead"]["max_text_length"] = args.max_text_length

    train = config["Train"]
    train["dataset"]["data_dir"] = str(prepared_dir)
    train["dataset"]["label_file_list"] = [str(prepared_dir / "train.txt")]
    train["dataset"]["ratio_list"] = [1.0]
    train["sampler"]["scales"] = [
        [args.image_width, 32],
        [args.image_width, 48],
        [args.image_width, 64],
    ]
    train["sampler"]["first_bs"] = args.batch_size
    train["loader"]["batch_size_per_card"] = args.batch_size
    train["loader"]["num_workers"] = args.num_workers
    rec_con_aug = find_transform(train["dataset"]["transforms"], "RecConAug")
    if rec_con_aug is not None:
        rec_con_aug["image_shape"] = [48, args.image_width, 3]
        rec_con_aug["max_text_length"] = args.max_text_length

    evaluation = config["Eval"]
    evaluation["dataset"]["data_dir"] = str(prepared_dir)
    evaluation["dataset"]["label_file_list"] = [str(prepared_dir / "validation.txt")]
    evaluation["loader"]["batch_size_per_card"] = args.batch_size * 2
    evaluation["loader"]["num_workers"] = max(1, args.num_workers // 2)
    resize = find_transform(evaluation["dataset"]["transforms"], "RecResizeImg")
    if resize is not None:
        resize["image_shape"] = [3, 48, args.image_width]

    with target_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def download_pretrained(value: str, work_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in ("http", "https"):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained model not found: {path}")
        return path

    destination = work_dir / "pretrained" / Path(parsed.path).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    temp_path = destination.with_suffix(destination.suffix + ".part")
    LOGGER.info("Downloading pretrained weights to %s", destination)
    try:
        urllib.request.urlretrieve(value, temp_path)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return destination


def verify_paddle_gpu() -> None:
    try:
        import paddle
    except ImportError as exc:
        raise RuntimeError("PaddlePaddle is not installed in this Python environment") from exc
    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("Installed PaddlePaddle build has no CUDA support")
    if paddle.device.cuda.device_count() < 1:
        raise RuntimeError("PaddlePaddle cannot see a CUDA GPU")
    try:
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=512.0,
            use_dynamic_loss_scaling=True,
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
            f"{numpy_version}; install the pinned dependency with: "
            "python -m pip install 'numpy<2.4'"
        ) from exc


def prepare_datasets(
    args: argparse.Namespace,
    work_dir: Path,
    character_dict: Path,
) -> dict[str, Any]:
    prepared_dir = work_dir / "prepared"
    prepared_dir.mkdir()
    characters = load_character_set(character_dict)
    train_all: list[PreparedSample] = []
    validation_all: list[PreparedSample] = []
    dataset_summaries: list[dict[str, Any]] = []

    with RejectionReport(prepared_dir / "rejected.jsonl") as report:
        for dataset_index, raw_path in enumerate(args.dataset_dir):
            dataset_dir = raw_path.expanduser().resolve()
            LOGGER.info("Loading dataset %d: %s", dataset_index, dataset_dir)
            dataset = load_hf_dataset(dataset_dir)
            train_split, validation_split, train_name, validation_name = select_splits(dataset)
            train_samples = process_split(
                train_split,
                dataset_dir,
                dataset_index,
                train_name,
                prepared_dir,
                characters,
                args.max_text_length,
                args.max_image_pixels,
                report,
            )
            if validation_split is None:
                train_samples, validation_samples = split_train_validation(
                    train_samples, args.validation_ratio, args.seed + dataset_index
                )
            else:
                validation_samples = process_split(
                    validation_split,
                    dataset_dir,
                    dataset_index,
                    validation_name or "validation",
                    prepared_dir,
                    characters,
                    args.max_text_length,
                    args.max_image_pixels,
                    report,
                )

            train_all.extend(train_samples)
            validation_all.extend(validation_samples)
            dataset_summaries.append(
                {
                    "dataset": str(dataset_dir),
                    "train_samples": len(train_samples),
                    "validation_samples": len(validation_samples),
                }
            )

        rejection_counts = dict(sorted(report.counts.items()))

    if not train_all:
        raise RuntimeError("No valid training samples remain after filtering")
    if not validation_all:
        raise RuntimeError(
            "No validation samples remain. Add data or provide a validation/valid/dev split."
        )

    random.Random(args.seed).shuffle(train_all)
    random.Random(args.seed + 1).shuffle(validation_all)
    write_label_file(prepared_dir / "train.txt", train_all)
    write_label_file(prepared_dir / "validation.txt", validation_all)
    summary = {
        "train_samples": len(train_all),
        "validation_samples": len(validation_all),
        "rejected_samples": sum(rejection_counts.values()),
        "rejection_counts": rejection_counts,
        "datasets": dataset_summaries,
    }
    with (prepared_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    validate_args(args)
    if not args.prepare_only:
        verify_paddle_gpu()
    work_dir = make_work_dir(args.work_dir)
    paddle_dir = args.paddleocr_dir.expanduser().resolve()
    config_source = resolve_config_path(paddle_dir, args.config)
    character_dict = (
        args.character_dict.expanduser().resolve()
        if args.character_dict
        else Path(__file__).with_name("vietnamese_dict.txt").resolve()
    )
    if not character_dict.is_file():
        raise FileNotFoundError(f"Character dictionary not found: {character_dict}")

    summary = prepare_datasets(args, work_dir, character_dict)
    config_target = work_dir / "resolved_config.yml"
    create_resolved_config(
        config_source,
        config_target,
        work_dir / "prepared",
        work_dir / "output",
        character_dict.resolve(),
        args,
    )
    LOGGER.info("Prepared summary: %s", json.dumps(summary, ensure_ascii=False))
    LOGGER.info("Resolved config: %s", config_target)
    if args.prepare_only:
        return 0

    pretrained = download_pretrained(args.pretrained_model, work_dir)
    with config_target.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["Global"]["pretrained_model"] = str(pretrained)
    with config_target.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    command = [sys.executable, "tools/train.py", "-c", str(config_target)]
    LOGGER.info("Starting training: %s", " ".join(command))
    subprocess.run(command, cwd=paddle_dir, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(2)

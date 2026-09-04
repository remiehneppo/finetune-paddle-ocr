"""Shared image admission primitives for OCR, detection, and VL preparation."""

from __future__ import annotations

import io
import os
import json
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping
from collections import Counter
from typing import Any, Type

from PIL import Image


class RejectionReport:
    """Shared rejection accounting with workflow-specific record adapters."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("w", encoding="utf-8", newline="\n")
        self.counts: Counter[str] = Counter()

    def _write(self, reason: str, detail: str = "", **fields: Any) -> None:
        self.counts[reason] += 1
        record = {
            **fields,
            "reason": reason,
            "detail": detail[:500],
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def reject(
        self,
        dataset: Path,
        split: str,
        row_index: int,
        reason: str,
        detail: str = "",
    ) -> None:
        self._write(
            reason,
            detail,
            dataset=str(dataset),
            split=split,
            row_index=row_index,
        )

    def add(
        self,
        source: Any,
        line_number: int,
        reason: str,
        detail: str = "",
        box_index: int | None = None,
    ) -> None:
        fields: dict[str, Any] = {
            "dataset": str(source.root),
            "label_file": str(source.labels),
            "line_number": line_number,
        }
        if box_index is not None:
            fields["box_index"] = box_index
        self._write(reason, detail, **fields)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "RejectionReport":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def validate_image(
    image: Image.Image,
    max_image_pixels: int,
    *,
    min_dimension: int = 1,
    pixel_limit_error: Type[Exception] = ValueError,
) -> tuple[int, int]:
    width, height = image.size
    if width <= min_dimension or height <= min_dimension:
        raise ValueError(f"invalid dimensions {width}x{height}")
    pixels = width * height
    if pixels > max_image_pixels:
        raise pixel_limit_error(
            f"image has {pixels} pixels; limit is {max_image_pixels}"
        )
    image.load()
    return width, height


def checked_image_copy(
    image: Image.Image,
    max_image_pixels: int,
    *,
    convert_rgb: bool = False,
    pixel_limit_error: Type[Exception] = ValueError,
) -> Image.Image:
    validate_image(
        image,
        max_image_pixels,
        pixel_limit_error=pixel_limit_error,
    )
    return image.convert("RGB") if convert_rgb else image.copy()


def open_image_value(
    value: Any,
    dataset_dir: Path,
    max_image_pixels: int,
    *,
    convert_rgb: bool = False,
    pixel_limit_error: Type[Exception] = ValueError,
) -> Image.Image:
    if isinstance(value, Image.Image):
        return checked_image_copy(
            value,
            max_image_pixels,
            convert_rgb=convert_rgb,
            pixel_limit_error=pixel_limit_error,
        )

    if isinstance(value, Mapping):
        raw_bytes = value.get("bytes")
        if raw_bytes is not None:
            with Image.open(io.BytesIO(raw_bytes)) as image:
                return checked_image_copy(
                    image,
                    max_image_pixels,
                    convert_rgb=convert_rgb,
                    pixel_limit_error=pixel_limit_error,
                )
        value = value.get("path")

    if isinstance(value, (str, os.PathLike)):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = dataset_dir / path
        with Image.open(path) as image:
            return checked_image_copy(
                image,
                max_image_pixels,
                convert_rgb=convert_rgb,
                pixel_limit_error=pixel_limit_error,
            )

    if hasattr(value, "__array_interface__"):
        return checked_image_copy(
            Image.fromarray(value),
            max_image_pixels,
            convert_rgb=convert_rgb,
            pixel_limit_error=pixel_limit_error,
        )

    raise TypeError(f"Unsupported image value type: {type(value).__name__}")


def inspect_image_file(path: Path, max_image_pixels: int) -> tuple[int, int, bytes, str]:
    data = path.read_bytes()
    if not data:
        raise ValueError("empty image file")
    try:
        image = Image.open(path)
    except (OSError, Image.UnidentifiedImageError) as exc:
        raise ValueError(f"cannot decode image: {exc}") from exc
    with image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("image has invalid dimensions")
        validate_image(image, max_image_pixels, min_dimension=-1)
    return width, height, data, sha256(data).hexdigest()


__all__ = [
    "RejectionReport",
    "checked_image_copy",
    "inspect_image_file",
    "open_image_value",
    "validate_image",
]

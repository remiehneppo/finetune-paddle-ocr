"""Geometry helpers for VL layout annotations."""

from __future__ import annotations

import math
from typing import Sequence

from .models import Annotation, Block, Point


def clamp_polygon(polygon: list[Point], width: int, height: int) -> list[Point]:
    max_x = float(max(width - 1, 0))
    max_y = float(max(height - 1, 0))
    return [
        (min(max(float(x), 0.0), max_x), min(max(float(y), 0.0), max_y))
        for x, y in polygon
    ]


def polygon_to_xyxy(polygon: Sequence[Point]) -> tuple[float, float, float, float]:
    xs = [float(x) for x, _ in polygon]
    ys = [float(y) for _, y in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_area(polygon: Sequence[Point]) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            float(polygon[index][0]) * float(polygon[(index + 1) % len(polygon)][1])
            - float(polygon[(index + 1) % len(polygon)][0])
            * float(polygon[index][1])
            for index in range(len(polygon))
        )
    ) / 2.0


def polygon_to_xywh(polygon: Sequence[Point]) -> list[float]:
    x1, y1, x2, y2 = polygon_to_xyxy(polygon)
    return [x1, y1, x2 - x1, y2 - y1]


def clamp_xyxy(
    xyxy: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    left = int(math.floor(min(x1, x2)))
    right = int(math.ceil(max(x1, x2)))
    top = int(math.floor(min(y1, y2)))
    bottom = int(math.ceil(max(y1, y2)))
    left = max(0, min(left, max(width - 1, 0)))
    right = max(0, min(right, width))
    top = max(0, min(top, max(height - 1, 0)))
    bottom = max(0, min(bottom, height))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


def crop_box_from_polygon(
    polygon: Sequence[Point], width: int, height: int
) -> tuple[int, int, int, int]:
    return clamp_xyxy(polygon_to_xyxy(polygon), width, height)


def xyxy_to_polygon(x1: float, y1: float, x2: float, y2: float) -> list[Point]:
    return [
        (float(x1), float(y1)),
        (float(x2), float(y1)),
        (float(x2), float(y2)),
        (float(x1), float(y2)),
    ]


def aggregate_text(blocks: list[Block]) -> str:
    parts = []
    for block in sorted(blocks, key=lambda item: item.order):
        if block.skipped or block.task is None:
            continue
        text = block.text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def normalize_annotation(annotation: Annotation) -> Annotation:
    ordered = sorted(annotation.blocks, key=lambda item: item.order)
    blocks = [
        block.model_copy(
            update={
                "order": order,
                "polygon": clamp_polygon(
                    block.polygon, annotation.image.width, annotation.image.height
                ),
            }
        )
        for order, block in enumerate(ordered)
    ]
    return annotation.model_copy(
        update={"blocks": blocks, "text": aggregate_text(blocks)}
    )

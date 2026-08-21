from __future__ import annotations

import math
from .catalog import ImageRecord
from .geometry import normalize_annotation, xyxy_to_polygon
from .models import Annotation, Block, ImageInfo
from .task_map import PP_DOCLAYOUTV3_LABEL_SET, map_layout_label


def _polygon(value) -> list[tuple[float, float]] | None:
    try:
        values = list(value)
        if len(values) == 4:
            try:
                polygon = [(float(point[0]), float(point[1])) for point in values]
            except (IndexError, TypeError, ValueError):
                polygon = xyxy_to_polygon(*[float(item) for item in values])
        elif len(values) == 8:
            polygon = [(float(values[i]), float(values[i + 1])) for i in range(0, 8, 2)]
        else:
            return None
    except (TypeError, ValueError):
        return None
    return polygon if all(math.isfinite(x) and math.isfinite(y) for x, y in polygon) else None


def _score(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and 0 <= result <= 1 else None


def _result_boxes(result) -> list:
    if hasattr(result, "json"):
        result = result.json
        if callable(result):
            result = result()
    if isinstance(result, dict):
        boxes = result.get("boxes")
        if boxes is not None:
            return boxes
        nested = result.get("result")
        if isinstance(nested, dict):
            return nested.get("boxes", [])
        nested = result.get("res")
        if isinstance(nested, dict):
            return nested.get("boxes", [])
        return nested if isinstance(nested, list) else []
    try:
        return result["boxes"]
    except (TypeError, KeyError, IndexError):
        return []


def normalize_layout_result(result, record: ImageRecord) -> Annotation:
    boxes = _result_boxes(result)
    blocks: list[Block] = []
    for item in boxes:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", item.get("class_name", ""))).strip()
        polygon = _polygon(item.get("coordinate", item.get("bbox", item.get("polygon"))))
        if label not in PP_DOCLAYOUTV3_LABEL_SET or polygon is None:
            continue
        blocks.append(
            Block(
                order=len(blocks),
                polygon=polygon,
                layout_label=label,
                task=map_layout_label(label),
                score=_score(item.get("score")),
                source="layout",
            )
        )
    blocks = [block.model_copy(update={"order": index}) for index, block in enumerate(blocks)]
    return normalize_annotation(
        Annotation(
            image=ImageInfo(
                path=record.relative_path,
                width=record.width or 1,
                height=record.height or 1,
                sha256=record.sha256,
            ),
            status="detected",
            blocks=blocks,
        )
    )


class LayoutDetectionEngine:
    def __init__(self, settings, pipeline):
        self.settings = settings
        self.pipeline = pipeline

    @classmethod
    def create(cls, settings):
        settings.validate()
        try:
            from paddleocr import LayoutDetection
        except ImportError as exc:
            raise RuntimeError("PaddleOCR LayoutDetection is unavailable") from exc
        pipeline = LayoutDetection(
            model_name="PP-DocLayoutV3",
            model_dir=str(settings.layout_model_dir.expanduser()),
            device=settings.device,
        )
        return cls(settings, pipeline)

    def detect(self, record: ImageRecord) -> Annotation:
        if record.error or record.width is None or record.height is None:
            raise ValueError(record.error or "image dimensions are unavailable")
        results = self.pipeline.predict(str(record.path))
        results = list(results)
        if len(results) != 1:
            raise RuntimeError(f"expected one layout result, got {len(results)}")
        return normalize_layout_result(results[0], record)

    def close(self) -> None:
        close = getattr(self.pipeline, "close", None)
        if close is not None:
            close()

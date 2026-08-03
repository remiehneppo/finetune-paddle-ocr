import math
from time import perf_counter

from .catalog import ImageRecord
from .geometry import normalize_annotation
from .models import Annotation, Block, ImageInfo, OCRMetadata
from .settings import LabelerSettings


class PaddleOCREngine:
    def __init__(self, settings: LabelerSettings, pipeline):
        self.settings = settings
        self.pipeline = pipeline

    @classmethod
    def create(cls, settings: LabelerSettings) -> "PaddleOCREngine":
        from paddleocr import PaddleOCR

        settings.validate()
        model_args = (
            {"text_detection_model_dir": str(settings.det_model_dir)}
            if settings.det_model_dir
            else {"text_detection_model_name": settings.det_model_name}
        )
        pipeline = PaddleOCR(
            **model_args,
            text_recognition_model_dir=str(settings.rec_model_dir),
            text_rec_input_shape=settings.text_rec_input_shape,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=settings.device,
        )
        return cls(settings=settings, pipeline=pipeline)

    def recognize(self, record: ImageRecord) -> Annotation:
        started = perf_counter()
        results = self.pipeline.predict(
            str(record.path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=self.settings.text_rec_score_thresh,
        )
        if len(results) != 1:
            raise RuntimeError(f"expected one OCR result, got {len(results)}")
        duration_ms = round((perf_counter() - started) * 1000)
        return normalize_ocr_result(results[0], record, self.settings, duration_ms)

    def close(self) -> None:
        close = getattr(self.pipeline, "close", None)
        if close is not None:
            close()


class PaddleOCRDetectionEngine:
    """Detection-only adapter that does not load a recognition model."""

    def __init__(self, settings: LabelerSettings, pipeline):
        self.settings = settings
        self.pipeline = pipeline

    @classmethod
    def create(cls, settings: LabelerSettings) -> "PaddleOCRDetectionEngine":
        from paddleocr import TextDetection

        settings.validate()
        model_args = (
            {"model_dir": str(settings.det_model_dir)}
            if settings.det_model_dir
            else {"model_name": settings.det_model_name}
        )
        pipeline = TextDetection(
            **model_args,
            device=settings.device,
            limit_side_len=settings.text_det_limit_side_len,
            limit_type=settings.text_det_limit_type,
            thresh=settings.text_det_thresh,
            box_thresh=settings.text_det_box_thresh,
            unclip_ratio=settings.text_det_unclip_ratio,
        )
        return cls(settings=settings, pipeline=pipeline)

    def recognize(self, record: ImageRecord) -> Annotation:
        started = perf_counter()
        results = self.pipeline.predict(str(record.path))
        if len(results) != 1:
            raise RuntimeError(f"expected one detection result, got {len(results)}")
        duration_ms = round((perf_counter() - started) * 1000)
        return normalize_detection_result(
            results[0], record, self.settings, duration_ms
        )

    def close(self) -> None:
        close = getattr(self.pipeline, "close", None)
        if close is not None:
            close()


def _as_polygon(value) -> list[tuple[float, float]] | None:
    try:
        if len(value) != 4:
            return None
        polygon = [(float(x), float(y)) for x, y in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in polygon):
        return None
    return polygon


def _as_score(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def normalize_ocr_result(result, record, settings, duration_ms: int) -> Annotation:
    texts = list(result["rec_texts"])
    scores = list(result["rec_scores"])
    polygons = list(result["rec_polys"])
    if not (len(texts) == len(scores) == len(polygons)):
        raise RuntimeError("PaddleOCR result arrays have different lengths")

    blocks = []
    for text, score, raw_polygon in zip(texts, scores, polygons):
        normalized_text = str(text).strip()
        polygon = _as_polygon(raw_polygon)
        normalized_score = _as_score(score)
        if not normalized_text or polygon is None or normalized_score is None:
            continue
        blocks.append(
            Block(
                order=len(blocks),
                text=normalized_text,
                polygon=polygon,
                score=normalized_score,
                source="ocr",
            )
        )

    annotation = Annotation(
        image=ImageInfo(
            path=record.relative_path,
            width=record.width,
            height=record.height,
            sha256=record.sha256,
        ),
        status="ocr",
        blocks=blocks,
        ocr=OCRMetadata(
            det_model=(
                str(settings.det_model_dir)
                if settings.det_model_dir
                else settings.det_model_name
            ),
            rec_model=settings.rec_model_dir.name,
            duration_ms=duration_ms,
        ),
    )
    return normalize_annotation(annotation)


def normalize_detection_result(
    result, record, settings, duration_ms: int
) -> Annotation:
    polygons = list(result["dt_polys"])
    scores = list(result["dt_scores"])
    if len(polygons) != len(scores):
        raise RuntimeError("PaddleOCR detection result arrays have different lengths")

    blocks = []
    for score, raw_polygon in zip(scores, polygons):
        polygon = _as_polygon(raw_polygon)
        normalized_score = _as_score(score)
        if polygon is None or normalized_score is None:
            continue
        blocks.append(
            Block(
                order=len(blocks),
                text="text",
                polygon=polygon,
                score=normalized_score,
                source="ocr",
            )
        )

    annotation = Annotation(
        image=ImageInfo(
            path=record.relative_path,
            width=record.width,
            height=record.height,
            sha256=record.sha256,
        ),
        status="ocr",
        blocks=blocks,
        ocr=OCRMetadata(
            task="detection",
            det_model=(
                str(settings.det_model_dir)
                if settings.det_model_dir
                else settings.det_model_name
            ),
            duration_ms=duration_ms,
        ),
    )
    return normalize_annotation(annotation)

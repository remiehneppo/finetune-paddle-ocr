from .models import Annotation, Block, Point


def clamp_polygon(polygon: list[Point], width: int, height: int) -> list[Point]:
    max_x = float(width - 1)
    max_y = float(height - 1)
    return [
        (min(max(float(x), 0.0), max_x), min(max(float(y), 0.0), max_y))
        for x, y in polygon
    ]


def aggregate_text(blocks: list[Block]) -> str:
    return "\n".join(block.text for block in sorted(blocks, key=lambda item: item.order))


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
    return annotation.model_copy(update={"blocks": blocks, "text": aggregate_text(blocks)})

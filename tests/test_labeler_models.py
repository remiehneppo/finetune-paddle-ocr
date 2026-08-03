import math
import unittest
from uuid import UUID

from pydantic import ValidationError

from ocr_labeler.geometry import clamp_polygon, normalize_annotation
from ocr_labeler.models import Annotation, Block, ImageInfo


class LabelerModelTests(unittest.TestCase):
    def test_annotation_orders_blocks_and_rebuilds_text(self):
        annotation = Annotation(
            image=ImageInfo(
                path="page-001.png", width=100, height=80, sha256="a" * 64
            ),
            blocks=[
                Block(
                    order=7,
                    text="Second",
                    polygon=[(2, 20), (80, 20), (80, 30), (2, 30)],
                    score=0.5,
                    source="ocr",
                ),
                Block(
                    order=2,
                    text="First",
                    polygon=[(1, 1), (90, 1), (90, 10), (1, 10)],
                    score=0.9,
                    source="ocr",
                ),
            ],
        )

        normalized = normalize_annotation(annotation)

        self.assertEqual([block.order for block in normalized.blocks], [0, 1])
        self.assertEqual(normalized.text, "First\nSecond")
        self.assertTrue(all(isinstance(block.id, UUID) for block in normalized.blocks))

    def test_polygon_is_clamped_to_image_bounds(self):
        polygon = [(-2, -3), (120, 0), (101, 91), (0, 90)]
        self.assertEqual(
            clamp_polygon(polygon, width=100, height=80),
            [(0.0, 0.0), (99.0, 0.0), (99.0, 79.0), (0.0, 79.0)],
        )

    def test_polygon_rejects_non_finite_or_wrong_point_count(self):
        with self.assertRaises(ValidationError):
            Block(
                order=0,
                text="bad",
                polygon=[(0, 0), (1, math.inf), (1, 1)],
                score=None,
                source="manual",
            )

    def test_completed_annotation_rejects_empty_block_text(self):
        with self.assertRaises(ValidationError):
            Annotation(
                status="completed",
                image=ImageInfo(
                    path="page.png", width=10, height=10, sha256="b" * 64
                ),
                blocks=[
                    Block(
                        order=0,
                        text=" ",
                        polygon=[(0, 0), (9, 0), (9, 9), (0, 9)],
                        score=None,
                        source="manual",
                    )
                ],
            )

    def test_annotation_rejects_duplicate_block_ids(self):
        shared_id = UUID("12345678-1234-5678-1234-567812345678")
        block = {
            "id": shared_id,
            "text": "text",
            "polygon": [(0, 0), (9, 0), (9, 9), (0, 9)],
            "score": None,
            "source": "manual",
        }
        with self.assertRaisesRegex(ValidationError, "block ids must be unique"):
            Annotation(
                image=ImageInfo(
                    path="page.png", width=10, height=10, sha256="c" * 64
                ),
                blocks=[
                    Block(order=0, **block),
                    Block(order=1, **block),
                ],
            )

    def test_manual_block_rejects_a_confidence_score(self):
        with self.assertRaisesRegex(ValidationError, "manual blocks cannot have a score"):
            Block(
                order=0,
                text="manual",
                polygon=[(0, 0), (9, 0), (9, 9), (0, 9)],
                score=0.5,
                source="manual",
            )

    def test_ocr_block_requires_a_confidence_score(self):
        with self.assertRaisesRegex(ValidationError, "ocr blocks require a score"):
            Block(
                order=0,
                text="ocr",
                polygon=[(0, 0), (9, 0), (9, 9), (0, 9)],
                score=None,
                source="ocr",
            )


if __name__ == "__main__":
    unittest.main()

"""Export orchestration for validated VL labeler annotations."""

from __future__ import annotations

import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .catalog import WorkspaceCatalog
    from .storage import AnnotationStore


class ExportError(RuntimeError):
    pass


def split_layout_pages(pages: list, seed: int = 42) -> tuple[list, list]:
    if len(pages) < 2:
        raise ExportError("layout export requires at least two valid completed pages")
    shuffled = list(pages)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * 0.1)))
    validation_ids = {id(page) for page in shuffled[:validation_count]}
    train = [page for page in pages if id(page) not in validation_ids]
    validation = [page for page in pages if id(page) in validation_ids]
    return train, validation


class AnnotationExportService:
    """Public export seam over a store's validated annotation snapshot methods."""

    def __init__(self, store: AnnotationStore) -> None:
        self.store = store

    def export_hf(self, catalog: WorkspaceCatalog, output_dir: Path) -> dict[str, Any]:
        return self.store._export_hf(catalog, output_dir)

    def export_layout(
        self, catalog: WorkspaceCatalog, output_dir: Path
    ) -> dict[str, Any]:
        return self.store._export_layout(catalog, output_dir)

    def export_all(self, catalog: WorkspaceCatalog, output_dir: Path) -> dict[str, Any]:
        output = output_dir.expanduser().resolve()
        if output.exists():
            raise ExportError(f"export path already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix=".vl-layout-all-export-", dir=output.parent)
        )
        dataset_root = temporary_root / "dataset"
        dataset_root.mkdir()
        try:
            hf = self.export_hf(catalog, dataset_root / "vl")
            layout = self.export_layout(catalog, dataset_root / "layout")
            os.replace(dataset_root, output)
        except BaseException:
            if output.exists():
                shutil.rmtree(output, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
        return {
            "path": str(output),
            "vl": {**hf, "path": str(output / "vl")},
            "layout": {**layout, "path": str(output / "layout")},
        }


__all__ = ["AnnotationExportService", "ExportError", "split_layout_pages"]

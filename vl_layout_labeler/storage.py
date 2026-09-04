from __future__ import annotations

import json
import io
import os
import secrets
import shutil
import stat
import tempfile
import threading
import unicodedata
from datetime import datetime
from pathlib import Path

from PIL import Image
from paddleocr_vl_contract import (
    PP_DOCLAYOUTV3_LABELS,
    PP_DOCLAYOUTV3_LABEL_SET,
    validate_target_for_task,
)

from .catalog import ImageRecord, WorkspaceCatalog, _file_sha256
from .export import AnnotationExportService, ExportError, split_layout_pages
from .geometry import (
    clamp_polygon,
    crop_box_from_polygon,
    normalize_annotation,
    polygon_area,
    polygon_to_xywh,
)
from .models import Annotation, ImageInfo


class RevisionConflict(RuntimeError):
    pass


class SourceImageChanged(RuntimeError):
    pass


class UnsafePersistencePath(ValueError):
    pass


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _open_dir(root: Path, directory: Path, *, create: bool) -> int | None:
    if not _is_within(directory, root):
        raise UnsafePersistencePath("persistence path escapes workspace root")
    relative = directory.relative_to(root)
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise UnsafePersistencePath("invalid persistence path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise UnsafePersistencePath("workspace root is not safe") from exc
    try:
        for part in relative.parts:
            if create:
                try:
                    os.mkdir(part, dir_fd=descriptor)
                except FileExistsError:
                    pass
            try:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                os.close(descriptor)
                return None
            except OSError as exc:
                raise UnsafePersistencePath("persistence directory is not safe") from exc
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _read_text(root: Path, path: Path) -> str | None:
    directory_fd = _open_dir(root, path.parent, create=False)
    if directory_fd is None:
        return None
    try:
        try:
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise UnsafePersistencePath("persistence file is not safe") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise UnsafePersistencePath("persistence file must be regular")
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = -1
                return handle.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    finally:
        os.close(directory_fd)


def _atomic_text(root: Path, path: Path, text: str) -> None:
    directory_fd = _open_dir(root, path.parent, create=True)
    if directory_fd is None:
        raise UnsafePersistencePath("cannot create persistence directory")
    temporary_name = f".{path.name}.{secrets.token_hex(12)}"
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


class AnnotationStore:
    def __init__(self, root: Path, data_dir_name: str = ".paddleocr-vl-labeler"):
        self.root = root.resolve(strict=True)
        self.data_dir = self.root / data_dir_name
        self.annotations_dir = self.data_dir / "annotations"
        self._save_lock = threading.Lock()
        self.exporter = AnnotationExportService(self)

    def _path(self, record: ImageRecord) -> Path:
        return self.annotations_dir / f"{Path(record.name).stem}.json"

    def _assert_source(self, record: ImageRecord, annotation: Annotation | None = None) -> None:
        try:
            current = record.path.resolve(strict=True)
            source_stat = current.stat()
        except (OSError, ValueError) as exc:
            raise SourceImageChanged(record.relative_path) from exc
        if current.parent != self.root or not stat.S_ISREG(source_stat.st_mode):
            raise SourceImageChanged(record.relative_path)
        if (
            source_stat.st_size != record.size_bytes
            or source_stat.st_mtime_ns != record.mtime_ns
            or _file_sha256(current) != record.sha256
        ):
            raise SourceImageChanged(record.relative_path)
        if annotation is not None:
            expected = (record.relative_path, record.width, record.height, record.sha256)
            actual = (
                annotation.image.path,
                annotation.image.width,
                annotation.image.height,
                annotation.image.sha256,
            )
            if actual != expected:
                raise SourceImageChanged(record.relative_path)

    def create_draft(self, record: ImageRecord) -> Annotation:
        if record.error or record.width is None or record.height is None:
            raise ValueError(record.error or "invalid image")
        return Annotation(
            image=ImageInfo(
                path=record.relative_path,
                width=record.width,
                height=record.height,
                sha256=record.sha256,
            )
        )

    def _load_saved(self, record: ImageRecord) -> Annotation | None:
        text = _read_text(self.root, self._path(record))
        return None if text is None else Annotation.model_validate_json(text)

    def has_annotation(self, record: ImageRecord) -> bool:
        return _read_text(self.root, self._path(record)) is not None

    def load(self, record: ImageRecord) -> Annotation:
        self._assert_source(record)
        annotation = self._load_saved(record)
        if annotation is None:
            return self.create_draft(record)
        self._assert_source(record, annotation)
        return annotation

    def save(self, record: ImageRecord, annotation: Annotation) -> Annotation:
        with self._save_lock:
            self._assert_source(record, annotation)
            current = self._load_saved(record)
            current_revision = current.revision if current is not None else 0
            if annotation.revision != current_revision:
                raise RevisionConflict(
                    f"expected revision {current_revision}, got {annotation.revision}"
                )
            annotation = annotation.model_copy(
                update={
                    "blocks": [
                        block.model_copy(update={"validation": None})
                        if (
                            block.validation is not None
                            and block.validation.text_hash != block.current_text_hash()
                        )
                        else block
                        for block in annotation.blocks
                    ]
                }
            )
            if (
                current is not None
                and current.status == "completed"
                and annotation.status == "completed"
                and (annotation.image != current.image or annotation.blocks != current.blocks)
            ):
                annotation = annotation.model_copy(update={"status": "edited"})
            saved = normalize_annotation(annotation).model_copy(
                update={
                    "revision": current_revision + 1,
                    "updated_at": datetime.now().astimezone(),
                }
            )
            _atomic_text(self.root, self._path(record), saved.model_dump_json(indent=2))
            return saved

    def _export_hf(self, catalog: WorkspaceCatalog, output_dir: Path) -> dict:
        output = output_dir.expanduser().resolve()
        if output.exists():
            raise ExportError(f"export path already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        pages: list[dict] = []
        for record in catalog.list_images():
            annotation = self._load_saved(record)
            if annotation is None or annotation.status != "completed":
                continue
            self._assert_source(record, annotation)
            page_rows: list[dict] = []
            with Image.open(record.path) as source:
                image = source.convert("RGB")
                for block in sorted(annotation.blocks, key=lambda item: item.order):
                    text = unicodedata.normalize("NFC", block.text).replace(
                        "\r\n", "\n"
                    ).replace("\r", "\n")
                    if block.skipped or block.task is None or not text.strip():
                        continue
                    try:
                        validate_target_for_task(text, block.task)
                    except ValueError as exc:
                        raise ExportError(
                            f"invalid {block.task} target on {record.name} block "
                            f"{block.id}: {exc}"
                        ) from exc
                    crop = image.crop(
                        crop_box_from_polygon(block.polygon, image.width, image.height)
                    )
                    name = f"{Path(record.name).stem}-{block.order:04d}-{block.id}.png"
                    buffer = io.BytesIO()
                    crop.save(buffer, format="PNG")
                    page_rows.append(
                        {
                            "image": {"bytes": buffer.getvalue(), "path": name},
                            "name": name,
                            "text": text,
                            "task": block.task,
                            "source_page_id": record.image_id,
                        }
                    )
            if page_rows:
                pages.append({"record": record, "rows": page_rows})
        if len(pages) < 2:
            raise ExportError(
                "HF export requires at least two completed pages with valid VL samples"
            )
        train_pages, validation_pages = split_layout_pages(pages)
        split_rows = {
            "train": [row for page in train_pages for row in page["rows"]],
            "validation": [
                row for page in validation_pages for row in page["rows"]
            ],
        }
        temporary_root = Path(
            tempfile.mkdtemp(prefix=".vl-layout-export-", dir=output.parent)
        )
        dataset_dir = temporary_root / "dataset"
        try:
            try:
                from datasets import (
                    Dataset,
                    DatasetDict,
                    Features,
                    Image as HFImage,
                    Value,
                )
            except ImportError as exc:
                raise ExportError("datasets is required for HF export") from exc
            features = Features(
                {
                    "image": HFImage(),
                    "text": Value("string"),
                    "task": Value("string"),
                    "source_page_id": Value("string"),
                }
            )
            dataset = DatasetDict(
                {
                    split_name: Dataset.from_dict(
                        {
                            "image": [row["image"] for row in rows],
                            "text": [row["text"] for row in rows],
                            "task": [row["task"] for row in rows],
                            "source_page_id": [
                                row["source_page_id"] for row in rows
                            ],
                        },
                        features=features,
                    )
                    for split_name, rows in split_rows.items()
                }
            )
            dataset.save_to_disk(dataset_dir)
            crops_dir = dataset_dir / "crops"
            crops_dir.mkdir()
            for rows in split_rows.values():
                for row in rows:
                    (crops_dir / row["name"]).write_bytes(row["image"]["bytes"])
            (dataset_dir / "vl_layout_labeler_export.json").write_text(
                json.dumps(
                    {
                        "samples": sum(len(rows) for rows in split_rows.values()),
                        "pages": len(pages),
                        "splits": {
                            name: {
                                "pages": len(train_pages)
                                if name == "train"
                                else len(validation_pages),
                                "samples": len(rows),
                            }
                            for name, rows in split_rows.items()
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(dataset_dir, output)
        except BaseException:
            if output.exists():
                shutil.rmtree(output, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
        return {
            "path": str(output),
            "samples": sum(len(rows) for rows in split_rows.values()),
            "pages": len(pages),
            "train_pages": len(train_pages),
            "validation_pages": len(validation_pages),
        }

    def _layout_page(self, record: ImageRecord, annotation: Annotation) -> dict | None:
        blocks = []
        for block in sorted(annotation.blocks, key=lambda item: item.order):
            if block.skipped or block.layout_label not in PP_DOCLAYOUTV3_LABEL_SET:
                continue
            polygon = clamp_polygon(
                block.polygon, annotation.image.width, annotation.image.height
            )
            area = polygon_area(polygon)
            if area <= 0:
                continue
            blocks.append(
                {
                    "block": block,
                    "polygon": polygon,
                    "area": area,
                }
            )
        if not blocks:
            return None
        return {"record": record, "annotation": annotation, "blocks": blocks}

    @staticmethod
    def _coco_split(pages: list[dict], image_ids: dict[str, int]) -> dict:
        category_ids = {
            label: index for index, label in enumerate(PP_DOCLAYOUTV3_LABELS, start=1)
        }
        images = []
        annotations = []
        annotation_id = 1
        for page in pages:
            record = page["record"]
            annotation = page["annotation"]
            image_id = image_ids[record.image_id]
            images.append(
                {
                    "id": image_id,
                    "file_name": record.name,
                    "width": annotation.image.width,
                    "height": annotation.image.height,
                }
            )
            for read_order, item in enumerate(page["blocks"]):
                block = item["block"]
                polygon = item["polygon"]
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_ids[block.layout_label],
                        "segmentation": [
                            [coordinate for point in polygon for coordinate in point]
                        ],
                        "bbox": polygon_to_xywh(polygon),
                        "area": item["area"],
                        "iscrowd": 0,
                        "read_order": read_order,
                    }
                )
                annotation_id += 1
        return {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": index, "name": label}
                for index, label in enumerate(PP_DOCLAYOUTV3_LABELS, start=1)
            ],
        }

    def _export_layout(self, catalog: WorkspaceCatalog, output_dir: Path) -> dict:
        output = output_dir.expanduser().resolve()
        if output.exists():
            raise ExportError(f"export path already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        pages = []
        for record in catalog.list_images():
            annotation = self._load_saved(record)
            if annotation is None or annotation.status != "completed":
                continue
            self._assert_source(record, annotation)
            page = self._layout_page(record, annotation)
            if page is not None:
                pages.append(page)
        train_pages, validation_pages = split_layout_pages(pages)
        image_ids = {
            page["record"].image_id: index for index, page in enumerate(pages, start=1)
        }
        temporary_root = Path(
            tempfile.mkdtemp(prefix=".layout-export-", dir=output.parent)
        )
        dataset_dir = temporary_root / "dataset"
        try:
            images_dir = dataset_dir / "images"
            annotations_dir = dataset_dir / "annotations"
            images_dir.mkdir(parents=True)
            annotations_dir.mkdir()
            for page in pages:
                record = page["record"]
                destination = images_dir / record.name
                shutil.copyfile(record.path, destination)
                if _file_sha256(destination) != record.sha256:
                    raise ExportError(f"source image changed during export: {record.name}")
            train_payload = self._coco_split(train_pages, image_ids)
            validation_payload = self._coco_split(validation_pages, image_ids)
            (annotations_dir / "instance_train.json").write_text(
                json.dumps(train_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (annotations_dir / "instance_val.json").write_text(
                json.dumps(validation_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest = {
                "format": "COCOInstSegDataset",
                "seed": 42,
                "categories": len(PP_DOCLAYOUTV3_LABELS),
                "pages": len(pages),
                "annotations": sum(len(page["blocks"]) for page in pages),
                "train_pages": [page["record"].name for page in train_pages],
                "validation_pages": [
                    page["record"].name for page in validation_pages
                ],
            }
            (dataset_dir / "export_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(dataset_dir, output)
        except BaseException:
            if output.exists():
                shutil.rmtree(output, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
        return {
            "path": str(output),
            "pages": len(pages),
            "annotations": sum(len(page["blocks"]) for page in pages),
        }

    def export_hf(self, catalog: WorkspaceCatalog, output_dir: Path) -> dict:
        return self.exporter.export_hf(catalog, output_dir)

    def export_layout(self, catalog: WorkspaceCatalog, output_dir: Path) -> dict:
        return self.exporter.export_layout(catalog, output_dir)

    def export_all(self, catalog: WorkspaceCatalog, output_dir: Path) -> dict:
        return self.exporter.export_all(catalog, output_dir)

import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from ocr_labeler.catalog import DuplicateStemError, WorkspaceCatalog
from ocr_labeler.models import Block
from ocr_labeler.storage import (
    AnnotationStore,
    RevisionConflict,
    SourceImageChanged,
    UnsafePersistencePath,
)


def make_image(path: Path, size=(64, 32), color="white"):
    Image.new("RGB", size, color).save(path)


class CatalogStorageTests(unittest.TestCase):
    def test_catalog_scans_direct_images_only_and_uses_opaque_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page-2.PNG")
            make_image(root / "page-10.jpg")
            nested = root / "nested"
            nested.mkdir()
            make_image(nested / "ignored.png")

            catalog = WorkspaceCatalog.open(root)
            records = catalog.list_images()

        self.assertEqual(
            [record.name for record in records], ["page-2.PNG", "page-10.jpg"]
        )
        self.assertTrue(all("/" not in record.image_id for record in records))

    def test_duplicate_stems_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            make_image(root / "page.jpg")
            with self.assertRaises(DuplicateStemError):
                WorkspaceCatalog.open(root)

    def test_symlink_cannot_escape_workspace_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "images"
            root.mkdir()
            outside = base / "outside.png"
            make_image(outside)
            (root / "leak.png").symlink_to(outside)
            with self.assertRaises(ValueError):
                WorkspaceCatalog.open(root)

    def test_corrupt_image_is_listed_as_error_without_blocking_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.png").write_bytes(b"not an image")
            records = WorkspaceCatalog.open(root).list_images()
        self.assertEqual(len(records), 1)
        self.assertIsNotNone(records[0].error)

    def test_save_is_revision_checked_and_export_is_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            catalog = WorkspaceCatalog.open(root)
            record = catalog.list_images()[0]
            store = AnnotationStore(root)
            annotation = store.create_draft(record)
            annotation.blocks = [
                Block(
                    order=0,
                    text="Việt Nam",
                    polygon=[(1, 1), (50, 1), (50, 20), (1, 20)],
                    score=None,
                    source="manual",
                )
            ]

            saved = store.save(record, annotation)
            with self.assertRaises(RevisionConflict):
                store.save(record, annotation)
            manifest = store.export_manifest(catalog)
            row = json.loads(manifest.read_text(encoding="utf-8").strip())

        self.assertEqual(saved.revision, 1)
        self.assertEqual(row["image"], "page.png")
        self.assertEqual(row["text"], "Việt Nam")
        self.assertEqual(len(row["blocks"]), 1)

    def test_changed_source_image_blocks_load_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            catalog = WorkspaceCatalog.open(root)
            record = catalog.list_images()[0]
            store = AnnotationStore(root)
            store.save(record, store.create_draft(record))
            make_image(root / "page.png", size=(80, 40), color="black")
            with self.assertRaises(SourceImageChanged):
                store.load(record)
            with self.assertRaises(SourceImageChanged):
                store.export_manifest(catalog)

    def test_changed_bytes_with_same_size_and_mtime_block_load_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            make_image(image_path)
            catalog = WorkspaceCatalog.open(root)
            record = catalog.list_images()[0]
            store = AnnotationStore(root)
            store.save(record, store.create_draft(record))
            before = image_path.stat()
            changed = bytearray(image_path.read_bytes())
            changed[-1] ^= 1
            image_path.write_bytes(changed)
            os.utime(image_path, ns=(before.st_atime_ns, before.st_mtime_ns))

            with self.assertRaises(SourceImageChanged):
                store.load(record)
            with self.assertRaises(SourceImageChanged):
                store.export_manifest(catalog)

    def test_stale_source_without_sidecar_is_rejected_on_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            make_image(image_path)
            record = WorkspaceCatalog.open(root).list_images()[0]
            image_path.unlink()

            with self.assertRaises(SourceImageChanged):
                AnnotationStore(root).load(record)

    def test_persistence_symlink_escape_is_rejected_without_writing_outside(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "images"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            make_image(root / "page.png")
            (root / ".paddleocr-labeler").symlink_to(outside, target_is_directory=True)
            record = WorkspaceCatalog.open(root).list_images()[0]

            with self.assertRaises(ValueError):
                AnnotationStore(root).save(record, AnnotationStore(root).create_draft(record))

            self.assertEqual(list(outside.iterdir()), [])

    def test_symlinked_persistence_directory_is_never_read_or_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "images"
            outside = base / "outside"
            root.mkdir()
            (outside / "annotations").mkdir(parents=True)
            make_image(root / "page.png")
            record = WorkspaceCatalog.open(root).list_images()[0]
            draft = AnnotationStore(root).create_draft(record)
            sentinel = outside / "annotations" / "page.json"
            sentinel.write_text(draft.model_dump_json(), encoding="utf-8")
            (root / ".paddleocr-labeler").symlink_to(
                outside, target_is_directory=True
            )
            store = AnnotationStore(root)

            with self.assertRaises(ValueError):
                store.has_annotation(record)
            with self.assertRaises(ValueError):
                store.load(record)
            with self.assertRaises(ValueError):
                store.export_manifest(WorkspaceCatalog.open(root))

            self.assertEqual(sentinel.read_text(encoding="utf-8"), draft.model_dump_json())
            self.assertFalse((outside / "manifest.jsonl").exists())

    def test_symlinked_sidecar_is_never_read_or_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "images"
            root.mkdir()
            make_image(root / "page.png")
            catalog = WorkspaceCatalog.open(root)
            record = catalog.list_images()[0]
            store = AnnotationStore(root)
            store.annotation_dir.mkdir(parents=True)
            sentinel = base / "sentinel.json"
            draft = store.create_draft(record)
            sentinel.write_text(draft.model_dump_json(), encoding="utf-8")
            store._path(record).symlink_to(sentinel)

            with self.assertRaises(ValueError):
                store.has_annotation(record)
            with self.assertRaises(ValueError):
                store.load(record)
            with self.assertRaises(ValueError):
                store.export_manifest(catalog)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), draft.model_dump_json())
            self.assertFalse((store.data_dir / "manifest.jsonl").exists())

    def test_unsafe_persistence_uses_a_specific_domain_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            record = WorkspaceCatalog.open(root).list_images()[0]
            outside = root / "outside"
            outside.mkdir()
            (root / ".paddleocr-labeler").symlink_to(
                outside, target_is_directory=True
            )

            with self.assertRaises(UnsafePersistencePath):
                AnnotationStore(root).load(record)

    def test_concurrent_same_revision_saves_allow_only_one_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            record = WorkspaceCatalog.open(root).list_images()[0]
            store = AnnotationStore(root)
            annotation = store.create_draft(record)
            start = threading.Barrier(3)
            first_atomic_write = threading.Event()
            second_atomic_write = threading.Event()
            results = []
            errors = []
            original_atomic_text = __import__("ocr_labeler.storage", fromlist=["_atomic_text"])._atomic_text

            def synchronized_atomic_text(root, path, text):
                if first_atomic_write.is_set():
                    second_atomic_write.set()
                else:
                    first_atomic_write.set()
                    second_atomic_write.wait(timeout=0.5)
                original_atomic_text(root, path, text)

            def save():
                start.wait()
                try:
                    results.append(store.save(record, annotation))
                except Exception as exc:
                    errors.append(exc)

            with patch("ocr_labeler.storage._atomic_text", synchronized_atomic_text):
                writers = [threading.Thread(target=save) for _ in range(2)]
                for writer in writers:
                    writer.start()
                start.wait()
                for writer in writers:
                    writer.join()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].revision, 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RevisionConflict)

    def test_two_stores_same_revision_allow_only_one_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            record = WorkspaceCatalog.open(root).list_images()[0]
            stores = [AnnotationStore(root), AnnotationStore(root)]
            annotation = stores[0].create_draft(record)
            start = threading.Barrier(3)
            first_atomic_write = threading.Event()
            second_atomic_write = threading.Event()
            results = []
            errors = []
            original_atomic_text = __import__(
                "ocr_labeler.storage", fromlist=["_atomic_text"]
            )._atomic_text

            def synchronized_atomic_text(root, path, text):
                if first_atomic_write.is_set():
                    second_atomic_write.set()
                else:
                    first_atomic_write.set()
                    second_atomic_write.wait(timeout=0.5)
                original_atomic_text(root, path, text)

            def save(store):
                start.wait()
                try:
                    results.append(store.save(record, annotation))
                except Exception as exc:
                    errors.append(exc)

            with patch("ocr_labeler.storage._atomic_text", synchronized_atomic_text):
                writers = [threading.Thread(target=save, args=(store,)) for store in stores]
                for writer in writers:
                    writer.start()
                start.wait()
                for writer in writers:
                    writer.join()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].revision, 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RevisionConflict)

    def test_internal_persistence_symlink_is_rejected_without_redirecting_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inside = root / "inside"
            outside = root / "outside"
            inside.mkdir()
            (outside / "annotations").mkdir(parents=True)
            make_image(root / "page.png")
            persistence_link = root / ".paddleocr-labeler"
            persistence_link.symlink_to(inside, target_is_directory=True)
            record = WorkspaceCatalog.open(root).list_images()[0]
            store = AnnotationStore(root)
            with self.assertRaises(ValueError):
                store.save(record, store.create_draft(record))

            self.assertEqual(list(inside.iterdir()), [])
            self.assertEqual(list((outside / "annotations").iterdir()), [])

    def test_internal_image_symlink_can_be_saved_and_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.png"
            make_image(target)
            (root / "page.png").symlink_to(target)
            catalog = WorkspaceCatalog.open(root)
            record = next(item for item in catalog.list_images() if item.name == "page.png")
            store = AnnotationStore(root)

            saved = store.save(record, store.create_draft(record))
            loaded = store.load(record)

        self.assertEqual(saved.revision, 1)
        self.assertEqual(loaded.revision, 1)

    def test_atomic_save_leaves_no_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            record = WorkspaceCatalog.open(root).list_images()[0]
            store = AnnotationStore(root)
            store.save(record, store.create_draft(record))
            leftovers = list(store.annotation_dir.glob(".page.json.*"))
        self.assertEqual(leftovers, [])

    def test_atomic_save_fsyncs_file_and_containing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_image(root / "page.png")
            record = WorkspaceCatalog.open(root).list_images()[0]
            store = AnnotationStore(root)
            fsynced_kinds = []
            real_fsync = os.fsync

            def recording_fsync(fd):
                mode = os.fstat(fd).st_mode
                fsynced_kinds.append("directory" if stat.S_ISDIR(mode) else "file")
                return real_fsync(fd)

            with patch("ocr_labeler.storage.os.fsync", side_effect=recording_fsync):
                store.save(record, store.create_draft(record))

        self.assertEqual(fsynced_kinds, ["file", "directory"])

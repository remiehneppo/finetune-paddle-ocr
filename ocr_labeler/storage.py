import json
import os
import secrets
import stat
import threading
from datetime import datetime
from pathlib import Path

from .catalog import ImageRecord, WorkspaceCatalog, _file_sha256
from .geometry import normalize_annotation
from .models import Annotation, ImageInfo


class RevisionConflict(RuntimeError):
    pass


class SourceImageChanged(RuntimeError):
    pass


class UnsafePersistencePath(ValueError):
    pass


_WORKSPACE_LOCKS: dict[Path, threading.Lock] = {}
_WORKSPACE_LOCKS_GUARD = threading.Lock()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _workspace_lock(root: Path) -> threading.Lock:
    with _WORKSPACE_LOCKS_GUARD:
        return _WORKSPACE_LOCKS.setdefault(root, threading.Lock())


def _open_persistence_dir(root: Path, path: Path) -> int:
    if not _is_within(path, root):
        raise UnsafePersistencePath("persistence path escapes workspace root")
    relative = path.relative_to(root)
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise UnsafePersistencePath("invalid persistence path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        directory_fd = os.open(root, flags)
    except OSError as exc:
        raise UnsafePersistencePath(
            "workspace root is not a safe directory"
        ) from exc
    try:
        for part in relative.parent.parts:
            try:
                os.mkdir(part, dir_fd=directory_fd)
            except FileExistsError:
                pass
            except OSError as exc:
                raise UnsafePersistencePath(
                    "persistence directory is not safe"
                ) from exc
            try:
                next_fd = os.open(part, flags, dir_fd=directory_fd)
            except OSError as exc:
                raise UnsafePersistencePath(
                    "persistence directory is not safe"
                ) from exc
            os.close(directory_fd)
            directory_fd = next_fd
    except BaseException:
        os.close(directory_fd)
        raise
    return directory_fd


def _open_existing_persistence_dir(root: Path, path: Path) -> int | None:
    if not _is_within(path, root):
        raise UnsafePersistencePath("persistence path escapes workspace root")
    relative = path.relative_to(root)
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise UnsafePersistencePath("invalid persistence path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        directory_fd = os.open(root, flags)
    except OSError as exc:
        raise UnsafePersistencePath(
            "workspace root is not a safe directory"
        ) from exc
    try:
        for part in relative.parent.parts:
            try:
                next_fd = os.open(part, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                os.close(directory_fd)
                return None
            except OSError as exc:
                raise UnsafePersistencePath(
                    "persistence directory is not safe"
                ) from exc
            os.close(directory_fd)
            directory_fd = next_fd
    except BaseException:
        try:
            os.close(directory_fd)
        except OSError:
            pass
        raise
    return directory_fd


def _read_persistence_text(root: Path, path: Path) -> str | None:
    directory_fd = _open_existing_persistence_dir(root, path)
    if directory_fd is None:
        return None
    try:
        try:
            fd = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise UnsafePersistencePath("persistence file is not safe") from exc
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise UnsafePersistencePath("persistence file must be regular")
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                fd = -1
                return handle.read()
        finally:
            if fd >= 0:
                os.close(fd)
    finally:
        os.close(directory_fd)


def _atomic_text(root: Path, path: Path, text: str) -> None:
    directory_fd = _open_persistence_dir(root, path)
    temp_name = f".{path.name}.{secrets.token_hex(16)}"
    try:
        fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temp_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except BaseException:
        try:
            os.unlink(temp_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(directory_fd)


class AnnotationStore:
    def __init__(self, root: Path, data_dir_name: str = ".paddleocr-labeler"):
        if (
            not data_dir_name.startswith(".")
            or "/" in data_dir_name
            or "\\" in data_dir_name
            or data_dir_name in {".", ".."}
        ):
            raise ValueError("invalid data directory name")
        self.root = root.resolve()
        self.data_dir = self.root / data_dir_name
        self.annotation_dir = self.data_dir / "annotations"
        self._save_lock = _workspace_lock(self.root)

    def _path(self, record: ImageRecord) -> Path:
        return self.annotation_dir / f"{Path(record.relative_path).stem}.json"

    def has_annotation(self, record: ImageRecord) -> bool:
        return _read_persistence_text(self.root, self._path(record)) is not None

    def create_draft(self, record: ImageRecord) -> Annotation:
        if record.error or record.width is None or record.height is None:
            raise ValueError(f"invalid source image: {record.name}: {record.error}")
        return Annotation(
            image=ImageInfo(
                path=record.relative_path,
                width=record.width,
                height=record.height,
                sha256=record.sha256,
            )
        )

    def load(self, record: ImageRecord) -> Annotation:
        self._assert_source(record)
        annotation = self._load_saved(record)
        if annotation is None:
            return self.create_draft(record)
        self._assert_source(record, annotation)
        return annotation

    def _load_saved(self, record: ImageRecord) -> Annotation | None:
        text = _read_persistence_text(self.root, self._path(record))
        if text is None:
            return None
        return Annotation.model_validate_json(text)

    def _assert_source(
        self, record: ImageRecord, annotation: Annotation | None = None
    ) -> None:
        expected_path = self.root / record.relative_path
        try:
            current_path = expected_path.resolve(strict=True)
            source_stat = current_path.stat()
            if not _is_within(current_path, self.root) or not stat.S_ISREG(
                source_stat.st_mode
            ):
                raise SourceImageChanged(record.relative_path)
            current_sha256 = _file_sha256(current_path)
        except SourceImageChanged:
            raise
        except (OSError, ValueError) as exc:
            raise SourceImageChanged(record.relative_path) from exc
        if (
            source_stat.st_size != record.size_bytes
            or source_stat.st_mtime_ns != record.mtime_ns
            or current_sha256 != record.sha256
        ):
            raise SourceImageChanged(record.relative_path)
        if annotation is None:
            return
        expected = (record.sha256, record.width, record.height, record.relative_path)
        actual = (
            annotation.image.sha256,
            annotation.image.width,
            annotation.image.height,
            annotation.image.path,
        )
        if actual != expected:
            raise SourceImageChanged(record.relative_path)

    def save(self, record: ImageRecord, annotation: Annotation) -> Annotation:
        with self._save_lock:
            self._assert_source(record, annotation)
            path = self._path(record)
            current_revision = 0
            current = self._load_saved(record)
            if current is not None:
                current_revision = current.revision
            if annotation.revision != current_revision:
                raise RevisionConflict(
                    f"expected revision {current_revision}, got {annotation.revision}"
                )
            saved = normalize_annotation(annotation).model_copy(
                update={
                    "revision": current_revision + 1,
                    "updated_at": datetime.now().astimezone(),
                }
            )
            _atomic_text(self.root, path, saved.model_dump_json(indent=2))
        return saved

    def export_manifest(self, catalog: WorkspaceCatalog) -> Path:
        rows = []
        for record in catalog.list_images():
            self._assert_source(record)
            annotation = self._load_saved(record)
            if annotation is None:
                continue
            self._assert_source(record, annotation)
            rows.append(
                json.dumps(
                    {
                        "image": annotation.image.path,
                        "width": annotation.image.width,
                        "height": annotation.image.height,
                        "text": annotation.text,
                        "blocks": [
                            block.model_dump(mode="json") for block in annotation.blocks
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        manifest = self.data_dir / "manifest.jsonl"
        _atomic_text(self.root, manifest, "".join(f"{row}\n" for row in rows))
        return manifest

    def export_detection_labels(self, catalog: WorkspaceCatalog) -> Path:
        rows = []
        for record in catalog.list_images():
            self._assert_source(record)
            annotation = self._load_saved(record)
            if annotation is None:
                continue
            self._assert_source(record, annotation)
            labels = [
                {
                    "transcription": block.text.strip() or "text",
                    "points": [[round(x), round(y)] for x, y in block.polygon],
                }
                for block in sorted(
                    annotation.blocks, key=lambda item: item.order
                )
            ]
            rows.append(
                f"{annotation.image.path}\t"
                f"{json.dumps(labels, ensure_ascii=False, separators=(chr(44), chr(58)))}"
            )
        labels_path = self.data_dir / "det_labels.txt"
        _atomic_text(
            self.root,
            labels_path,
            "".join(f"{row}\n" for row in rows),
        )
        return labels_path

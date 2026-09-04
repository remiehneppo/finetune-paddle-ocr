from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class DuplicateStemError(ValueError):
    pass


class UnknownImageError(KeyError):
    pass


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    name: str
    path: Path
    relative_path: str
    width: int | None
    height: int | None
    sha256: str
    size_bytes: int
    mtime_ns: int
    error: str | None = None


def _natural_key(value: str):
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WorkspaceCatalog:
    def __init__(self, root: Path, records: list[ImageRecord]):
        self.root = root
        self._records = records
        self._by_id = {record.image_id: record for record in records}

    @classmethod
    def open(cls, root: Path) -> "WorkspaceCatalog":
        resolved = root.expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("workspace root must be a directory")
        paths = sorted(
            (
                path
                for path in resolved.iterdir()
                if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
            ),
            key=lambda path: _natural_key(path.name),
        )
        stems = [path.stem.casefold() for path in paths]
        if len(stems) != len(set(stems)):
            raise DuplicateStemError("image filenames must have unique stems")
        records = []
        for path in paths:
            resolved_path = path.resolve(strict=True)
            if resolved_path.parent != resolved:
                raise ValueError(f"image symlink escapes workspace: {path.name}")
            relative = path.relative_to(resolved).as_posix()
            image_id = sha256(relative.encode("utf-8")).hexdigest()[:24]
            source_stat = path.stat()
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                error = None
            except Exception as exc:
                width, height = None, None
                error = f"{type(exc).__name__}: {exc}"
            records.append(
                ImageRecord(
                    image_id=image_id,
                    name=path.name,
                    path=path,
                    relative_path=relative,
                    width=width,
                    height=height,
                    sha256=_file_sha256(path),
                    size_bytes=source_stat.st_size,
                    mtime_ns=source_stat.st_mtime_ns,
                    error=error,
                )
            )
        return cls(resolved, records)

    def list_images(self) -> list[ImageRecord]:
        return list(self._records)

    def get(self, image_id: str) -> ImageRecord:
        try:
            record = self._by_id[image_id]
        except KeyError as exc:
            raise UnknownImageError(image_id) from exc
        if record.path.resolve(strict=True).parent != self.root:
            raise UnknownImageError(image_id)
        return record


__all__ = [
    "DuplicateStemError",
    "ImageRecord",
    "SUPPORTED_EXTENSIONS",
    "UnknownImageError",
    "WorkspaceCatalog",
    "_file_sha256",
]

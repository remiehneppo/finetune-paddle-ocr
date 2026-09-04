"""Safe source-image streaming shared by OCR and VL labeler applications."""

from __future__ import annotations

import os
import stat
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Type

from fastapi.responses import StreamingResponse

from labeler_catalog import ImageRecord, WorkspaceCatalog


class OpenFileStreamingResponse(StreamingResponse):
    def __init__(self, fd: int, *, content_length: int, media_type: str):
        self._fd = fd
        self._fd_lock = Lock()
        super().__init__(
            self._iter_chunks(),
            media_type=media_type,
            headers={"Content-Length": str(content_length)},
        )

    async def _iter_chunks(self):
        while True:
            chunk = os.read(self._fd, 1024 * 1024)
            if not chunk:
                return
            yield chunk

    def close(self) -> None:
        with self._fd_lock:
            fd = self._fd
            self._fd = -1
        if fd >= 0:
            os.close(fd)

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.close()


def open_source_image(
    catalog: WorkspaceCatalog,
    record: ImageRecord,
    source_changed_error: Type[Exception],
) -> tuple[int, int]:
    relative = Path(record.relative_path)
    if len(relative.parts) != 1 or relative.name != record.name:
        raise source_changed_error(record.relative_path)
    directory_fd = -1
    image_fd = -1
    try:
        resolved_target = record.path.resolve(strict=True)
        if resolved_target.parent != catalog.root:
            raise source_changed_error(record.relative_path)
        directory_fd = os.open(
            catalog.root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        image_fd = os.open(
            resolved_target.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        image_stat = os.fstat(image_fd)
        if not stat.S_ISREG(image_stat.st_mode):
            raise source_changed_error(record.relative_path)
        digest = sha256()
        while chunk := os.read(image_fd, 1024 * 1024):
            digest.update(chunk)
        os.lseek(image_fd, 0, os.SEEK_SET)
        if (
            image_stat.st_size != record.size_bytes
            or image_stat.st_mtime_ns != record.mtime_ns
            or digest.hexdigest() != record.sha256
        ):
            raise source_changed_error(record.relative_path)
        return image_fd, image_stat.st_size
    except source_changed_error:
        if image_fd >= 0:
            os.close(image_fd)
        raise
    except (OSError, ValueError) as exc:
        if image_fd >= 0:
            os.close(image_fd)
        raise source_changed_error(record.relative_path) from exc
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)


__all__ = ["OpenFileStreamingResponse", "open_source_image"]

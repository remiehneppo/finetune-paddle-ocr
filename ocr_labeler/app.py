from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
import mimetypes
import os
from pathlib import Path
import stat
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from .batch import BatchManager, InferenceCoordinator
from .catalog import UnknownImageError, WorkspaceCatalog
from .models import Annotation
from .ocr_engine import PaddleOCRDetectionEngine, PaddleOCREngine
from .storage import (
    AnnotationStore,
    RevisionConflict,
    SourceImageChanged,
    UnsafePersistencePath,
)


class OpenWorkspaceRequest(BaseModel):
    path: str


class OCRRequest(BaseModel):
    replace_existing: bool = False


@dataclass(frozen=True)
class WorkspacePair:
    catalog: WorkspaceCatalog
    store: AnnotationStore


class AppState:
    def __init__(self, settings, engine):
        self.settings = settings
        self.engine = engine
        self.coordinator = InferenceCoordinator(engine)
        self.batch = BatchManager(self.coordinator)
        self._workspace = None
        self.workspace_lock = Lock()

    def require_workspace(self):
        with self.workspace_lock:
            return self._require_workspace_locked()

    def _require_workspace_locked(self):
        workspace = self._workspace
        if workspace is None:
            raise HTTPException(status_code=409, detail="open a workspace first")
        return workspace.catalog, workspace.store

    def _set_workspace_locked(self, catalog):
        self._workspace = WorkspacePair(
            catalog,
            AnnotationStore(catalog.root, self.settings.data_dir_name),
        )


class OpenFileStreamingResponse(StreamingResponse):
    def __init__(
        self,
        fd: int,
        *,
        content_length: int,
        media_type: str,
    ):
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


def _open_source_image(catalog: WorkspaceCatalog, record) -> tuple[int, int]:
    relative = Path(record.relative_path)
    if len(relative.parts) != 1 or relative.name != record.name:
        raise SourceImageChanged(record.relative_path)
    directory_fd = -1
    image_fd = -1
    try:
        resolved_target = record.path.resolve(strict=True)
        if resolved_target.parent != catalog.root:
            raise SourceImageChanged(record.relative_path)
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
            raise SourceImageChanged(record.relative_path)
        digest = sha256()
        while chunk := os.read(image_fd, 1024 * 1024):
            digest.update(chunk)
        os.lseek(image_fd, 0, os.SEEK_SET)
        if (
            image_stat.st_size != record.size_bytes
            or image_stat.st_mtime_ns != record.mtime_ns
            or digest.hexdigest() != record.sha256
        ):
            raise SourceImageChanged(record.relative_path)
        return image_fd, image_stat.st_size
    except SourceImageChanged:
        if image_fd >= 0:
            os.close(image_fd)
        raise
    except (OSError, ValueError) as exc:
        if image_fd >= 0:
            os.close(image_fd)
        raise SourceImageChanged(record.relative_path) from exc
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)


def create_app(settings, engine=None, initial_workspace=None) -> FastAPI:
    owns_engine = engine is None
    if owns_engine:
        active_engine = (
            PaddleOCRDetectionEngine.create(settings)
            if settings.task == "detection"
            else PaddleOCREngine.create(settings)
        )
    else:
        active_engine = engine
    state = AppState(settings, active_engine)

    @asynccontextmanager
    async def lifespan(app):
        try:
            if initial_workspace is not None:
                catalog = WorkspaceCatalog.open(Path(initial_workspace))
                with state.workspace_lock:
                    state._set_workspace_locked(catalog)
            yield
        finally:
            try:
                await state.batch.shutdown()
            finally:
                if owns_engine:
                    active_engine.close()

    app = FastAPI(
        title=(
            "PaddleOCR Detection Labeler"
            if settings.task == "detection"
            else "PaddleOCR Labeler"
        ),
        lifespan=lifespan,
    )
    app.state.labeler = state
    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static_dir / "index.html")

    @app.exception_handler(UnknownImageError)
    async def unknown_image_handler(request, exc):
        return JSONResponse(status_code=404, content={"detail": "unknown image"})

    @app.exception_handler(RevisionConflict)
    async def revision_handler(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(SourceImageChanged)
    async def source_changed_handler(request, exc):
        return JSONResponse(
            status_code=409,
            content={"detail": f"source image changed: {exc}"},
        )

    @app.exception_handler(UnsafePersistencePath)
    async def unsafe_persistence_handler(request, exc):
        return JSONResponse(
            status_code=409,
            content={"detail": f"unsafe persistence path: {exc}"},
        )

    @app.get("/api/health")
    def health():
        with state.workspace_lock:
            workspace = state._workspace
            workspace_root = (
                str(workspace.catalog.root) if workspace is not None else None
            )
        return {
            "ready": True,
            "workspace": workspace_root,
            "task": settings.task,
            "device": settings.device,
            "det_model": (
                str(settings.det_model_dir)
                if settings.det_model_dir
                else settings.det_model_name
            ),
            "rec_model": (
                str(settings.rec_model_dir)
                if settings.task == "ocr"
                else None
            ),
        }

    @app.post("/api/workspace/open")
    def open_workspace(request: OpenWorkspaceRequest):
        try:
            catalog = WorkspaceCatalog.open(Path(request.path))
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with state.workspace_lock:
            if state.batch.snapshot().state in {"queued", "running", "cancelling"}:
                raise HTTPException(status_code=409, detail="cancel the batch first")
            state._set_workspace_locked(catalog)
        return {"root": str(catalog.root), "images": len(catalog.list_images())}

    @app.get("/api/images")
    def list_images():
        catalog, store = state.require_workspace()
        images = []
        for record in catalog.list_images():
            saved = store.has_annotation(record)
            status = "error" if record.error else "not_ocr"
            if saved and not record.error:
                status = store.load(record).status
            images.append(
                {
                    "image_id": record.image_id,
                    "name": record.name,
                    "width": record.width,
                    "height": record.height,
                    "status": status,
                    "error": record.error,
                }
            )
        return {"images": images}

    @app.get("/api/images/{image_id}/content")
    def image_content(image_id: str):
        catalog, _ = state.require_workspace()
        try:
            record = catalog.get(image_id)
        except UnknownImageError:
            if any(item.image_id == image_id for item in catalog.list_images()):
                raise SourceImageChanged(image_id)
            raise
        except (OSError, ValueError) as exc:
            raise SourceImageChanged(image_id) from exc
        fd, content_length = _open_source_image(catalog, record)
        media_type = mimetypes.guess_type(record.name)[0] or "application/octet-stream"
        return OpenFileStreamingResponse(
            fd,
            content_length=content_length,
            media_type=media_type,
        )

    @app.get("/api/images/{image_id}/annotation")
    def get_annotation(image_id: str):
        catalog, store = state.require_workspace()
        return store.load(catalog.get(image_id))

    @app.put("/api/images/{image_id}/annotation")
    def save_annotation(image_id: str, annotation: Annotation):
        catalog, store = state.require_workspace()
        return store.save(catalog.get(image_id), annotation)

    @app.post("/api/images/{image_id}/ocr")
    @app.post("/api/images/{image_id}/detect")
    def ocr_image(image_id: str, request: OCRRequest):
        catalog, store = state.require_workspace()
        record = catalog.get(image_id)
        if record.error:
            raise HTTPException(status_code=422, detail=record.error)
        try:
            existing = store.load(record) if store.has_annotation(record) else None
        except ValidationError as exc:
            raise HTTPException(
                status_code=422, detail="existing annotation sidecar is invalid"
            ) from exc
        if existing is not None and not request.replace_existing and existing.blocks:
            raise HTTPException(
                status_code=409,
                detail="annotation already contains blocks; confirm replacement",
            )
        annotation = state.coordinator.recognize(record)
        if existing is not None:
            annotation = annotation.model_copy(update={"revision": existing.revision})
        return annotation

    @app.post("/api/batch")
    def start_batch():
        with state.workspace_lock:
            catalog, store = state._require_workspace_locked()
            try:
                return state.batch.start(catalog, store).to_dict()
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/batch")
    def get_batch():
        return state.batch.snapshot().to_dict()

    @app.post("/api/batch/cancel")
    def cancel_batch():
        return state.batch.cancel().to_dict()

    @app.post("/api/export")
    def export():
        catalog, store = state.require_workspace()
        manifest = store.export_manifest(catalog)
        records = sum(store.has_annotation(record) for record in catalog.list_images())
        if settings.task == "detection":
            labels = store.export_detection_labels(catalog)
            return {
                "path": str(labels),
                "manifest_path": str(manifest),
                "format": "paddleocr_detection",
                "records": records,
            }
        return {
            "path": str(manifest),
            "format": "jsonl",
            "records": records,
        }

    return app

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
import mimetypes
import os
import stat
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from .batch import BatchManager, GPUCoordinator
from .catalog import UnknownImageError, WorkspaceCatalog
from .layout_engine import LayoutDetectionEngine
from .models import Annotation, DetectRequest, ExportRequest, PrelabelRequest
from .storage import (
    AnnotationStore,
    ExportError,
    RevisionConflict,
    SourceImageChanged,
    UnsafePersistencePath,
)
from .task_map import PP_DOCLAYOUTV3_LABELS
from .vl_client import VLClient


class OpenWorkspaceRequest(BaseModel):
    path: str


@dataclass(frozen=True)
class Workspace:
    catalog: WorkspaceCatalog
    store: AnnotationStore


class OpenFileStreamingResponse(StreamingResponse):
    def __init__(self, fd: int, *, content_length: int, media_type: str):
        self.fd = fd
        super().__init__(self._iter(), media_type=media_type, headers={"Content-Length": str(content_length)})

    async def _iter(self):
        try:
            while True:
                chunk = os.read(self.fd, 1024 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1


def _open_image(catalog: WorkspaceCatalog, record):
    path = record.path.resolve(strict=True)
    if path.parent != catalog.root:
        raise SourceImageChanged(record.relative_path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    source_stat = os.fstat(descriptor)
    if not stat.S_ISREG(source_stat.st_mode):
        os.close(descriptor)
        raise SourceImageChanged(record.relative_path)
    digest = sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    if (
        source_stat.st_size != record.size_bytes
        or source_stat.st_mtime_ns != record.mtime_ns
        or digest.hexdigest() != record.sha256
    ):
        os.close(descriptor)
        raise SourceImageChanged(record.relative_path)
    return descriptor, source_stat.st_size


class AppState:
    def __init__(self, settings, layout_engine, vl_client):
        self.settings = settings
        self.coordinator = GPUCoordinator(layout_engine, vl_client)
        self.batch = BatchManager(self.coordinator)
        self._workspace: Workspace | None = None
        self.lock = Lock()

    def require_workspace(self) -> Workspace:
        with self.lock:
            if self._workspace is None:
                raise HTTPException(status_code=409, detail="open a workspace first")
            return self._workspace

    def set_workspace(self, catalog: WorkspaceCatalog) -> None:
        self._workspace = Workspace(
            catalog,
            AnnotationStore(catalog.root, self.settings.data_dir_name),
        )


def create_app(settings, layout_engine=None, vl_client=None, initial_workspace=None) -> FastAPI:
    owns_layout = layout_engine is None
    owns_vl = vl_client is None
    active_layout = layout_engine or LayoutDetectionEngine.create(settings)
    active_vl = vl_client or VLClient(settings)
    if owns_vl:
        active_vl.check_ready()
    state = AppState(settings, active_layout, active_vl)
    static_dir = Path(__file__).with_name("static")

    @asynccontextmanager
    async def lifespan(app):
        if initial_workspace is not None:
            catalog = WorkspaceCatalog.open(Path(initial_workspace))
            with state.lock:
                state.set_workspace(catalog)
        try:
            yield
        finally:
            await state.batch.shutdown()
            if owns_layout:
                active_layout.close()
            if owns_vl:
                active_vl.close()

    app = FastAPI(title="PaddleOCR-VL Layout Labeler", lifespan=lifespan)
    app.state.labeler = state
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
        return JSONResponse(status_code=409, content={"detail": f"source image changed: {exc}"})

    @app.exception_handler(UnsafePersistencePath)
    async def unsafe_path_handler(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ExportError)
    async def export_handler(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/api/health")
    def health():
        workspace = state._workspace
        return {
            "ready": True,
            "workspace": str(workspace.catalog.root) if workspace else None,
            "device": settings.device,
            "layout_model": str(settings.layout_model_dir),
            "vl_base_url": settings.vl_base_url,
            "vl_model": settings.vl_model,
        }

    @app.get("/api/taxonomy")
    def taxonomy():
        return {"layout_labels": PP_DOCLAYOUTV3_LABELS}

    @app.post("/api/workspace/open")
    def open_workspace(request: OpenWorkspaceRequest):
        try:
            catalog = WorkspaceCatalog.open(Path(request.path))
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if state.batch.snapshot().state in {"queued", "running", "cancelling"}:
            raise HTTPException(status_code=409, detail="cancel the batch first")
        with state.lock:
            state.set_workspace(catalog)
        return {"root": str(catalog.root), "images": len(catalog.list_images())}

    @app.get("/api/images")
    def list_images():
        workspace = state.require_workspace()
        images = []
        for record in workspace.catalog.list_images():
            status = "error" if record.error else "draft"
            if not record.error and workspace.store.has_annotation(record):
                status = workspace.store.load(record).status
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
        workspace = state.require_workspace()
        record = workspace.catalog.get(image_id)
        fd, size = _open_image(workspace.catalog, record)
        return OpenFileStreamingResponse(
            fd,
            content_length=size,
            media_type=mimetypes.guess_type(record.name)[0] or "application/octet-stream",
        )

    @app.get("/api/images/{image_id}/annotation")
    def get_annotation(image_id: str):
        workspace = state.require_workspace()
        return workspace.store.load(workspace.catalog.get(image_id))

    @app.put("/api/images/{image_id}/annotation")
    def save_annotation(image_id: str, payload: dict):
        workspace = state.require_workspace()
        record = workspace.catalog.get(image_id)
        current = workspace.store.load(record)
        try:
            annotation = Annotation.model_validate(payload)
        except ValidationError as original_error:
            if current.status != "completed" or payload.get("status") != "completed":
                raise HTTPException(
                    status_code=422, detail=str(original_error.errors()[0]["msg"])
                ) from original_error
            try:
                annotation = Annotation.model_validate({**payload, "status": "edited"})
            except ValidationError as edited_error:
                raise HTTPException(
                    status_code=422, detail=str(edited_error.errors()[0]["msg"])
                ) from edited_error
        if (
            current.status == "completed"
            and annotation.status == "completed"
            and (annotation.image != current.image or annotation.blocks != current.blocks)
        ):
            annotation = annotation.model_copy(update={"status": "edited"})
        return workspace.store.save(record, annotation)

    @app.post("/api/images/{image_id}/detect")
    def detect_image(image_id: str, request: DetectRequest):
        workspace = state.require_workspace()
        record = workspace.catalog.get(image_id)
        existing = workspace.store.load(record)
        if existing.blocks and not request.replace_existing:
            raise HTTPException(status_code=409, detail="annotation already contains blocks; confirm replacement")
        detected = state.coordinator.detect(record).model_copy(update={"revision": existing.revision})
        return workspace.store.save(record, detected)

    @app.post("/api/images/{image_id}/prelabel")
    def prelabel_image(image_id: str, request: PrelabelRequest):
        workspace = state.require_workspace()
        record = workspace.catalog.get(image_id)
        existing = workspace.store.load(record)
        if not existing.blocks:
            raise HTTPException(status_code=409, detail="detect layout before prelabeling")
        try:
            updated = state.coordinator.prelabel(
                record,
                existing,
                block_ids=request.block_ids,
                replace_existing=request.replace_existing,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return workspace.store.save(record, updated)

    @app.post("/api/images/{image_id}/complete")
    def complete_image(image_id: str):
        workspace = state.require_workspace()
        record = workspace.catalog.get(image_id)
        current = workspace.store.load(record)
        try:
            annotation = Annotation.model_validate(
                {**current.model_dump(mode="python"), "status": "completed"}
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc.errors()[0]["msg"]),
            ) from exc
        return workspace.store.save(record, annotation)

    @app.post("/api/batch/{operation}")
    def start_batch(operation: str):
        workspace = state.require_workspace()
        try:
            return state.batch.start(operation, workspace.catalog, workspace.store).to_dict()
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/batch")
    def batch_status():
        return state.batch.snapshot().to_dict()

    @app.delete("/api/batch")
    def cancel_batch():
        return state.batch.cancel().to_dict()

    @app.post("/api/export")
    def export_hf(request: ExportRequest):
        workspace = state.require_workspace()
        return workspace.store.export_hf(workspace.catalog, Path(request.output_dir))

    @app.post("/api/export/hf")
    def export_hf_explicit(request: ExportRequest):
        workspace = state.require_workspace()
        return workspace.store.export_hf(workspace.catalog, Path(request.output_dir))

    @app.post("/api/export/layout")
    def export_layout(request: ExportRequest):
        workspace = state.require_workspace()
        return workspace.store.export_layout(
            workspace.catalog, Path(request.output_dir)
        )

    @app.post("/api/export/all")
    def export_all(request: ExportRequest):
        workspace = state.require_workspace()
        return workspace.store.export_all(workspace.catalog, Path(request.output_dir))

    return app

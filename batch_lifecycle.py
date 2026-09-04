"""Shared lifecycle primitives for labeler batch operations."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from threading import Event, Lock, Thread
from typing import Any, Callable


class BatchLifecycleMixin:
    """Own batch state transitions while leaving record policy to subclasses."""

    def _init_lifecycle(
        self, snapshot_factory: Callable[..., Any], error_factory: Callable[..., Any]
    ) -> None:
        self._lifecycle_snapshot_factory = snapshot_factory
        self._lifecycle_error_factory = error_factory
        self._lifecycle_state_lock = Lock()
        self._lifecycle_cancel = Event()
        self._lifecycle_thread: Thread | None = None
        self._lifecycle_snapshot = snapshot_factory()

    @property
    def _state_lock(self):
        return self._lifecycle_state_lock

    @_state_lock.setter
    def _state_lock(self, value):
        self._lifecycle_state_lock = value

    @property
    def _cancel(self):
        return self._lifecycle_cancel

    @_cancel.setter
    def _cancel(self, value):
        self._lifecycle_cancel = value

    @property
    def _thread(self):
        return self._lifecycle_thread

    @_thread.setter
    def _thread(self, value):
        self._lifecycle_thread = value

    @property
    def _snapshot(self):
        return self._lifecycle_snapshot

    @_snapshot.setter
    def _snapshot(self, value):
        self._lifecycle_snapshot = value

    def _start_job(self, snapshot: Any, target, args: tuple[Any, ...]) -> None:
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("a batch job is already running")
            self._cancel.clear()
            self._snapshot = snapshot
            self._thread = Thread(target=target, args=args, daemon=True)
            self._thread.start()

    def snapshot(self):
        with self._state_lock:
            return deepcopy(self._snapshot)

    def cancel(self):
        with self._state_lock:
            if self._snapshot.state in {"queued", "running"}:
                self._snapshot.state = "cancelling"
                self._cancel.set()
            return deepcopy(self._snapshot)

    def wait(self, timeout=None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    async def shutdown(self) -> None:
        self.cancel()
        await asyncio.to_thread(self.wait)

    def _begin(self) -> bool:
        with self._state_lock:
            if self._cancel.is_set():
                self._snapshot.state = "cancelled"
                self._snapshot.current_image = None
                return False
            self._snapshot.state = "running"
            return True

    def _claim(self, image: str) -> bool:
        with self._state_lock:
            if self._cancel.is_set():
                self._snapshot.state = "cancelled"
                self._snapshot.current_image = None
                return False
            self._snapshot.current_image = image
            return True

    def _finish(self) -> None:
        with self._state_lock:
            self._snapshot.state = (
                "cancelled" if self._cancel.is_set() else "completed"
            )
            self._snapshot.current_image = None

    def _fail(self, exc: Exception) -> None:
        with self._state_lock:
            image = self._snapshot.current_image or "<batch>"
            self._snapshot.state = "failed"
            self._snapshot.current_image = None
            self._snapshot.failed += 1
            self._snapshot.errors.append(self._lifecycle_error_factory(image, str(exc)))

    def _increment(self, name: str) -> None:
        with self._state_lock:
            setattr(self._snapshot, name, getattr(self._snapshot, name) + 1)

    def _add_error(self, image: str, message: str) -> None:
        with self._state_lock:
            self._snapshot.failed += 1
            self._snapshot.errors.append(self._lifecycle_error_factory(image, message))

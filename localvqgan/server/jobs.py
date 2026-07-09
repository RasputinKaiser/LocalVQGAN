import asyncio
import io
import math
import threading
import time
from collections import deque
from dataclasses import fields
from pathlib import Path
from typing import Callable

from localvqgan.pipeline.animation import Keyframe, render_animation
from localvqgan.pipeline.backends import resolve_engine
from localvqgan.pipeline.frames import GenerationOOM
from localvqgan.pipeline.outputs import RunWriter
from localvqgan.pipeline.settings import GenerationSettings


class Busy(Exception):
    pass


_SETTINGS_FIELDS = {f.name for f in fields(GenerationSettings)}


def generation_settings_from_dict(settings_dict: dict) -> GenerationSettings:
    known = {k: v for k, v in settings_dict.items() if k in _SETTINGS_FIELDS}
    return GenerationSettings(**known)


class JobManager:
    def __init__(self, generator_factory: Callable[[], object], outputs_root: Path,
                 default_engine: str = "torch"):
        self._factory = generator_factory
        self._generator: object | None = None
        self.default_engine = default_engine
        self.outputs_root = Path(outputs_root)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._pending: deque = deque()
        self._subs: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.latest_preview: bytes | None = None
        self._state = {"state": "idle"}

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def generator(self):
        if self._generator is None:
            self._generator = self._factory()
        return self._generator

    def _generator_for(self, engine_name: str):
        if engine_name == self.default_engine:
            return self.generator  # existing warm default-engine generator
        attr = f"_{engine_name}_generator"
        if getattr(self, attr, None) is None:
            from localvqgan.pipeline.backends import make_generator
            setattr(self, attr, make_generator(engine_name))
        return getattr(self, attr)

    def status(self) -> dict:
        return {**self._state, "queued": len(self._pending)}

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def _publish(self, msg: dict) -> None:
        if msg.get("type") not in ("download", "queue"):
            self._state = {**self._state,
                           **{k: v for k, v in msg.items() if k != "image_jpeg"}}
        if self._loop is None:
            return

        def push():
            for q in list(self._subs):
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(msg)

        self._loop.call_soon_threadsafe(push)

    def _spawn_locked(self, kind: str, settings: GenerationSettings,
                      writer: RunWriter, kfs: list | None) -> None:
        self._cancel = threading.Event()
        if kind == "still":
            self._thread = threading.Thread(
                target=self._run_still, args=(settings, writer), daemon=True)
        else:
            self._thread = threading.Thread(
                target=self._run_animation, args=(settings, writer, kfs), daemon=True)
        self._thread.start()

    def _start_or_queue(self, kind: str, settings: GenerationSettings,
                        writer: RunWriter, kfs: list | None, queue: bool) -> str:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if not queue:
                    raise Busy()
                self._pending.append((kind, settings, writer, kfs))
                queued = len(self._pending)
            else:
                self._spawn_locked(kind, settings, writer, kfs)
                queued = None
        if queued is not None:
            self._publish({"type": "queue", "queued": queued})
        return writer.run_id

    def _start_next(self) -> None:
        with self._lock:
            if not self._pending:
                return
            kind, settings, writer, kfs = self._pending.popleft()
            self._spawn_locked(kind, settings, writer, kfs)
            queued = len(self._pending)
        self._publish({"type": "queue", "queued": queued})

    @property
    def queued(self) -> int:
        return len(self._pending)

    def start_still(self, settings_dict: dict, queue: bool = False) -> str:
        settings = generation_settings_from_dict(settings_dict)
        writer = RunWriter(self.outputs_root, settings)
        return self._start_or_queue("still", settings, writer, None, queue)

    def start_animation(self, settings_dict: dict, keyframes: list[dict],
                        queue: bool = False) -> str:
        settings = generation_settings_from_dict(settings_dict)
        writer = RunWriter(self.outputs_root, settings)
        kfs = [Keyframe(**k) for k in keyframes]
        return self._start_or_queue("animation", settings, writer, kfs, queue)

    def cancel(self) -> None:
        # Stop means stop everything: drop what's waiting, then cancel the
        # current job (skipping just the current run would surprise more).
        with self._lock:
            self._pending.clear()
        self._publish({"type": "queue", "queued": 0})
        self._cancel.set()

    def _preview_jpeg(self, img) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        self.latest_preview = buf.getvalue()
        return self.latest_preview

    def _run_still(self, settings: GenerationSettings, writer: RunWriter) -> None:
        try:
            self._publish({"state": "running", "run_id": writer.run_id,
                           "iteration": 0, "total": settings.iterations,
                           "phase": "loading"})
            engine_name, reason = resolve_engine(settings.engine, settings.checkpoint,
                                                 settings.clip_model,
                                                 width=settings.width, height=settings.height)
            gen = self._generator_for(engine_name)
            self._publish({"state": "running", "run_id": writer.run_id,
                           "engine": engine_name, "engine_reason": reason,
                           "phase": "loading"})
            gen.load(settings.checkpoint, settings.clip_model)
            t0, last_img = time.time(), None
            for u in gen.generate(settings, cancel=self._cancel):
                msg = {"state": "running", "run_id": writer.run_id,
                       "phase": "generating", "iteration": u.iteration,
                       "total": u.total,
                       "its_per_sec": round(u.iteration / max(time.time() - t0, 1e-6), 2)}
                if u.loss is not None and math.isfinite(u.loss):
                    msg["loss"] = u.loss
                if u.image is not None:
                    writer.save_frame(u.iteration, u.image)
                    last_img = u.image
                    msg["image_jpeg"] = self._preview_jpeg(u.image)
                self._publish(msg)
            if last_img is not None:
                writer.save_final(last_img)
            writer.write_sidecar({"engine_used": engine_name})
            self._publish({"state": "done", "run_id": writer.run_id})
        except GenerationOOM as e:
            self._publish({"state": "error", "error": str(e)})
        except Exception as e:  # surface, don't kill the server
            self._publish({"state": "error", "error": f"{type(e).__name__}: {e}"})
        finally:
            self._start_next()

    def _run_animation(self, settings: GenerationSettings, writer: RunWriter,
                       kfs: list) -> None:
        try:
            self._publish({"state": "running", "run_id": writer.run_id,
                           "iteration": 0, "total": sum(k.frames for k in kfs),
                           "phase": "loading"})
            engine_name, reason = resolve_engine(settings.engine, settings.checkpoint,
                                                 settings.clip_model,
                                                 width=settings.width, height=settings.height)
            gen = self._generator_for(engine_name)
            self._publish({"state": "running", "run_id": writer.run_id,
                           "engine": engine_name, "engine_reason": reason,
                           "phase": "loading"})
            gen.load(settings.checkpoint, settings.clip_model)

            def cb(done, total, img):
                msg = {"state": "running", "run_id": writer.run_id,
                       "phase": "animating", "iteration": done, "total": total}
                if img is not None:
                    msg["image_jpeg"] = self._preview_jpeg(img)
                self._publish(msg)

            render_animation(gen, settings, kfs, writer, self._cancel, cb,
                             extra={"engine_used": engine_name})
            self._publish({"state": "done", "run_id": writer.run_id})
        except GenerationOOM as e:
            self._publish({"state": "error", "error": str(e)})
        except Exception as e:
            self._publish({"state": "error", "error": f"{type(e).__name__}: {e}"})
        finally:
            self._start_next()

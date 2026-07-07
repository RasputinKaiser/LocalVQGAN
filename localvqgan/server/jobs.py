import asyncio
import io
import math
import threading
import time
from dataclasses import fields
from pathlib import Path
from typing import Callable

from localvqgan.pipeline.animation import Keyframe, render_animation
from localvqgan.pipeline.backends import resolve_engine
from localvqgan.pipeline.generator import GenerationOOM, Generator
from localvqgan.pipeline.outputs import RunWriter
from localvqgan.pipeline.settings import GenerationSettings


class Busy(Exception):
    pass


_SETTINGS_FIELDS = {f.name for f in fields(GenerationSettings)}


def generation_settings_from_dict(settings_dict: dict) -> GenerationSettings:
    known = {k: v for k, v in settings_dict.items() if k in _SETTINGS_FIELDS}
    return GenerationSettings(**known)


class JobManager:
    def __init__(self, generator_factory: Callable[[], Generator], outputs_root: Path):
        self._factory = generator_factory
        self._generator: Generator | None = None
        self.outputs_root = Path(outputs_root)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._subs: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.latest_preview: bytes | None = None
        self._state = {"state": "idle"}

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = self._factory()
        return self._generator

    def _generator_for(self, engine_name: str):
        if engine_name == "torch":
            return self.generator  # existing warm torch generator
        if getattr(self, "_mlx_generator", None) is None:
            from localvqgan.pipeline.backends import make_generator
            self._mlx_generator = make_generator("mlx")
        return self._mlx_generator

    def status(self) -> dict:
        return dict(self._state)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def _publish(self, msg: dict) -> None:
        if msg.get("type") != "download":
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

    def _begin(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise Busy()
            self._cancel = threading.Event()

    def start_still(self, settings_dict: dict) -> str:
        self._begin()
        settings = generation_settings_from_dict(settings_dict)
        writer = RunWriter(self.outputs_root, settings)
        self._thread = threading.Thread(
            target=self._run_still, args=(settings, writer), daemon=True)
        self._thread.start()
        return writer.run_id

    def start_animation(self, settings_dict: dict, keyframes: list[dict]) -> str:
        self._begin()
        settings = generation_settings_from_dict(settings_dict)
        writer = RunWriter(self.outputs_root, settings)
        kfs = [Keyframe(**k) for k in keyframes]
        self._thread = threading.Thread(
            target=self._run_animation, args=(settings, writer, kfs), daemon=True)
        self._thread.start()
        return writer.run_id

    def cancel(self) -> None:
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

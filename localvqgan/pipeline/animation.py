import threading
from dataclasses import dataclass
from typing import Callable

from PIL import Image

from localvqgan.pipeline.generator import Generator
from localvqgan.pipeline.outputs import RunWriter
from localvqgan.pipeline.settings import GenerationSettings


@dataclass
class Keyframe:
    prompts: str
    frames: int
    zoom: float = 1.0
    pan_x: int = 0
    pan_y: int = 0
    iterations_per_frame: int = 8


def transform_image(img: Image.Image, zoom: float, pan_x: int, pan_y: int) -> Image.Image:
    w, h = img.size
    cw, ch = int(w / zoom), int(h / zoom)
    left = (w - cw) // 2 + pan_x
    top = (h - ch) // 2 + pan_y
    left = max(0, min(w - cw, left))
    top = max(0, min(h - ch, top))
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)


def render_animation(g: Generator, base: GenerationSettings, keyframes: list[Keyframe],
                     writer: RunWriter, cancel: threading.Event | None,
                     progress_cb: Callable[[int, int, Image.Image | None], None],
                     extra: dict | None = None) -> None:
    total = sum(k.frames for k in keyframes)
    done = 0
    current: Image.Image | None = None
    if base.init_image:
        current = Image.open(base.init_image).convert("RGB")
    for kf in keyframes:
        for _ in range(kf.frames):
            if cancel is not None and cancel.is_set():
                return
            s = GenerationSettings(**{**base.to_dict(),
                                      "prompts": kf.prompts,
                                      "iterations": kf.iterations_per_frame,
                                      "display_freq": kf.iterations_per_frame,
                                      "seed": base.seed if done == 0 else -1,
                                      "init_image": None})
            if current is not None:
                s.init_image = _stash(writer, current)
            last = None
            for update in g.generate(s, cancel=cancel):
                last = update
            if last is None or last.image is None:
                return
            done += 1
            writer.save_frame(done, last.image)
            progress_cb(done, total, last.image)
            current = transform_image(last.image, kf.zoom, kf.pan_x, kf.pan_y)
    writer.save_final(current)
    writer.write_sidecar({"animation": [k.__dict__ for k in keyframes], **(extra or {})})


def _stash(writer: RunWriter, img: Image.Image) -> str:
    p = writer.dir / "_carry.png"
    img.save(p)
    return str(p)

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from localvqgan.pipeline.settings import GenerationSettings


class RunWriter:
    def __init__(self, outputs_root: Path, settings: GenerationSettings):
        outputs_root.mkdir(parents=True, exist_ok=True)
        existing = sorted(p.name for p in outputs_root.glob("run-*"))
        nxt = int(existing[-1].split("-")[1]) + 1 if existing else 1
        self.run_id = f"run-{nxt:04d}"
        self.dir = outputs_root / self.run_id
        (self.dir / "frames").mkdir(parents=True)
        self.settings = settings

    def save_frame(self, iteration: int, img: Image.Image) -> None:
        img.save(self.dir / "frames" / f"{iteration:05d}.png")

    def save_final(self, img: Image.Image) -> None:
        img.save(self.dir / "final.png")

    def write_sidecar(self, extra: dict | None = None) -> None:
        data = self.settings.to_dict() | (extra or {})
        (self.dir / "settings.json").write_text(json.dumps(data, indent=2))

    def make_timelapse(self, fps: int = 30) -> Path:
        out = self.dir / "timelapse.mp4"
        frames = sorted((self.dir / "frames").glob("*.png"))
        with imageio.get_writer(out, fps=fps, codec="libx264",
                                pixelformat="yuv420p") as w:
            for f in frames:
                w.append_data(np.asarray(Image.open(f)))
        return out


def list_runs(outputs_root: Path) -> list[dict]:
    runs = []
    for d in sorted(outputs_root.glob("run-*"), reverse=True):
        sidecar = d / "settings.json"
        runs.append({
            "run_id": d.name,
            "final": (d / "final.png").exists(),
            "timelapse": (d / "timelapse.mp4").exists(),
            "settings": json.loads(sidecar.read_text()) if sidecar.exists() else {},
        })
    return runs

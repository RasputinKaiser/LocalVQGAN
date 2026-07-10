"""Backend-neutral generation types.

Lives outside the torch backend so the MLX engine (and the server) can import
these without pulling in torch — the slim Apple-Silicon install has no torch.
"""
from dataclasses import dataclass

from PIL import Image


@dataclass
class FrameUpdate:
    iteration: int
    total: int
    image: Image.Image | None
    loss: float | None


class GenerationOOM(RuntimeError):
    pass

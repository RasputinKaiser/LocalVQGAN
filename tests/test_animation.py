from pathlib import Path

import pytest
import torch
from PIL import Image

from localvqgan.pipeline.animation import Keyframe, render_animation, transform_image
from localvqgan.pipeline.generator import Generator
from localvqgan.pipeline.outputs import RunWriter
from localvqgan.pipeline.settings import GenerationSettings

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_vqgan.yaml"


def test_transform_zoom_keeps_size():
    img = Image.new("RGB", (64, 64))
    out = transform_image(img, zoom=1.05, pan_x=2, pan_y=0)
    assert out.size == (64, 64)


@pytest.mark.slow
def test_animation_produces_frames(tmp_path):
    g = Generator(torch.device("cpu"))
    g.load_from_paths(FIXTURE, None, "ViT-B-32")
    s = GenerationSettings(width=64, height=64, cutouts=4, seed=7)
    w = RunWriter(tmp_path, s)
    kfs = [Keyframe(prompts="a forest", frames=2, zoom=1.02, iterations_per_frame=2),
           Keyframe(prompts="a city", frames=2, zoom=1.02, iterations_per_frame=2)]
    seen = []
    render_animation(g, s, kfs, w, cancel=None,
                     progress_cb=lambda done, total, img: seen.append((done, total)))
    assert len(list((w.dir / "frames").glob("*.png"))) == 4
    assert seen[-1] == (4, 4)

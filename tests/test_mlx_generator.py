import threading
from pathlib import Path

import pytest

pytest.importorskip("mlx")

from localvqgan.pipeline.settings import GenerationSettings

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_vqgan.yaml"


@pytest.mark.slow
def test_generates_and_is_self_reproducible():
    from localvqgan.pipeline.backends.mlx_backend.generator import MlxGenerator

    def run():
        g = MlxGenerator()
        g.load_from_paths(FIXTURE, None, "ViT-B-32")
        s = GenerationSettings(prompts="a red square", width=64, height=64,
                               iterations=3, cutouts=4, seed=42, display_freq=1)
        return list(g.generate(s))

    a = run()
    assert len(a) == 3 and a[-1].image is not None and a[-1].image.size == (64, 64)
    b = run()
    assert list(a[-1].image.getdata()) == list(b[-1].image.getdata())


@pytest.mark.slow
def test_cancel_stops_early():
    from localvqgan.pipeline.backends.mlx_backend.generator import MlxGenerator
    g = MlxGenerator()
    g.load_from_paths(FIXTURE, None, "ViT-B-32")
    cancel = threading.Event()
    s = GenerationSettings(prompts="x", width=64, height=64, iterations=100,
                           cutouts=4, seed=1, display_freq=1)
    seen = 0
    for _ in g.generate(s, cancel=cancel):
        seen += 1
        if seen == 2:
            cancel.set()
    assert seen < 100

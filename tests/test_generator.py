import threading
from pathlib import Path

import pytest
import torch

from localvqgan.pipeline.generator import Generator
from localvqgan.pipeline.settings import GenerationSettings

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_vqgan.yaml"


@pytest.mark.slow
def test_seed_reproducibility_cpu():
    def run():
        # the fixture model has random weights; pin them so only
        # generation determinism is under test (real ckpts are fixed)
        torch.manual_seed(0)
        g = Generator(torch.device("cpu"))
        g.load_from_paths(FIXTURE, None, "ViT-B-32")
        s = GenerationSettings(prompts="a red square", width=64, height=64,
                               iterations=3, cutouts=4, seed=42, display_freq=1)
        return list(g.generate(s))
    a, b = run(), run()
    assert len(a) == len(b) == 3
    assert list(a[-1].image.getdata()) == list(b[-1].image.getdata())


@pytest.mark.slow
def test_cancel_stops_early():
    g = Generator(torch.device("cpu"))
    g.load_from_paths(FIXTURE, None, "ViT-B-32")
    cancel = threading.Event()
    s = GenerationSettings(prompts="x", width=64, height=64, iterations=100,
                           cutouts=4, seed=1, display_freq=1)
    seen = 0
    for u in g.generate(s, cancel=cancel):
        seen += 1
        if seen == 2:
            cancel.set()
    assert seen < 100

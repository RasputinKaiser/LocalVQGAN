import pytest
import torch

from localvqgan.pipeline import checkpoints
from localvqgan.pipeline.generator import Generator
from localvqgan.pipeline.settings import GenerationSettings


@pytest.mark.slow
def test_end_to_end_small():
    torch.set_num_threads(2)  # keep the machine usable during real runs
    name = "imagenet_16384"
    if not checkpoints.is_downloaded(name):
        checkpoints.download(name, progress_cb=lambda *a: None)
    g = Generator()
    g.load(name, "ViT-B-32")
    s = GenerationSettings(prompts="a matte painting of a lighthouse at dusk",
                           width=128, height=128, iterations=5, cutouts=8,
                           seed=123, display_freq=5)
    frames = list(g.generate(s))
    assert frames[-1].image is not None
    assert frames[-1].image.size == (128, 128)

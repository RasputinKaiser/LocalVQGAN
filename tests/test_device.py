import torch

from localvqgan.pipeline.device import pick_device


def test_prefer_overrides():
    assert pick_device("cpu") == torch.device("cpu")


def test_auto_returns_device():
    assert pick_device().type in ("mps", "cuda", "cpu")

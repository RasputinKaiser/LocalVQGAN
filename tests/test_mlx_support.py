import pytest

pytest.importorskip("mlx")

from localvqgan.pipeline.backends.mlx_backend import supports


def test_supported_matrix():
    assert supports("imagenet_16384", "ViT-B-32")
    assert supports("wikiart_16384", "ViT-B-16")
    assert not supports("gumbel_8192", "ViT-B-32")
    assert not supports("imagenet_16384", "ViT-L-14")

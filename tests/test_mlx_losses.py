import numpy as np
import pytest

pytest.importorskip("mlx")
import mlx.core as mx
import torch

from localvqgan.pipeline.backends.mlx_backend.cutouts import make_cutouts
from localvqgan.pipeline.backends.mlx_backend.losses import (
    clamp_with_grad, prompt_loss, replace_grad, vector_quantize)
from localvqgan.pipeline.clip_guide import Prompt


def test_prompt_loss_matches_torch():
    e = np.random.RandomState(0).randn(8, 512).astype(np.float32)
    t = np.random.RandomState(1).randn(1, 512).astype(np.float32)
    for weight, stop in [(1.0, float("-inf")), (1.5, float("-inf")), (-1.0, float("-inf")), (2.0, -0.5)]:
        ref = float(Prompt(torch.tensor(t), weight, stop)(torch.tensor(e)))
        out = float(prompt_loss(mx.array(e), mx.array(t), weight, stop))
        assert abs(ref - out) < 1e-4, (weight, stop, ref, out)


def test_vector_quantize_snaps():
    codebook = mx.eye(4)
    x = mx.array([[[0.9, 0.1, 0.0, 0.0]]])
    q = vector_quantize(x, codebook)
    assert np.allclose(np.array(q), [[[1.0, 0.0, 0.0, 0.0]]])


def test_vector_quantize_straight_through_grad():
    codebook = mx.eye(4)
    def f(x):
        return vector_quantize(x, codebook).sum()
    g = mx.grad(f)(mx.array([[[0.9, 0.1, 0.0, 0.0]]]))
    assert np.allclose(np.array(g), 1.0)  # gradient passes straight through


def test_clamp_with_grad_forward():
    y = clamp_with_grad(mx.array([-1.0, 0.5, 2.0]), 0.0, 1.0)
    assert float(y.min()) >= 0 and float(y.max()) <= 1


def test_cutouts_shape_and_grad():
    mx.random.seed(0)
    img = mx.random.uniform(shape=(1, 96, 96, 3))
    out = make_cutouts(img, cutn=4, cut_size=64)
    assert out.shape == (4, 64, 64, 3)
    def f(x):
        return make_cutouts(x, cutn=4, cut_size=64).sum()
    g = mx.grad(f)(img)
    assert g.shape == img.shape

import threading

import numpy as np
import pytest

pytest.importorskip("mlx")
import mlx.core as mx
import torch

from localvqgan.pipeline.backends.mlx_backend.cutouts import (
    _resize_bilinear,
    _sharpness,
    _sharpness_kernel,
    make_cutouts,
    resize_identity_skip_enabled,
    sharpness_kernel_cache_enabled,
)
from localvqgan.pipeline.backends.mlx_backend.losses import (
    clamp_with_grad,
    prompt_loss,
    vector_quantize,
)
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


def test_resize_bilinear_upsample_matches_torch():
    rng = np.random.RandomState(2)
    img_np = rng.rand(1, 17, 23, 3).astype(np.float32)
    out = _resize_bilinear(mx.array(img_np), 32)
    ref = torch.nn.functional.interpolate(
        torch.tensor(img_np).permute(0, 3, 1, 2),
        size=(32, 32),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    out_np = np.array(out)
    assert np.max(np.abs(out_np - ref.numpy())) < 1e-5
    assert out_np.min() >= 0
    assert out_np.max() <= 1


def test_resize_bilinear_downsample_matches_torch():
    rng = np.random.RandomState(3)
    img_np = rng.rand(1, 96, 96, 3).astype(np.float32)
    out = _resize_bilinear(mx.array(img_np), 64)
    ref = torch.nn.functional.interpolate(
        torch.tensor(img_np).permute(0, 3, 1, 2),
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    assert np.max(np.abs(np.array(out) - ref.numpy())) < 1e-5


def test_resize_identity_skip_matches_value_and_grad(monkeypatch):
    rng = np.random.RandomState(7)
    img_np = rng.rand(1, 32, 32, 3).astype(np.float32)

    def run_with_flag(value: str):
        monkeypatch.setenv("LOCALVQGAN_MLX_RESIZE_IDENTITY_SKIP", value)
        return mx.value_and_grad(
            lambda img: _resize_bilinear(img, 32).sum()
        )(mx.array(img_np))

    baseline_out, baseline_grad = run_with_flag("0")
    candidate_out, candidate_grad = run_with_flag("1")
    mx.eval(baseline_out, baseline_grad, candidate_out, candidate_grad)

    assert np.array_equal(np.array(candidate_out), np.array(baseline_out))
    assert np.array_equal(np.array(candidate_grad), np.array(baseline_grad))


def test_resize_identity_skip_rejects_invalid_env(monkeypatch):
    monkeypatch.setenv("LOCALVQGAN_MLX_RESIZE_IDENTITY_SKIP", "maybe")

    with pytest.raises(ValueError, match="LOCALVQGAN_MLX_RESIZE_IDENTITY_SKIP"):
        resize_identity_skip_enabled()


def test_sharpness_preserves_border():
    batch = mx.ones((2, 8, 8, 3))
    factor = mx.array([1.3, 1.3])
    out = _sharpness(batch, factor)
    out_np = np.array(out)
    batch_np = np.array(batch)
    assert np.array_equal(out_np[:, 0, :, :], batch_np[:, 0, :, :])
    assert np.array_equal(out_np[:, -1, :, :], batch_np[:, -1, :, :])
    assert np.array_equal(out_np[:, :, 0, :], batch_np[:, :, 0, :])
    assert np.array_equal(out_np[:, :, -1, :], batch_np[:, :, -1, :])
    assert out_np.min() >= 0
    assert out_np.max() <= 1


def test_sharpness_kernel_cache_reuses_per_thread(monkeypatch):
    from localvqgan.pipeline.backends.mlx_backend import cutouts

    monkeypatch.delenv("LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE", raising=False)
    cutouts._thread_local.sharpness_kernels = {}

    first = _sharpness_kernel(3)
    second = _sharpness_kernel(3)

    assert sharpness_kernel_cache_enabled()
    assert first is second

    other_thread_kernels = []

    def run_in_thread():
        other_thread_kernels.append(_sharpness_kernel(3))

    t = threading.Thread(target=run_in_thread)
    t.start()
    t.join()

    assert other_thread_kernels[0] is not first
    assert np.array_equal(np.array(other_thread_kernels[0]), np.array(first))


def test_sharpness_kernel_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE", "0")

    first = _sharpness_kernel(3)
    second = _sharpness_kernel(3)

    assert not sharpness_kernel_cache_enabled()
    assert first is not second
    assert np.array_equal(np.array(first), np.array(second))


def test_sharpness_kernel_cache_rejects_invalid_env(monkeypatch):
    monkeypatch.setenv("LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE", "maybe")

    with pytest.raises(ValueError, match="LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE"):
        sharpness_kernel_cache_enabled()


def test_cutouts_shape_and_grad():
    mx.random.seed(0)
    img = mx.random.uniform(shape=(1, 96, 96, 3))
    out = make_cutouts(img, cutn=4, cut_size=64)
    assert out.shape == (4, 64, 64, 3)
    def f(x):
        return make_cutouts(x, cutn=4, cut_size=64).sum()
    g = mx.grad(f)(img)
    assert g.shape == img.shape


def test_clamp_with_grad_half_precision_grad():
    # Regression: fp32-hardcoded bounds fed to the custom vjp under dtype
    # promotion produced silent NaN gradients for fp16/bf16 inputs (fe3081f).
    ref = np.array(mx.grad(
        lambda t: clamp_with_grad(t, mx.array(0.0), mx.array(1.0)).sum()
    )(mx.array([-1.0, 0.25, 0.75, 2.0])))
    for dt in (mx.float16, mx.bfloat16):
        x = mx.array([-1.0, 0.25, 0.75, 2.0]).astype(dt)
        g = mx.grad(
            lambda t: clamp_with_grad(t, mx.array(0.0), mx.array(1.0)).sum()
        )(x)
        assert g.dtype == dt
        g_np = np.array(g.astype(mx.float32))
        assert np.all(np.isfinite(g_np)), dt
        assert np.allclose(g_np, ref), dt

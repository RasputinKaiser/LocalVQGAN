import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx")

from localvqgan.pipeline.settings import GenerationSettings

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_vqgan.yaml"


def _tiny_generator():
    from localvqgan.pipeline.backends.mlx_backend.generator import MlxGenerator

    g = MlxGenerator()
    g.load_from_paths(FIXTURE, None, "ViT-B-32")
    return g


def _tiny_settings(cutouts: int, iterations: int = 2):
    s = GenerationSettings(prompts="a red square", width=64, height=64,
                           iterations=iterations, cutouts=cutouts, seed=42,
                           display_freq=1)
    return s


def _run_tiny_generation(cutouts: int):
    g = _tiny_generator()
    s = _tiny_settings(cutouts)
    return list(g.generate(s))


def test_adam_matches_torch_bias_correction():
    import mlx.core as mx
    import mlx.optimizers as mo
    import numpy as np
    import torch
    from localvqgan.pipeline.backends.mlx_backend.generator import make_adam

    x0 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    grads = [np.array(g, dtype=np.float32) for g in
             ([0.5, -0.1, 0.2], [0.3, 0.3, -0.4], [-0.2, 0.1, 0.1])]

    t = torch.tensor(x0, requires_grad=True)
    topt = torch.optim.Adam([t], lr=0.1)
    for g in grads:
        topt.zero_grad(); t.grad = torch.tensor(g); topt.step()

    m = {"z": mx.array(x0)}
    mopt = make_adam(0.1)
    for g in grads:
        m = mopt.apply_gradients({"z": mx.array(g)}, m)
    assert np.allclose(np.array(m["z"]), t.detach().numpy(), atol=1e-5)


def test_chunked_cutout_threshold_gate():
    from localvqgan.pipeline.backends.mlx_backend.generator import (
        _is_large_canvas,
    )

    assert not _is_large_canvas(256, 256, 32)
    assert _is_large_canvas(512, 512, 32)


@pytest.mark.slow
@pytest.mark.parametrize("cutouts", [4, 5])
def test_chunked_cutouts_match_seeded_unchunked_generation(monkeypatch, cutouts):
    from localvqgan.pipeline.backends.mlx_backend import generator
    import mlx.core as mx

    monkeypatch.setenv("LOCALVQGAN_MLX_COMPILE", "0")
    monkeypatch.setattr(generator, "MLX_LARGE_CANVAS_PIXEL_THRESHOLD", 10**12)
    unchunked = _run_tiny_generation(cutouts)

    monkeypatch.setattr(generator, "MLX_LARGE_CANVAS_PIXEL_THRESHOLD", 0)
    monkeypatch.setattr(generator, "CUTOUT_CHUNK_SIZE", 2)
    # This isolates chunking math from the large-canvas bf16 precision policy.
    monkeypatch.setattr(
        generator.MlxGenerator,
        "_select_vqgan_dtype",
        lambda self, large_canvas, did_fp32_retry: self.vqgan.set_dtype(mx.float32),
    )
    chunked = _run_tiny_generation(cutouts)

    unchunked_losses = np.array([f.loss for f in unchunked], dtype=np.float32)
    chunked_losses = np.array([f.loss for f in chunked], dtype=np.float32)
    assert np.allclose(chunked_losses, unchunked_losses, atol=1e-5), (
        chunked_losses,
        unchunked_losses,
    )

    # The RNG recipe is identical, but chunking changes only float accumulation
    # order; one uint8 level allows harmless final-image quantization drift.
    chunked_img = np.asarray(chunked[-1].image, dtype=np.int16)
    unchunked_img = np.asarray(unchunked[-1].image, dtype=np.int16)
    assert np.max(np.abs(chunked_img - unchunked_img)) <= 1


@pytest.mark.slow
def test_generates_and_is_self_reproducible():
    def run():
        g = _tiny_generator()
        s = _tiny_settings(cutouts=4, iterations=3)
        return list(g.generate(s))

    a = run()
    assert len(a) == 3 and a[-1].image is not None and a[-1].image.size == (64, 64)
    b = run()
    assert list(a[-1].image.getdata()) == list(b[-1].image.getdata())


@pytest.mark.slow
def test_large_canvas_path_is_self_reproducible(monkeypatch):
    from localvqgan.pipeline.backends.mlx_backend import generator

    monkeypatch.setattr(generator, "MLX_LARGE_CANVAS_PIXEL_THRESHOLD", 0)
    monkeypatch.setattr(generator, "CUTOUT_CHUNK_SIZE", 2)

    def run():
        g = _tiny_generator()
        s = _tiny_settings(cutouts=4, iterations=3)
        return list(g.generate(s))

    a = run()
    assert len(a) == 3 and a[-1].image is not None and a[-1].image.size == (64, 64)
    b = run()
    assert list(a[-1].image.getdata()) == list(b[-1].image.getdata())


@pytest.mark.slow
def test_large_canvas_path_does_not_compile_chunk_closures(monkeypatch):
    from localvqgan.pipeline.backends.mlx_backend import generator

    compile_calls = 0
    real_compile = generator.mx.compile

    def counting_compile(fn):
        nonlocal compile_calls
        compile_calls += 1
        return real_compile(fn)

    monkeypatch.setattr(generator.mx, "compile", counting_compile)
    monkeypatch.setattr(generator, "MLX_LARGE_CANVAS_PIXEL_THRESHOLD", 0)
    monkeypatch.setattr(generator, "CUTOUT_CHUNK_SIZE", 2)

    g = _tiny_generator()
    s = _tiny_settings(cutouts=4, iterations=6)
    list(g.generate(s))

    assert compile_calls == 0


@pytest.mark.slow
def test_vqgan_dtype_follows_large_canvas_policy(monkeypatch):
    from localvqgan.pipeline.backends.mlx_backend import generator
    import mlx.core as mx

    small = _tiny_generator()
    list(small.generate(_tiny_settings(cutouts=2)))
    assert small.vqgan._dtype == mx.float32

    monkeypatch.setattr(generator, "MLX_LARGE_CANVAS_PIXEL_THRESHOLD", 0)
    large = _tiny_generator()
    list(large.generate(_tiny_settings(cutouts=2)))
    assert large.vqgan._dtype == mx.bfloat16


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


@pytest.mark.slow
def test_generate_works_across_threads():
    # The server runs each job in a fresh thread and MLX streams are
    # thread-local: any lazy param (e.g. set_dtype casts that mx.compile
    # traps in its trace) raises "There is no Stream(gpu, N) in current
    # thread" on the next job's thread unless materialized at load time.
    import mlx.core as mx
    from localvqgan.pipeline.backends.mlx_backend.generator import MlxGenerator

    g = MlxGenerator()
    g.load_from_paths(FIXTURE, None, "ViT-B-32")
    g.clip.set_dtype(mx.float16)  # the precision split load() ships
    s = GenerationSettings(prompts="a red square", width=64, height=64,
                           iterations=2, cutouts=2, seed=1, display_freq=1)
    errors = []

    def run():
        try:
            frames = list(g.generate(s))
            assert frames and frames[-1].image is not None
        except Exception as exc:  # noqa: BLE001 - recorded for the main thread
            errors.append(exc)

    for _ in range(2):
        t = threading.Thread(target=run)
        t.start()
        t.join()
    assert not errors, errors

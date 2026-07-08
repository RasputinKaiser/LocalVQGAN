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
    from localvqgan.pipeline.backends.mlx_backend.generator import _adam_update, make_adam

    x0 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    grads = [np.array(g, dtype=np.float32) for g in
             ([0.5, -0.1, 0.2], [0.3, 0.3, -0.4], [-0.2, 0.1, 0.1])]

    t = torch.tensor(x0, requires_grad=True)
    topt = torch.optim.Adam([t], lr=0.1)
    for g in grads:
        topt.zero_grad(); t.grad = torch.tensor(g); topt.step()

    m = {"z": mx.array(x0)}
    mopt = make_adam(0.1)
    fz = mx.array(x0)
    fm = mx.zeros_like(fz)
    fv = mx.zeros_like(fz)
    fstep = mx.array(0, dtype=mx.uint64)
    for g in grads:
        m = mopt.apply_gradients({"z": mx.array(g)}, m)
        fz, fm, fv, fstep = _adam_update(fz, mx.array(g), fm, fv, fstep, 0.1)
        mx.eval(fz, fm, fv, fstep)
    assert np.allclose(np.array(m["z"]), t.detach().numpy(), atol=1e-5)
    assert np.allclose(np.array(fz), t.detach().numpy(), atol=1e-5)
    assert np.allclose(np.array(fz), np.array(m["z"]), atol=1e-5)


def test_chunked_cutout_threshold_gate():
    from localvqgan.pipeline.backends.mlx_backend.generator import (
        _is_large_canvas,
    )

    assert not _is_large_canvas(256, 256, 32)
    assert _is_large_canvas(512, 512, 32)


def test_large_canvas_tuning_knobs_default(monkeypatch):
    from localvqgan.pipeline.backends.mlx_backend import generator

    monkeypatch.delenv("LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("LOCALVQGAN_MLX_CACHE_LIMIT_GB", raising=False)
    monkeypatch.delenv("LOCALVQGAN_MLX_ASYNC_CHUNKS", raising=False)
    monkeypatch.delenv("LOCALVQGAN_MLX_COMPILED_OPTIMIZER", raising=False)
    monkeypatch.delenv("LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE", raising=False)
    monkeypatch.delenv("LOCALVQGAN_MLX_SINGLE_LOSS_SUM_FASTPATH", raising=False)

    assert generator._cutout_chunk_size() == 8
    assert generator._large_canvas_cache_limit_bytes() == 4 * 1024**3
    assert generator._async_chunks_enabled()
    assert not generator._small_canvas_compiled_optimizer_enabled()
    assert generator._small_canvas_preview_reuse_enabled()
    assert generator._single_loss_sum_fastpath_enabled()


def test_large_canvas_tuning_knobs_honor_env(monkeypatch):
    from localvqgan.pipeline.backends.mlx_backend import generator

    monkeypatch.setenv("LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE", "16")
    monkeypatch.setenv("LOCALVQGAN_MLX_CACHE_LIMIT_GB", "6")
    monkeypatch.setenv("LOCALVQGAN_MLX_ASYNC_CHUNKS", "0")
    monkeypatch.setenv("LOCALVQGAN_MLX_COMPILED_OPTIMIZER", "0")
    monkeypatch.setenv("LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE", "1")
    monkeypatch.setenv("LOCALVQGAN_MLX_SINGLE_LOSS_SUM_FASTPATH", "0")

    assert generator._cutout_chunk_size() == 16
    assert generator._large_canvas_cache_limit_bytes() == 6 * 1024**3
    assert not generator._async_chunks_enabled()
    assert not generator._small_canvas_compiled_optimizer_enabled()
    assert generator._small_canvas_preview_reuse_enabled()
    assert not generator._single_loss_sum_fastpath_enabled()


def test_large_canvas_tuning_knobs_reject_invalid_env(monkeypatch):
    from localvqgan.pipeline.backends.mlx_backend import generator

    monkeypatch.setenv("LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE", "0")
    with pytest.raises(ValueError, match="LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE"):
        generator._cutout_chunk_size()

    monkeypatch.setenv("LOCALVQGAN_MLX_CACHE_LIMIT_GB", "-1")
    with pytest.raises(ValueError, match="LOCALVQGAN_MLX_CACHE_LIMIT_GB"):
        generator._large_canvas_cache_limit_bytes()

    monkeypatch.setenv("LOCALVQGAN_MLX_ASYNC_CHUNKS", "maybe")
    with pytest.raises(ValueError, match="LOCALVQGAN_MLX_ASYNC_CHUNKS"):
        generator._async_chunks_enabled()

    monkeypatch.setenv("LOCALVQGAN_MLX_COMPILED_OPTIMIZER", "maybe")
    with pytest.raises(ValueError, match="LOCALVQGAN_MLX_COMPILED_OPTIMIZER"):
        generator._small_canvas_compiled_optimizer_enabled()

    monkeypatch.setenv("LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE", "maybe")
    with pytest.raises(ValueError, match="LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE"):
        generator._small_canvas_preview_reuse_enabled()

    monkeypatch.setenv("LOCALVQGAN_MLX_SINGLE_LOSS_SUM_FASTPATH", "maybe")
    with pytest.raises(ValueError, match="LOCALVQGAN_MLX_SINGLE_LOSS_SUM_FASTPATH"):
        generator._single_loss_sum_fastpath_enabled()


def test_single_loss_sum_fastpath_matches_value_and_grad(monkeypatch):
    import mlx.core as mx
    from localvqgan.pipeline.backends.mlx_backend import generator

    values = mx.array([1.5, -2.0, 0.25])

    def run_with_flag(value: str):
        monkeypatch.setenv("LOCALVQGAN_MLX_SINGLE_LOSS_SUM_FASTPATH", value)
        return mx.value_and_grad(
            lambda x: generator._sum_losses([(x * x).mean()])
        )(values)

    baseline_out, baseline_grad = run_with_flag("0")
    candidate_out, candidate_grad = run_with_flag("1")
    mx.eval(baseline_out, baseline_grad, candidate_out, candidate_grad)

    assert np.array_equal(np.array(candidate_out), np.array(baseline_out))
    assert np.array_equal(np.array(candidate_grad), np.array(baseline_grad))


def test_multi_loss_sum_keeps_existing_order(monkeypatch):
    import mlx.core as mx
    from localvqgan.pipeline.backends.mlx_backend import generator

    monkeypatch.setenv("LOCALVQGAN_MLX_SINGLE_LOSS_SUM_FASTPATH", "1")
    values = mx.array([1.5, -2.0, 0.25])

    def current_order(x):
        return generator._sum_losses([(x * x).mean(), (x + 1).mean()])

    def previous_order(x):
        losses = [(x * x).mean(), (x + 1).mean()]
        return sum(losses, mx.array(0.0))

    candidate_out, candidate_grad = mx.value_and_grad(current_order)(values)
    baseline_out, baseline_grad = mx.value_and_grad(previous_order)(values)
    mx.eval(candidate_out, candidate_grad, baseline_out, baseline_grad)

    assert np.array_equal(np.array(candidate_out), np.array(baseline_out))
    assert np.array_equal(np.array(candidate_grad), np.array(baseline_grad))


def test_compile_step_state_is_per_wrapper(monkeypatch):
    import mlx.core as mx
    from localvqgan.pipeline.backends.mlx_backend import generator

    g = generator.MlxGenerator()

    def fallback_fn(z):
        return z + 1, z + 2

    def good_fn(z):
        return z + 3, z + 4

    def fake_compile(fn):
        def compiled(z):
            if fn is fallback_fn:
                raise RuntimeError("compiled path failed")
            return z + 30, z + 40

        return compiled

    monkeypatch.setattr(generator.mx, "compile", fake_compile)

    fallback_step = g._compile_step(fallback_fn)
    good_step = g._compile_step(good_fn)

    fallback_loss, fallback_grad = fallback_step(mx.array(1.0))
    good_loss, good_grad = good_step(mx.array(1.0))
    fallback_again_loss, fallback_again_grad = fallback_step(mx.array(1.0))
    mx.eval(fallback_loss, fallback_grad, good_loss, good_grad)
    mx.eval(fallback_again_loss, fallback_again_grad)

    assert float(fallback_loss.item()) == 2.0
    assert float(fallback_grad.item()) == 3.0
    assert float(good_loss.item()) == 31.0
    assert float(good_grad.item()) == 41.0
    assert float(fallback_again_loss.item()) == 2.0
    assert float(fallback_again_grad.item()) == 3.0


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
    assert np.array_equal(np.asarray(a[-1].image), np.asarray(b[-1].image))


@pytest.mark.slow
def test_small_canvas_compiled_optimizer_matches_previous_default(monkeypatch):
    monkeypatch.delenv("LOCALVQGAN_MLX_COMPILE", raising=False)
    monkeypatch.setenv("LOCALVQGAN_MLX_COMPILED_OPTIMIZER", "0")
    previous_default = _run_tiny_generation(cutouts=4)

    monkeypatch.delenv("LOCALVQGAN_MLX_COMPILED_OPTIMIZER", raising=False)
    folded = _run_tiny_generation(cutouts=4)

    previous_losses = np.array([f.loss for f in previous_default], dtype=np.float32)
    folded_losses = np.array([f.loss for f in folded], dtype=np.float32)
    assert np.allclose(folded_losses, previous_losses, atol=1e-5), (
        folded_losses,
        previous_losses,
    )
    assert np.array_equal(
        np.asarray(folded[-1].image),
        np.asarray(previous_default[-1].image),
    )


@pytest.mark.slow
def test_small_canvas_preview_reuse_matches_previous_default(monkeypatch):
    monkeypatch.delenv("LOCALVQGAN_MLX_COMPILE", raising=False)
    monkeypatch.setenv("LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE", "0")
    previous_default = _run_tiny_generation(cutouts=4)

    monkeypatch.setenv("LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE", "1")
    preview_reuse = _run_tiny_generation(cutouts=4)

    previous_losses = np.array([f.loss for f in previous_default], dtype=np.float32)
    reuse_losses = np.array([f.loss for f in preview_reuse], dtype=np.float32)
    assert np.array_equal(reuse_losses, previous_losses), (
        reuse_losses,
        previous_losses,
    )
    assert np.array_equal(
        np.asarray(preview_reuse[-1].image),
        np.asarray(previous_default[-1].image),
    )


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
    assert np.array_equal(np.asarray(a[-1].image), np.asarray(b[-1].image))


@pytest.mark.slow
def test_large_canvas_separate_compile_matches_eager(monkeypatch):
    from localvqgan.pipeline.backends.mlx_backend import generator
    import mlx.core as mx

    monkeypatch.setattr(generator, "MLX_LARGE_CANVAS_PIXEL_THRESHOLD", 0)
    monkeypatch.setattr(generator, "CUTOUT_CHUNK_SIZE", 2)
    monkeypatch.setattr(
        generator.MlxGenerator,
        "_select_vqgan_dtype",
        lambda self, large_canvas, did_fp32_retry: self.vqgan.set_dtype(mx.float32),
    )

    monkeypatch.setenv("LOCALVQGAN_MLX_COMPILE", "0")
    eager = _run_tiny_generation(cutouts=4)

    monkeypatch.delenv("LOCALVQGAN_MLX_COMPILE", raising=False)
    compiled = _run_tiny_generation(cutouts=4)

    eager_losses = np.array([f.loss for f in eager], dtype=np.float32)
    compiled_losses = np.array([f.loss for f in compiled], dtype=np.float32)
    assert np.allclose(compiled_losses, eager_losses, atol=1e-5), (
        compiled_losses,
        eager_losses,
    )
    assert np.array_equal(np.asarray(compiled[-1].image), np.asarray(eager[-1].image))


@pytest.mark.slow
def test_large_canvas_async_chunks_match_sync(monkeypatch):
    from localvqgan.pipeline.backends.mlx_backend import generator
    import mlx.core as mx

    monkeypatch.setenv("LOCALVQGAN_MLX_COMPILE", "0")
    monkeypatch.setattr(generator, "MLX_LARGE_CANVAS_PIXEL_THRESHOLD", 0)
    monkeypatch.setattr(generator, "CUTOUT_CHUNK_SIZE", 2)
    monkeypatch.setattr(
        generator.MlxGenerator,
        "_select_vqgan_dtype",
        lambda self, large_canvas, did_fp32_retry: self.vqgan.set_dtype(mx.float32),
    )

    monkeypatch.setenv("LOCALVQGAN_MLX_ASYNC_CHUNKS", "0")
    sync = _run_tiny_generation(cutouts=4)

    monkeypatch.delenv("LOCALVQGAN_MLX_ASYNC_CHUNKS", raising=False)
    async_run = _run_tiny_generation(cutouts=4)

    sync_losses = np.array([f.loss for f in sync], dtype=np.float32)
    async_losses = np.array([f.loss for f in async_run], dtype=np.float32)
    assert np.allclose(async_losses, sync_losses, atol=1e-5), (
        async_losses,
        sync_losses,
    )
    assert np.array_equal(np.asarray(async_run[-1].image), np.asarray(sync[-1].image))


@pytest.mark.slow
def test_large_canvas_async_chunks_are_scheduled(monkeypatch):
    from localvqgan.pipeline.backends.mlx_backend import generator
    import mlx.core as mx

    monkeypatch.setenv("LOCALVQGAN_MLX_COMPILE", "0")
    monkeypatch.delenv("LOCALVQGAN_MLX_ASYNC_CHUNKS", raising=False)
    monkeypatch.setattr(generator, "MLX_LARGE_CANVAS_PIXEL_THRESHOLD", 0)
    monkeypatch.setattr(generator, "CUTOUT_CHUNK_SIZE", 2)
    monkeypatch.setattr(
        generator.MlxGenerator,
        "_select_vqgan_dtype",
        lambda self, large_canvas, did_fp32_retry: self.vqgan.set_dtype(mx.float32),
    )

    async_eval_calls = 0
    real_async_eval = generator.mx.async_eval

    def counting_async_eval(*args):
        nonlocal async_eval_calls
        async_eval_calls += 1
        return real_async_eval(*args)

    monkeypatch.setattr(generator.mx, "async_eval", counting_async_eval)

    list(_tiny_generator().generate(_tiny_settings(cutouts=4, iterations=1)))

    assert async_eval_calls == 2


@pytest.mark.slow
def test_large_canvas_preview_reuses_loss_decode(monkeypatch):
    from localvqgan.pipeline.backends.mlx_backend import generator
    import mlx.core as mx

    monkeypatch.setenv("LOCALVQGAN_MLX_COMPILE", "0")
    monkeypatch.setattr(generator, "MLX_LARGE_CANVAS_PIXEL_THRESHOLD", 0)
    monkeypatch.setattr(generator, "CUTOUT_CHUNK_SIZE", 2)
    monkeypatch.setattr(
        generator.MlxGenerator,
        "_select_vqgan_dtype",
        lambda self, large_canvas, did_fp32_retry: self.vqgan.set_dtype(mx.float32),
    )

    g = _tiny_generator()
    original_synth = g._synth
    synth_calls = 0

    def counting_synth(z):
        nonlocal synth_calls
        synth_calls += 1
        if synth_calls > 2:
            raise AssertionError("large-canvas preview decoded a second time")
        return original_synth(z)

    monkeypatch.setattr(g, "_synth", counting_synth)
    frames = list(g.generate(_tiny_settings(cutouts=2, iterations=1)))

    assert synth_calls == 2
    assert frames[-1].image is not None


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

    # T4 compiles the fixed-shape synth and synth-pullback once per generation;
    # the per-chunk CLIP branch remains eager and must not compile per iteration.
    assert compile_calls == 2


@pytest.mark.slow
def test_large_canvas_cache_limit_is_scoped_to_attempt(monkeypatch):
    from localvqgan.pipeline.backends.mlx_backend import generator

    events = []

    def fake_set_cache_limit(limit):
        events.append(("set", limit))
        return "previous-limit"

    def fake_clear_cache():
        events.append(("clear", None))

    monkeypatch.setattr(generator.mx, "set_cache_limit", fake_set_cache_limit)
    monkeypatch.setattr(generator.mx, "clear_cache", fake_clear_cache)
    monkeypatch.setattr(generator, "MLX_LARGE_CANVAS_PIXEL_THRESHOLD", 0)
    monkeypatch.setenv("LOCALVQGAN_MLX_CACHE_LIMIT_GB", "6")

    g = _tiny_generator()
    s = _tiny_settings(cutouts=2, iterations=1)
    list(g.generate(s))

    assert events[0] == ("set", 6 * 1024**3)
    assert events[-2:] == [("set", "previous-limit"), ("clear", None)]


@pytest.mark.slow
def test_small_canvas_does_not_touch_cache_limit(monkeypatch):
    from localvqgan.pipeline.backends.mlx_backend import generator

    def fail_set_cache_limit(limit):
        raise AssertionError(f"unexpected cache limit call: {limit}")

    def fail_clear_cache():
        raise AssertionError("unexpected cache clear call")

    monkeypatch.setattr(generator.mx, "set_cache_limit", fail_set_cache_limit)
    monkeypatch.setattr(generator.mx, "clear_cache", fail_clear_cache)
    monkeypatch.setattr(generator, "MLX_LARGE_CANVAS_PIXEL_THRESHOLD", 10**12)

    g = _tiny_generator()
    s = _tiny_settings(cutouts=2, iterations=1)
    list(g.generate(s))


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

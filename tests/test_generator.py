import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from localvqgan.pipeline.generator import Generator
from localvqgan.pipeline.settings import GenerationSettings
from localvqgan.pipeline.vqgan import load_vqgan

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_vqgan.yaml"


def test_fast_mode_eligibility():
    g = Generator(torch.device("cpu"))
    g.vqgan = load_vqgan(FIXTURE, None, g.device)  # tiny fixture: f=2
    # 64² -> 32×32 tokens (even), coarse 16×16 >= 8 : eligible when flagged on
    assert g._fast_mode_eligible(GenerationSettings(width=64, height=64, fast_mode=True))
    assert not g._fast_mode_eligible(GenerationSettings(width=64, height=64))  # off
    # too small: 16² -> 8×8 tokens, coarse 4×4 < 8
    assert not g._fast_mode_eligible(GenerationSettings(width=16, height=16, fast_mode=True))
    assert g._fast_mode_eligible(
        GenerationSettings(width=512, height=512, cutouts=32, fast_mode=True))


@pytest.mark.slow
def test_fast_mode_runs_two_stages_and_is_finite():
    g = Generator(torch.device("cpu"))
    g.load_from_paths(FIXTURE, None, "ViT-B-32")
    s = GenerationSettings(prompts="a red square", width=64, height=64,
                           iterations=6, cutouts=4, seed=42, display_freq=1,
                           fast_mode=True)
    frames = list(g.generate(s))
    # 0.6*6 -> 4 coarse + 2 fine, global iteration numbers 1..6
    assert [f.iteration for f in frames] == [1, 2, 3, 4, 5, 6]
    assert all(f.total == 6 for f in frames)
    assert all(np.isfinite(f.loss) for f in frames)
    # two-stage really happened: coarse frames decode at 32², fine at the 64² target
    assert frames[0].image.size == (32, 32)
    assert frames[-1].image.size == (64, 64)
    final = np.asarray(frames[-1].image).astype(np.float32)
    assert final.std() > 5  # not a degenerate flat/NaN image


@pytest.mark.slow
def test_fast_mode_fine_stage_falls_back_to_fp32(monkeypatch):
    # On MPS fp16 the decoder overflows on upsampled latents (measured
    # 2026-07-09, imagenet_16384 256², seed 123): the fine stage's first loss
    # goes non-finite and the driver must redo just that stage with an fp32
    # VQGAN instead of dying or discarding the coarse work. CPU can't run the
    # fp16 path, so report fp16 to the driver's guard while the real models
    # stay fp32, and poison the first fine-stage synth to force the fallback.
    g = Generator(torch.device("cpu"))
    g.load_from_paths(FIXTURE, None, "ViT-B-32")
    monkeypatch.setattr(g, "_model_dtype", lambda precision: torch.float16)
    monkeypatch.setattr(g, "_set_model_dtype", lambda precision: None)
    dtype_calls = []
    original_set_dtype = g.vqgan.set_dtype
    monkeypatch.setattr(
        g.vqgan, "set_dtype",
        lambda dtype: (dtype_calls.append(dtype), original_set_dtype(dtype)))
    original_synth = g._synth
    poisoned = []

    def synth(z):
        out = original_synth(z)
        if z.shape[-1] == 32 and not poisoned:  # first fine-stage decode
            poisoned.append(True)
            return out * float("nan")
        return out

    monkeypatch.setattr(g, "_synth", synth)
    s = GenerationSettings(prompts="a red square", width=64, height=64,
                           iterations=6, cutouts=4, seed=42, display_freq=1,
                           fast_mode=True)
    frames = list(g.generate(s))
    assert poisoned  # the non-finite fine iteration actually happened
    assert torch.float32 in dtype_calls  # fallback switched the VQGAN to fp32
    # no duplicate or missing iterations despite the retried fine stage
    assert [f.iteration for f in frames] == [1, 2, 3, 4, 5, 6]
    assert all(np.isfinite(f.loss) for f in frames)


@pytest.mark.slow
def test_fast_mode_fine_stage_toggles_decoder_checkpointing(monkeypatch):
    # Put the checkpointing threshold between the 32² coarse stage and the 64²
    # fine stage so the stage boundary must flip it, without a real 512² run.
    from localvqgan.pipeline.backends import torch_backend
    monkeypatch.setattr(torch_backend, "TORCH_DECODE_CHECKPOINT_MIN_PIXELS", 32 * 32)
    g = Generator(torch.device("cpu"))
    g.load_from_paths(FIXTURE, None, "ViT-B-32")
    calls = []
    original = g.vqgan.set_decoder_checkpointing
    monkeypatch.setattr(g.vqgan, "set_decoder_checkpointing",
                        lambda enabled: (calls.append(enabled), original(enabled)))
    s = GenerationSettings(prompts="a red square", width=64, height=64,
                           iterations=6, cutouts=4, seed=42, display_freq=1,
                           fast_mode=True)
    frames = list(g.generate(s))
    assert calls == [False, True]  # coarse full-activation, fine checkpointed
    assert [f.iteration for f in frames] == [1, 2, 3, 4, 5, 6]
    assert all(np.isfinite(f.loss) for f in frames)


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
    assert np.array_equal(np.asarray(a[-1].image), np.asarray(b[-1].image))


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

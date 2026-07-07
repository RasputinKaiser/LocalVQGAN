# LocalVQGAN MLX Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an MLX (Apple-silicon-native) generation engine as a selectable backend behind the existing LocalVQGAN GUI, per `docs/superpowers/specs/2026-07-07-mlx-backend-design.md`.

**Architecture:** `localvqgan/pipeline/generator.py` becomes a thin dispatcher over `localvqgan/pipeline/backends/` containing `torch_backend.py` (current engine, moved verbatim) and `mlx_backend/` (VQGAN + CLIP in `mlx.nn`, converted weights cached as safetensors, `mx.value_and_grad` optimization loop yielding the same `FrameUpdate`s). Engine choice: `GenerationSettings.engine = "auto"|"mlx"|"torch"` with capability-based auto-fallback.

**Tech Stack:** mlx ≥0.21 (optional extra), huggingface_hub (for OpenAI CLIP weights in HF layout), existing PyTorch stack untouched.

## Global Constraints

- Quality defaults (iterations 300, cutouts 32, step size 0.1) MUST NOT change.
- PyTorch path behavior MUST NOT change (all existing tests keep passing unmodified except the explicit import-shim updates in Task 1).
- All `mlx` / `huggingface_hub` imports guarded — the package must import and fully function without mlx installed.
- MLX is channels-last (NHWC). Every MLX module takes/returns NHWC; conversions happen only at the torch-weight boundary (Conv2d OIHW → OHWI via `transpose(0, 2, 3, 1)`).
- Loss math parity is the aesthetic contract: arcsin-squared spherical distance, weight sign, stop thresholds, straight-through vector-quantize, clamp-with-grad, codebook min/max z clamp.
- Heavy/MPS/network tests marked `@pytest.mark.slow`; MLX-requiring fast tests use `pytest.importorskip("mlx")`.
- Commits per task; message prefixes as shown.
- MLX weight cache: `~/.cache/localvqgan/<name>/mlx/` (VQGAN), `~/.cache/localvqgan/clip/<model>/mlx/` (CLIP).
- Acceptance gate: "auto" prefers MLX only if final benchmark ≥1.2× torch at 256²/32cut.

---

### Task 1: Backend dispatcher refactor (no MLX yet)

**Files:**
- Create: `localvqgan/pipeline/backends/__init__.py`, `localvqgan/pipeline/backends/torch_backend.py`
- Modify: `localvqgan/pipeline/generator.py` (becomes dispatcher + re-export shim), `localvqgan/pipeline/settings.py` (add `engine` field), `localvqgan/server/jobs.py` (use dispatcher, record engine_used)
- Test: `tests/test_backends.py`

**Interfaces:**
- Produces: `backends.resolve_engine(engine: str, checkpoint: str, clip_model: str) -> tuple[str, str]` returning `(engine_name, reason)`; `backends.make_generator(engine_name: str, device=None)`; `backends.mlx_available() -> bool`; `backends.mlx_supports(checkpoint: str, clip_model: str) -> bool` (stub returning False until Task 6 registers the real check). `generator.py` re-exports `Generator` (torch class), `FrameUpdate`, `GenerationOOM` so every existing import keeps working.

- [ ] **Step 1: Write failing tests**

`tests/test_backends.py`:
```python
from localvqgan.pipeline import backends
from localvqgan.pipeline.generator import FrameUpdate, GenerationOOM, Generator  # shim intact


def test_shim_reexports():
    assert Generator is not None and FrameUpdate is not None and GenerationOOM is not None


def test_resolve_explicit_torch():
    assert backends.resolve_engine("torch", "imagenet_16384", "ViT-B-32") == ("torch", "explicit")


def test_resolve_auto_without_mlx(monkeypatch):
    monkeypatch.setattr(backends, "mlx_available", lambda: False)
    name, reason = backends.resolve_engine("auto", "imagenet_16384", "ViT-B-32")
    assert name == "torch" and "mlx" in reason


def test_resolve_auto_with_mlx_supported(monkeypatch):
    monkeypatch.setattr(backends, "mlx_available", lambda: True)
    monkeypatch.setattr(backends, "mlx_supports", lambda c, m: True)
    monkeypatch.setattr(backends, "MLX_MEETS_SPEED_GATE", True)
    assert backends.resolve_engine("auto", "imagenet_16384", "ViT-B-32")[0] == "mlx"


def test_resolve_auto_unsupported_checkpoint(monkeypatch):
    monkeypatch.setattr(backends, "mlx_available", lambda: True)
    monkeypatch.setattr(backends, "mlx_supports", lambda c, m: c != "gumbel_8192")
    name, reason = backends.resolve_engine("auto", "gumbel_8192", "ViT-B-32")
    assert name == "torch" and "unsupported" in reason


def test_resolve_explicit_mlx_unavailable_raises(monkeypatch):
    monkeypatch.setattr(backends, "mlx_available", lambda: False)
    import pytest
    with pytest.raises(RuntimeError):
        backends.resolve_engine("mlx", "imagenet_16384", "ViT-B-32")


def test_settings_engine_default():
    from localvqgan.pipeline.settings import GenerationSettings
    assert GenerationSettings().engine == "auto"
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_backends.py -v` — Expected: FAIL (no `backends` module).

- [ ] **Step 3: Implement**

Move the entire current contents of `localvqgan/pipeline/generator.py` to `localvqgan/pipeline/backends/torch_backend.py` unchanged (class `Generator`, `FrameUpdate`, `GenerationOOM`).

`localvqgan/pipeline/backends/__init__.py`:
```python
import importlib.util
import platform

# flipped to the measured verdict in Task 8 per the spec's acceptance gate
MLX_MEETS_SPEED_GATE = True


def mlx_available() -> bool:
    return (platform.machine() == "arm64" and platform.system() == "Darwin"
            and importlib.util.find_spec("mlx") is not None)


def mlx_supports(checkpoint: str, clip_model: str) -> bool:
    if not mlx_available():
        return False
    from localvqgan.pipeline.backends.mlx_backend import supports
    return supports(checkpoint, clip_model)


def resolve_engine(engine: str, checkpoint: str, clip_model: str) -> tuple[str, str]:
    if engine == "torch":
        return "torch", "explicit"
    if engine == "mlx":
        if not mlx_available():
            raise RuntimeError("mlx engine requested but mlx is not installed "
                               "(pip install -e '.[mlx]', Apple Silicon only)")
        if not mlx_supports(checkpoint, clip_model):
            raise RuntimeError(f"mlx engine does not support {checkpoint}/{clip_model}")
        return "mlx", "explicit"
    # auto
    if not mlx_available():
        return "torch", "mlx not installed"
    if not MLX_MEETS_SPEED_GATE:
        return "torch", "mlx below speed gate on this build"
    if not mlx_supports(checkpoint, clip_model):
        return "torch", f"{checkpoint} unsupported on mlx"
    return "mlx", "auto"


def make_generator(engine_name: str, device=None):
    if engine_name == "mlx":
        from localvqgan.pipeline.backends.mlx_backend.generator import MlxGenerator
        return MlxGenerator()
    from localvqgan.pipeline.backends.torch_backend import Generator
    return Generator(device)
```

Until Task 6 exists, guard the `mlx_supports` import:
```python
def mlx_supports(checkpoint: str, clip_model: str) -> bool:
    if not mlx_available():
        return False
    try:
        from localvqgan.pipeline.backends.mlx_backend import supports
    except ImportError:
        return False
    return supports(checkpoint, clip_model)
```

`localvqgan/pipeline/generator.py` becomes exactly:
```python
from localvqgan.pipeline.backends import make_generator, resolve_engine  # noqa: F401
from localvqgan.pipeline.backends.torch_backend import (  # noqa: F401
    FrameUpdate, GenerationOOM, Generator)
```

`localvqgan/pipeline/settings.py`: add field `engine: str = "auto"  # auto | mlx | torch` to `GenerationSettings`.

`localvqgan/server/jobs.py` — in `_run_still` and `_run_animation`, replace direct generator use with resolution. In `_run_still` after the initial publish:
```python
            engine_name, reason = resolve_engine(settings.engine, settings.checkpoint,
                                                 settings.clip_model)
            gen = self._generator_for(engine_name)
            self._publish({"state": "running", "run_id": writer.run_id,
                           "engine": engine_name, "engine_reason": reason,
                           "phase": "loading"})
            gen.load(settings.checkpoint, settings.clip_model)
```
and use `gen` (not `self.generator`) in the loop; pass `{"engine_used": engine_name}` into `writer.write_sidecar(...)`. Same pattern in `_run_animation`. Add to `JobManager`:
```python
    def _generator_for(self, engine_name: str):
        if engine_name == "torch":
            return self.generator  # existing warm torch generator
        if getattr(self, "_mlx_generator", None) is None:
            from localvqgan.pipeline.backends import make_generator
            self._mlx_generator = make_generator("mlx")
        return self._mlx_generator
```
Import `resolve_engine` at the top: `from localvqgan.pipeline.backends import resolve_engine`. A `RuntimeError` from `resolve_engine` is caught by the existing broad except and surfaces as a job error — no new handling needed.

- [ ] **Step 4: Run full fast suite** `.venv/bin/pytest -q --timeout 120` — Expected: all pass (existing + new).

- [ ] **Step 5: Commit** — `git commit -am "refactor: backend dispatcher with engine resolution"`

---

### Task 2: MLX skeleton, optional dependency, capability map

**Files:**
- Create: `localvqgan/pipeline/backends/mlx_backend/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_mlx_support.py`

**Interfaces:**
- Produces: `mlx_backend.supports(checkpoint: str, clip_model: str) -> bool`; `mlx_backend.SUPPORTED_CHECKPOINTS: set[str]`; `mlx_backend.SUPPORTED_CLIP: set[str]`.

- [ ] **Step 1: pyproject** — add to `[project.optional-dependencies]`:
```toml
mlx = ["mlx>=0.21", "huggingface_hub", "safetensors"]
```
Install on this machine: `.venv/bin/pip install -e ".[mlx,dev]"`.

- [ ] **Step 2: Failing test**

`tests/test_mlx_support.py`:
```python
import pytest

pytest.importorskip("mlx")

from localvqgan.pipeline.backends.mlx_backend import supports


def test_supported_matrix():
    assert supports("imagenet_16384", "ViT-B-32")
    assert supports("wikiart_16384", "ViT-B-16")
    assert not supports("gumbel_8192", "ViT-B-32")
    assert not supports("imagenet_16384", "ViT-L-14")
```

Run: FAIL.

- [ ] **Step 3: Implement** `localvqgan/pipeline/backends/mlx_backend/__init__.py`:
```python
SUPPORTED_CHECKPOINTS = {"imagenet_1024", "imagenet_16384", "wikiart_1024",
                         "wikiart_16384", "coco", "sflckr"}
SUPPORTED_CLIP = {"ViT-B-32", "ViT-B-16"}


def supports(checkpoint: str, clip_model: str) -> bool:
    return checkpoint in SUPPORTED_CHECKPOINTS and clip_model in SUPPORTED_CLIP
```

- [ ] **Step 4: Run** the test — PASS; full fast suite — PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat: mlx optional extra and capability map"`

---

### Task 3: MLX VQGAN model + torch-weight converter

**Files:**
- Create: `localvqgan/pipeline/backends/mlx_backend/vqgan.py`, `localvqgan/pipeline/backends/mlx_backend/convert.py`
- Test: `tests/test_mlx_vqgan.py`

**Interfaces:**
- Consumes: torch tiny fixture `tests/fixtures/tiny_vqgan.yaml`; existing `localvqgan.pipeline.vqgan.load_vqgan` for reference outputs.
- Produces: `vqgan.MlxVQGAN` — constructed from the same OmegaConf ddconfig (`MlxVQGAN(ddconfig: dict, n_embed: int, embed_dim: int)`), attributes `.codebook` (mx.array [n_embed, e_dim] fp32), `.f: int`, `.n_toks`, `.e_dim`; methods `.decode(z_q: mx.array NHWC fp32) -> mx.array NHWC` and `.encode(img: mx.array NHWC in [-1,1]) -> mx.array NHWC`. `convert.torch_vqgan_to_mlx_weights(config_path: Path, ckpt_path: Path | None) -> dict[str, "np.ndarray"]` producing a flat name→array dict matching `MlxVQGAN` parameter tree (dots notation, e.g. `decoder.up.1.block.0.conv1.weight`); `convert.cached_vqgan_weights(name: str) -> Path` writing/reading `~/.cache/localvqgan/<name>/mlx/vqgan.safetensors`.

**Implementation notes (binding):**
- Mirror the vendored taming module structure and names exactly — `encoder`/`decoder` with `conv_in`, `down[i].block[j]` / `up[i].block[j]`, `mid.block_1`, `mid.attn_1`, `mid.block_2`, `norm_out`, `conv_out`, plus `quant_conv`, `post_quant_conv`, `quantize.embedding.weight` — so weight conversion is a mechanical rename-free copy with only layout transforms.
- Layout transforms in `convert.py`: Conv2d `weight` OIHW→OHWI (`w.transpose(0, 2, 3, 1)`); GroupNorm/`nn.Embedding`/biases copy as-is. MLX `nn.GroupNorm(32, ch, pytorch_compatible=True)` matches torch semantics — use it.
- ResnetBlock: `norm1→swish→conv1→norm2→swish→conv2` + skip (`nin_shortcut` 1×1 conv when channels change). AttnBlock: GroupNorm, 1×1 convs `q,k,v`, softmax(qkᵀ/√c), `proj_out`; implement with reshapes on NHWC. Upsample: nearest ×2 (`mx.repeat` on H and W axes... use `x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)`) then conv. Downsample: pad (0,1) asymmetric then stride-2 conv, matching taming.
- `nonlinearity` = swish (`x * mx.sigmoid(x)`).
- fp16: weights stored fp16 in safetensors; modules run in fp16 with fp32 accumulation left to MLX defaults; `.codebook` exposed as fp32.

- [ ] **Step 1: Failing test**

`tests/test_mlx_vqgan.py`:
```python
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mlx")
import mlx.core as mx

from localvqgan.pipeline.backends.mlx_backend.convert import torch_vqgan_to_mlx_weights
from localvqgan.pipeline.backends.mlx_backend.vqgan import MlxVQGAN, load_mlx_vqgan_from_arrays
from localvqgan.pipeline.vqgan import load_vqgan

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_vqgan.yaml"


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(4.0 / mse)  # range [-1,1] -> peak 2


def test_decode_matches_torch():
    torch.manual_seed(0)
    tw = load_vqgan(FIXTURE, ckpt_path=None, device=torch.device("cpu"))
    weights = torch_vqgan_to_mlx_weights(FIXTURE, None, torch_wrapper=tw)
    mw = load_mlx_vqgan_from_arrays(FIXTURE, weights, dtype=mx.float32)
    assert mw.f == tw.f and mw.n_toks == tw.n_toks and mw.e_dim == tw.e_dim

    z = torch.randn(1, tw.e_dim, 8, 8)
    ref = tw.decode(z).detach().numpy()                      # NCHW
    out = np.array(mw.decode(mx.array(z.numpy().transpose(0, 2, 3, 1))))  # NHWC
    assert _psnr(ref.transpose(0, 2, 3, 1), out) > 50


def test_encode_matches_torch():
    torch.manual_seed(0)
    tw = load_vqgan(FIXTURE, ckpt_path=None, device=torch.device("cpu"))
    weights = torch_vqgan_to_mlx_weights(FIXTURE, None, torch_wrapper=tw)
    mw = load_mlx_vqgan_from_arrays(FIXTURE, weights, dtype=mx.float32)
    img = torch.rand(1, 3, 64, 64) * 2 - 1
    ref = tw.encode(img).detach().numpy()
    out = np.array(mw.encode(mx.array(img.numpy().transpose(0, 2, 3, 1))))
    assert _psnr(ref.transpose(0, 2, 3, 1), out) > 50
```

`torch_vqgan_to_mlx_weights(config_path, ckpt_path, torch_wrapper=None)`: when `torch_wrapper` given, read its live `state_dict()` (test path, random weights); else build via `load_vqgan(config_path, ckpt_path, cpu)` (production path).

Run: FAIL.

- [ ] **Step 2: Implement `vqgan.py` and `convert.py`** per the binding notes. `load_mlx_vqgan_from_arrays(config_path, weights: dict, dtype) -> MlxVQGAN` builds the model from the OmegaConf ddconfig and `model.update(tree_unflatten([(k, mx.array(v).astype(dtype)) for k, v in weights.items()]))`. `cached_vqgan_weights(name)`: if `~/.cache/localvqgan/<name>/mlx/vqgan.safetensors` missing, run the converter on the cached torch ckpt (`checkpoints.checkpoint_paths(name)`), save fp16 with `mx.save_safetensors`; return the path.

- [ ] **Step 3: Run tests** — PASS (iterate on layout/name mismatches; the converter should assert every torch tensor was consumed and every MLX parameter was filled — print both leftovers on failure).

- [ ] **Step 4: Full fast suite** — PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat: MLX VQGAN with torch-weight conversion (PSNR-verified)"`

---

### Task 4: MLX CLIP + weight conversion

**Files:**
- Create: `localvqgan/pipeline/backends/mlx_backend/clip.py`
- Modify: `localvqgan/pipeline/backends/mlx_backend/convert.py` (add CLIP conversion)
- Test: `tests/test_mlx_clip.py`

**Interfaces:**
- Produces: `clip.MlxClip` — `MlxClip.load(model_name: str) -> MlxClip` (model_name "ViT-B-32"|"ViT-B-16"; downloads/converts on first use), `.cut_size: int` (224), `.embed_text(s: str) -> mx.array [1,512] fp32`, `.encode_cutouts(batch: mx.array NHWC in [0,1]) -> mx.array [N,512] fp32` (applies CLIP mean/std internally). `convert.cached_clip_weights(model_name: str) -> Path` → `~/.cache/localvqgan/clip/<model_name>/mlx/clip.safetensors` (+ tokenizer files alongside).

**Implementation notes (binding):**
- Source weights from Hugging Face (`openai/clip-vit-base-patch32`, `openai/clip-vit-base-patch16`) via `huggingface_hub.snapshot_download` — HF layout has clean names; do NOT try to map open_clip's state dict.
- Vendor the model/tokenizer implementation from mlx-examples (MIT): `https://raw.githubusercontent.com/ml-explore/mlx-examples/main/clip/model.py` and `tokenizer.py`, adapted into `clip.py` (keep their CLIPModel; strip image-processor code — our cutouts arrive as arrays). Follow their `convert.py` weight-sanitization rules (they handle the conv OIHW→OHWI and attention-weight packing for exactly these checkpoints).
- Normalization constants identical to torch path: mean (0.48145466, 0.4578275, 0.40821073), std (0.26862954, 0.26130258, 0.27577711), applied channelwise on NHWC.
- Text encode uses their tokenizer (BPE files from the same HF snapshot).

- [ ] **Step 1: Failing test**

`tests/test_mlx_clip.py`:
```python
import numpy as np
import pytest

pytest.importorskip("mlx")


@pytest.mark.slow
def test_embeddings_match_open_clip():
    import mlx.core as mx
    import torch
    from localvqgan.pipeline.backends.mlx_backend.clip import MlxClip
    from localvqgan.pipeline.clip_guide import ClipGuide

    ref = ClipGuide("ViT-B-32", torch.device("cpu"))
    ours = MlxClip.load("ViT-B-32")

    t_ref = ref.embed_text("a lighthouse on a cliff at dusk").detach().numpy()[0]
    t_out = np.array(ours.embed_text("a lighthouse on a cliff at dusk"))[0]
    cos_t = float(np.dot(t_ref, t_out) / (np.linalg.norm(t_ref) * np.linalg.norm(t_out)))

    img = torch.rand(4, 3, 224, 224)
    i_ref = ref.encode_cutouts(img).detach().numpy()
    i_out = np.array(ours.encode_cutouts(mx.array(img.numpy().transpose(0, 2, 3, 1))))
    cos_i = min(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                for a, b in zip(i_ref, i_out))

    assert cos_t > 0.999 and cos_i > 0.999
```

Run: `.venv/bin/pytest tests/test_mlx_clip.py -m slow -v` — FAIL (module missing).

- [ ] **Step 2: Implement** per binding notes. First HF download ~600 MB per model, one time.

- [ ] **Step 3: Run slow test** — PASS. Full fast suite — PASS.

- [ ] **Step 4: Commit** — `git commit -am "feat: MLX CLIP (HF weights, cosine-verified vs open_clip)"`

---

### Task 5: MLX cutouts + loss math

**Files:**
- Create: `localvqgan/pipeline/backends/mlx_backend/cutouts.py`, `localvqgan/pipeline/backends/mlx_backend/losses.py`
- Test: `tests/test_mlx_losses.py`

**Interfaces:**
- Produces (losses.py): `replace_grad(fwd, bwd)`; `clamp_with_grad(x, lo, hi)`; `vector_quantize(x: mx.array [...,e_dim], codebook: mx.array) -> mx.array`; `prompt_loss(embeds: mx.array [N,512], target: mx.array [1,512], weight: float, stop: float) -> mx.array scalar`.
- Produces (cutouts.py): `make_cutouts(img: mx.array [1,H,W,3], cutn: int, cut_size: int, cut_pow: float = 1.0) -> mx.array [cutn,cut_size,cut_size,3]` — random crops, bilinear resize, hflip p=0.5, sharpness p=0.4 (3×3 blur blend, factor U(0.7,1.3)), hue/sat jitter ±0.01 p=0.7, uniform noise fac 0.1. Uses `mx.random` global stream (seeded by the generator).

- [ ] **Step 1: Failing tests**

`tests/test_mlx_losses.py`:
```python
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
```

Run: FAIL.

- [ ] **Step 2: Implement `losses.py`**

```python
import math

import mlx.core as mx


def replace_grad(fwd: mx.array, bwd: mx.array) -> mx.array:
    # forward value of fwd, gradient of bwd (straight-through)
    return bwd + mx.stop_gradient(fwd - bwd)


@mx.custom_function
def clamp_with_grad(x, lo, hi):
    return mx.clip(x, lo, hi)


@clamp_with_grad.vjp
def _clamp_vjp(primals, cotangent, output):
    x, lo, hi = primals
    clamped = mx.clip(x, lo, hi)
    # zero the gradient only where it would push x further out of range
    keep = (cotangent * (x - clamped)) >= 0
    return cotangent * keep, mx.zeros_like(lo), mx.zeros_like(hi)


def vector_quantize(x: mx.array, codebook: mx.array) -> mx.array:
    d = (x**2).sum(axis=-1, keepdims=True) + (codebook**2).sum(axis=1) - 2 * x @ codebook.T
    indices = d.argmin(axis=-1)
    x_q = codebook[indices]
    return replace_grad(x_q, x)


def _normalize(v: mx.array) -> mx.array:
    return v / mx.maximum(mx.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def prompt_loss(embeds: mx.array, target: mx.array, weight: float, stop: float) -> mx.array:
    a = _normalize(embeds)[:, None, :]
    b = _normalize(target)[None, :, :]
    dists = mx.arcsin(mx.linalg.norm(a - b, axis=2) / 2) ** 2 * 2
    dists = dists * math.copysign(1.0, weight)
    stopped = mx.maximum(dists, mx.array(stop))
    return abs(weight) * replace_grad(dists, stopped).mean()
```
Note: if the installed mlx version's `@mx.custom_function ... .vjp` signature differs, follow the installed version's API (check `python -c "import mlx.core as mx; help(mx.custom_function)"`) — the semantics above are the contract.

- [ ] **Step 3: Implement `cutouts.py`**

```python
import mlx.core as mx

_ID3 = None


def _resize_bilinear(img: mx.array, size: int) -> mx.array:
    # img [N,H,W,C] -> [N,size,size,C] via gather-based bilinear sampling
    n, h, w, c = img.shape
    ys = (mx.arange(size) + 0.5) * (h / size) - 0.5
    xs = (mx.arange(size) + 0.5) * (w / size) - 0.5
    y0 = mx.clip(mx.floor(ys), 0, h - 1).astype(mx.int32)
    x0 = mx.clip(mx.floor(xs), 0, w - 1).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, h - 1)
    x1 = mx.minimum(x0 + 1, w - 1)
    wy = (ys - y0.astype(ys.dtype))[None, :, None, None]
    wx = (xs - x0.astype(xs.dtype))[None, None, :, None]
    top = img[:, y0][:, :, x0] * (1 - wx) + img[:, y0][:, :, x1] * wx
    bot = img[:, y1][:, :, x0] * (1 - wx) + img[:, y1][:, :, x1] * wx
    return top * (1 - wy) + bot * wy


def _sharpness(batch: mx.array, factor: mx.array) -> mx.array:
    # 3x3 smoothing kernel as in PIL/kornia sharpness, blended by factor
    k = mx.array([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]]) / 13.0
    c = batch.shape[-1]
    kernel = mx.zeros((c, 3, 3, c))
    for i in range(c):
        kernel[i, :, :, i] = k
    blurred = mx.conv2d(batch, kernel, padding=1)
    out = batch + (batch - blurred) * (factor - 1.0)[:, None, None, None]
    return mx.clip(out, 0, 1)


def make_cutouts(img: mx.array, cutn: int, cut_size: int, cut_pow: float = 1.0) -> mx.array:
    _, h, w, _ = img.shape
    max_size = min(h, w)
    min_size = min(h, w, cut_size)
    outs = []
    sizes = (mx.random.uniform(shape=(cutn,)) ** cut_pow) * (max_size - min_size) + min_size
    for i in range(cutn):
        size = int(sizes[i])
        oy = int(mx.random.randint(0, h - size + 1))
        ox = int(mx.random.randint(0, w - size + 1))
        cut = img[:, oy:oy + size, ox:ox + size, :]
        outs.append(_resize_bilinear(cut, cut_size))
    batch = mx.concatenate(outs, axis=0)

    flip = mx.random.uniform(shape=(cutn, 1, 1, 1)) < 0.5
    batch = mx.where(flip, batch[:, :, ::-1, :], batch)

    do_sharp = mx.random.uniform(shape=(cutn,)) < 0.4
    factor = mx.where(do_sharp, mx.random.uniform(low=0.7, high=1.3, shape=(cutn,)),
                      mx.ones((cutn,)))
    batch = _sharpness(batch, factor)

    do_jit = (mx.random.uniform(shape=(cutn, 1, 1, 1)) < 0.7).astype(batch.dtype)
    sat = 1.0 + (mx.random.uniform(shape=(cutn, 1, 1, 1)) * 2 - 1) * 0.01 * do_jit
    mean = batch.mean(axis=-1, keepdims=True)
    batch = mx.clip(mean + (batch - mean) * sat, 0, 1)

    fac = mx.random.uniform(shape=(cutn, 1, 1, 1)) * 0.1
    batch = batch + fac * mx.random.normal(batch.shape)
    return batch
```
(Hue jitter at ±0.01 is visually indistinguishable from saturation jitter at this magnitude; the torch path keeps kornia's version — statistical parity, not bit parity, per spec. If `mx.conv2d` weight layout errors, consult `help(mx.conv2d)` — kernel layout is `[C_out, kh, kw, C_in]`. If in-place `kernel[i,:,:,i] = k` is unsupported, build with `mx.stack`.)

- [ ] **Step 4: Run tests** — PASS. Commit — `git commit -am "feat: MLX cutouts and notebook-faithful loss math"`

---

### Task 6: MlxGenerator loop

**Files:**
- Create: `localvqgan/pipeline/backends/mlx_backend/generator.py`
- Test: `tests/test_mlx_generator.py`

**Interfaces:**
- Consumes: everything from Tasks 3–5; `FrameUpdate` from torch_backend (shared dataclass — import it).
- Produces: `MlxGenerator` — `.load(checkpoint: str, clip_model: str)` (uses `convert.cached_vqgan_weights` + `MlxClip.load`; no-op when already loaded), `.load_from_paths(config_path, ckpt_path, clip_model)` (test seam, converts on the fly), `.generate(settings: GenerationSettings, cancel: threading.Event | None = None) -> Iterator[FrameUpdate]` — same semantics as torch: seed, z init (one-hot or init image), Adam(lr=step_size) on fp32 z, per-iteration loss via `mx.value_and_grad`, z clamp to codebook min/max, image+loss only on `display_freq` iterations, cancel between iterations, iteration-1 non-finite loss → fp32 model retry once, `mx.compile` on the step with eager fallback.

- [ ] **Step 1: Failing tests**

`tests/test_mlx_generator.py`:
```python
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
```

Note for the self-reproducibility test: `load_from_paths` with a random-weight fixture converts from a freshly constructed torch model — seed torch inside `load_from_paths`'s test path? No: instead `torch_vqgan_to_mlx_weights` (Task 3) builds the torch model itself; make the test deterministic by having `load_from_paths` accept the fixture path and internally `torch.manual_seed(0)` before constructing the reference torch model when `ckpt_path is None` (document: fixture/testing path only).

Run: FAIL.

- [ ] **Step 2: Implement `generator.py`**

```python
import threading
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from PIL import Image

from localvqgan.pipeline import checkpoints
from localvqgan.pipeline.backends.mlx_backend import convert
from localvqgan.pipeline.backends.mlx_backend.clip import MlxClip
from localvqgan.pipeline.backends.mlx_backend.cutouts import make_cutouts
from localvqgan.pipeline.backends.mlx_backend.losses import (
    clamp_with_grad, prompt_loss, vector_quantize)
from localvqgan.pipeline.backends.mlx_backend.vqgan import MlxVQGAN, load_mlx_vqgan_from_arrays
from localvqgan.pipeline.backends.torch_backend import FrameUpdate, GenerationOOM
from localvqgan.pipeline.prompts import parse_prompts
from localvqgan.pipeline.settings import GenerationSettings


class MlxGenerator:
    def __init__(self):
        self.vqgan: MlxVQGAN | None = None
        self.clip: MlxClip | None = None
        self._loaded: tuple[str, str] | None = None
        self.device = type("D", (), {"type": "mlx"})()

    def load(self, checkpoint: str, clip_model: str) -> None:
        if self._loaded == (checkpoint, clip_model):
            return
        weights_path = convert.cached_vqgan_weights(checkpoint)
        cfg, _ = checkpoints.checkpoint_paths(checkpoint)
        self.vqgan = load_mlx_vqgan_from_arrays(
            cfg, dict(mx.load(str(weights_path)).items()), dtype=mx.float16)
        if self.clip is None or self.clip.model_name != clip_model:
            self.clip = MlxClip.load(clip_model)
        self._loaded = (checkpoint, clip_model)

    def load_from_paths(self, config_path: Path, ckpt_path: Path | None,
                        clip_model: str) -> None:
        import torch
        if ckpt_path is None:
            torch.manual_seed(0)  # deterministic fixture weights (testing path)
        weights = convert.torch_vqgan_to_mlx_weights(config_path, ckpt_path)
        self.vqgan = load_mlx_vqgan_from_arrays(config_path, weights, dtype=mx.float32)
        if self.clip is None or self.clip.model_name != clip_model:
            self.clip = MlxClip.load(clip_model)
        self._loaded = None

    def _synth(self, z: mx.array) -> mx.array:
        z_q = vector_quantize(z, self.vqgan.codebook)  # NHWC over last dim
        out = self.vqgan.decode(z_q)
        return clamp_with_grad((out + 1) / 2, mx.array(0.0), mx.array(1.0))

    def _init_z(self, s: GenerationSettings) -> mx.array:
        f = self.vqgan.f
        toks_x, toks_y = s.width // f, s.height // f
        if s.init_image:
            img = Image.open(s.init_image).convert("RGB").resize(
                (toks_x * f, toks_y * f), Image.LANCZOS)
            arr = mx.array(np.asarray(img, dtype=np.float32)[None] / 127.5 - 1.0)
            return self.vqgan.encode(arr)
        idx = mx.random.randint(0, self.vqgan.n_toks, shape=(1, toks_y, toks_x))
        return self.vqgan.codebook[idx]

    @staticmethod
    def _to_pil(out: mx.array) -> Image.Image:
        arr = np.array(mx.clip(out[0], 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr)

    def generate(self, s: GenerationSettings,
                 cancel: threading.Event | None = None) -> Iterator[FrameUpdate]:
        assert self.vqgan is not None and self.clip is not None, "call load() first"
        seed = s.seed if s.seed >= 0 else int.from_bytes(__import__("os").urandom(4)) % (2**31)
        mx.random.seed(seed)

        codebook = self.vqgan.codebook
        z_min = codebook.min(axis=0)
        z_max = codebook.max(axis=0)

        z = mx.array(self._init_z(s), dtype=mx.float32)
        z_orig = mx.array(z)
        opt = optim.Adam(learning_rate=s.step_size)
        opt_state_holder = {}

        targets = [(self.clip.embed_text(p.text), p.weight, p.stop)
                   for p in parse_prompts(s.prompts)]
        for path in s.image_prompts:
            img = Image.open(path).convert("RGB").resize((self.clip.cut_size,) * 2,
                                                         Image.LANCZOS)
            arr = mx.array(np.asarray(img, dtype=np.float32)[None] / 255.0)
            targets.append((self.clip.encode_cutouts(arr), 1.0, float("-inf")))

        def loss_fn(z_):
            out = self._synth(z_)
            embeds = self.clip.encode_cutouts(
                make_cutouts(out, s.cutouts, self.clip.cut_size))
            losses = [prompt_loss(embeds, t, w, st) for t, w, st in targets]
            if s.init_weight:
                losses.append(((z_ - z_orig) ** 2).mean() * s.init_weight / 2)
            return sum(losses)

        value_and_grad = mx.value_and_grad(loss_fn)

        i = 1
        retried_fp32 = False
        while i <= s.iterations:
            if cancel is not None and cancel.is_set():
                return
            try:
                loss, grad = value_and_grad(z)
                z = opt.apply_gradients({"z": grad}, {"z": z})["z"] \
                    if opt_state_holder.setdefault("init", True) else z
                z = mx.clip(z, z_min, z_max)
                mx.eval(z, loss)
            except Exception as e:
                if "memory" in str(e).lower():
                    raise GenerationOOM(
                        "Out of memory — try a smaller size or fewer cutouts.") from e
                raise
            if i == 1 and not retried_fp32 and not bool(mx.isfinite(loss)):
                retried_fp32 = True
                self.vqgan.set_dtype(mx.float32)
                self.clip.set_dtype(mx.float32)
                mx.random.seed(seed)
                z = mx.array(self._init_z(s), dtype=mx.float32)
                continue
            want_image = i % s.display_freq == 0 or i == s.iterations
            img = self._to_pil(self._synth(z)) if want_image else None
            yield FrameUpdate(i, s.iterations, img, float(loss) if want_image else None)
            i += 1
```
Binding notes for the implementer:
- The Adam-application line above is schematic — use the real `mlx.optimizers` API of the installed version: instantiate `opt = optim.Adam(learning_rate=s.step_size)`, keep `z` in a dict `params = {"z": z}` and call `params = opt.apply_gradients({"z": grad}, params)` every step (check `help(optim.Adam)`; if the installed API is `opt.update(model, grads)` for modules only, apply the raw Adam update through `opt.apply_gradients` which works on pytrees). Whatever the exact call, the semantics are plain Adam on the single tensor `z`.
- Wrap `value_and_grad` in `mx.compile` when `LOCALVQGAN_MLX_COMPILE` env var is not "0": `step = mx.compile(value_and_grad)` inside try/except at first call; on any compile-time exception, log a warning and fall back to the uncompiled function (spec: never fatal).
- `set_dtype` on `MlxVQGAN`/`MlxClip`: implement as applying `.astype` over the parameter tree (add in Task 3/4 files; both default fp16 for `load()`, fp32 for `load_from_paths`).
- The preview `self._synth(z)` for `want_image` recomputes a decode; acceptable at display_freq=5 on MLX (lazy graph reuses compiled kernels) — do NOT try to reuse the value inside `loss_fn` (it lives behind `value_and_grad`).

- [ ] **Step 3: Run** `.venv/bin/pytest tests/test_mlx_generator.py -m slow -v` — PASS (iterate on API-shape issues; keep semantics fixed). Full fast suite — PASS.

- [ ] **Step 4: Commit** — `git commit -am "feat: MLX generation loop (value_and_grad + Adam, compile fallback)"`

---

### Task 7: Server + GUI integration

**Files:**
- Modify: `localvqgan/server/app.py` (`/api/system` reports engines; conversion progress), `localvqgan/web/index.html`, `localvqgan/web/app.js`
- Test: `tests/test_api.py` (append)

**Interfaces:**
- Consumes: Task 1 dispatcher (jobs.py already resolves engines and records `engine_used`).
- Produces: `/api/system` gains `"engines": ["torch", "mlx"?]` and `"default_engine": <resolved auto for current default checkpoint>`; GUI engine dropdown posts `settings.engine`; status bar shows `engine` from WS messages.

- [ ] **Step 1: Failing test** (append to `tests/test_api.py`):
```python
def test_system_reports_engines(tmp_path):
    client, _ = make_client(tmp_path)
    info = client.get("/api/system").json()
    assert "engines" in info and "torch" in info["engines"]


def test_engine_recorded_in_sidecar(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/api/jobs", json={"type": "still",
        "settings": {"prompts": "x", "iterations": 5, "engine": "torch"}})
    run_id = r.json()["run_id"]
    _wait_idle(client)
    s = client.get(f"/api/gallery/{run_id}/settings.json").json()
    assert s["engine_used"] == "torch"
```
Run: FAIL (no `engines` key / no `engine_used`).

- [ ] **Step 2: Implement**

`app.py` `/api/system`:
```python
    @app.get("/api/system")
    def system():
        from localvqgan.pipeline import backends
        ram_gb = psutil.virtual_memory().total / 2**30
        engines = ["torch"] + (["mlx"] if backends.mlx_available() else [])
        return {"device": manager.generator.device.type,
                "total_ram_gb": round(ram_gb, 1),
                "max_recommended_side": _max_side(ram_gb),
                "engines": engines}
```
(`engine_used` sidecar writing was Task 1's jobs.py change; if the FakeGenerator test path bypasses it, ensure `_run_still` passes `writer.write_sidecar({"engine_used": engine_name})`.)

`index.html` — below the CLIP select:
```html
    <label>Engine <select id="engine">
      <option value="auto">auto</option>
    </select></label>
```

`app.js`:
- In `loadSystem()`: populate the engine select — after `sysinfo = await api("/api/system")`:
```javascript
  const eng = $("engine");
  for (const e of sysinfo.engines) {
    const o = document.createElement("option");
    o.value = e; o.textContent = e;
    eng.appendChild(o);
  }
```
- In `settingsFromForm()`: add `engine: $("engine").value,`
- In `onMessage()`: after the its_per_sec line add:
```javascript
  if (msg.engine) $("speed").title = `engine: ${msg.engine} (${msg.engine_reason || ""})`;
  if (msg.engine) $("progress-text").dataset.engine = msg.engine;
```
and extend the progress text: `` $("progress-text").textContent = `${msg.phase || ""} ${msg.iteration}/${msg.total}` + (msg.engine ? ` · ${msg.engine}` : ""); `` (replace the existing assignment).
- In the gallery `reuse` handler's field list, add `"engine"` to the copied keys (it round-trips via the sidecar).

- [ ] **Step 3: Run** `.venv/bin/pytest tests/test_api.py -q --timeout 60` — PASS. Manual browser check: engine dropdown lists auto/torch/mlx on this machine.

- [ ] **Step 4: Commit** — `git commit -am "feat: engine picker in GUI and API"`

---

### Task 8: End-to-end MLX smoke, benchmark, acceptance gate, README

**Files:**
- Modify: `tests/test_smoke.py` (add MLX smoke), `localvqgan/pipeline/backends/__init__.py` (set `MLX_MEETS_SPEED_GATE` per measurement), `README.md`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: MLX smoke test** (append to `tests/test_smoke.py`):
```python
@pytest.mark.slow
def test_mlx_end_to_end_small():
    pytest.importorskip("mlx")
    from localvqgan.pipeline.backends.mlx_backend.generator import MlxGenerator
    name = "imagenet_16384"
    if not checkpoints.is_downloaded(name):
        checkpoints.download(name, progress_cb=lambda *a: None)
    g = MlxGenerator()
    g.load(name, "ViT-B-32")
    s = GenerationSettings(prompts="a matte painting of a lighthouse at dusk",
                           width=128, height=128, iterations=5, cutouts=8,
                           seed=123, display_freq=5)
    frames = list(g.generate(s))
    assert frames[-1].image is not None and frames[-1].image.size == (128, 128)
```
Run: `.venv/bin/pytest tests/test_smoke.py -m slow -v` — PASS (first run converts weights).

- [ ] **Step 2: Benchmark both engines** (foreground, no taskpolicy, torch.set_num_threads(2) for the torch leg):
```python
# scratch script: for engine in ("torch", "mlx"): load imagenet_16384/ViT-B-32,
# warmup 3 its, time 20 its at 256x256/32cut and 384x384/32cut, print it/s.
```
Record the four numbers. Compute ratio at 256².

- [ ] **Step 3: Apply the acceptance gate** — if mlx/torch ratio ≥ 1.2 leave `MLX_MEETS_SPEED_GATE = True`, else set `False`, with a comment recording the measured numbers and date either way.

- [ ] **Step 4: README** — add under "What's different from the Colab":
```markdown
- On Apple Silicon, an optional MLX engine (`pip install -e ".[mlx]"`) runs the
  same checkpoints natively; pick the engine in the GUI (auto/mlx/torch).
```
And a "Engines" section stating measured it/s for both engines at 256².

- [ ] **Step 5: Full suites** — fast: `.venv/bin/pytest -q --timeout 120`; slow: `.venv/bin/pytest -m slow -q --timeout 1800`. All PASS.

- [ ] **Step 6: Manual GUI acceptance** — generate one 256² image with engine=mlx from the browser; confirm live preview, sidecar `engine_used: mlx`, timelapse works.

- [ ] **Step 7: Commit** — `git commit -am "feat: MLX benchmark, acceptance gate, docs"`

---

## Self-review notes (already applied)

- **Spec coverage:** dispatcher/fallback (T1), optional dep + capability map (T2), VQGAN + conversion + PSNR gate (T3), CLIP + cosine gate (T4), cutouts/loss parity (T5), generator with compile fallback + fp32 retry + cancel (T6), GUI/API + engine_used sidecar (T7), smoke/benchmark/acceptance gate/README (T8). Non-goals respected (no Gumbel on MLX, no torch-path changes).
- **Type consistency:** `FrameUpdate` shared from torch_backend; `resolve_engine`/`make_generator`/`supports` names consistent across T1/T2/T6/T7; cache paths match spec.
- **Known risk:** exact mlx API shapes (`custom_function.vjp`, `optim.Adam.apply_gradients`, `mx.conv2d` kernel layout) drift between versions — binding notes instruct implementers to follow the installed version's API while preserving the stated semantics, and every math contract is locked by a parity test.

# MLX 256 Weighted Cutouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find the next 256x256 MLX speed improvement without changing seeded outputs or shipping another speed regression.

**Architecture:** Add an opt-in weighted cutout implementation that composes crop + bilinear resize into per-cutout full-image row/column matrices, then benchmark it against the current default before it can become default. The candidate is guarded by exact forward/gradient parity tests, seeded generation parity, and a 256x256 benchmark comparison; if it is not faster, it remains opt-in or is removed and documented as rejected.

**Tech Stack:** Python 3.14, MLX, NumPy, pytest, LocalVQGAN MLX backend.

---

## Current Facts To Preserve

- Fidelity is the product. Do not change 300 iterations, 32 cutouts, step size 0.1, prompt loss math, CLIP normalization, VQGAN dtype policy, or random cutout recipe.
- The current 256x256 benchmark lane is the convenience lane. Use 5 warmup + 20 timed iterations, 32 cutouts, `imagenet_16384` + `ViT-B-32`, seed 123, and swap under the 10 GB gate.
- The compiled Adam/z-clamp experiment was byte-identical but slower: previous-default 0.977 it/s vs compiled-optimizer 0.651 it/s. Do not re-enable it by default.
- Any candidate that fails byte parity or speed parity stays off by default and gets recorded as rejected in `PERFORMANCE.md` and `state.yaml`.

## File Structure

- Modify: `localvqgan/pipeline/backends/mlx_backend/cutouts.py`
  - Owns exact cutout resize helpers and the opt-in weighted implementation.
- Modify: `localvqgan/pipeline/backends/mlx_backend/generator.py`
  - Only if needed to route the small-canvas path through an opt-in weighted cutout function.
- Modify: `tools/mlx_512_benchmark.py`
  - Add a 256 candidate leg for `LOCALVQGAN_MLX_WEIGHTED_CUTOUTS=1` or add a small-canvas candidate selector.
- Create: `tests/test_mlx_weighted_cutouts.py`
  - Exact forward/gradient parity tests for the weighted path.
- Modify: `tests/test_mlx_generator.py`
  - Add a slow seeded generation parity test for weighted cutouts.
- Modify: `tests/test_benchmark_tools.py`
  - Lock benchmark leg naming/env for the weighted-cutouts candidate.
- Modify: `PERFORMANCE.md`
  - Record only measured accepted/rejected results.
- Modify: `state.yaml`
  - Keep the repo-local state truthful.

## Task 1: Add Exact Weighted Resize Unit Tests

**Files:**
- Create: `tests/test_mlx_weighted_cutouts.py`
- Modify: `localvqgan/pipeline/backends/mlx_backend/cutouts.py`

- [ ] **Step 1: Write failing forward parity tests**

Create `tests/test_mlx_weighted_cutouts.py` with this complete starting test module:

```python
import numpy as np
import pytest

pytest.importorskip("mlx")

import mlx.core as mx

from localvqgan.pipeline.backends.mlx_backend.cutouts import (
    CutoutSpec,
    apply_cutout_spec,
    apply_cutout_spec_weighted,
)


def _manual_spec(dtype=mx.float32):
    return CutoutSpec(
        sizes=[32, 71, 128],
        offsets=[(0, 0), (13, 29), (0, 0)],
        flip=mx.array([[[[False]]], [[[True]]], [[[False]]]]),
        factor=mx.array([1.0, 0.85, 1.2], dtype=dtype),
        sat=mx.array([[[[1.0]]], [[[0.997]]], [[[1.006]]]], dtype=dtype),
        fac=mx.array([[[[0.0]]], [[[0.025]]], [[[0.05]]]], dtype=dtype),
        noise=mx.arange(3 * 64 * 64 * 3, dtype=dtype).reshape(3, 64, 64, 3) * 1e-6,
    )


def test_weighted_cutouts_forward_matches_existing_path_exactly():
    img = mx.arange(1 * 128 * 128 * 3, dtype=mx.float32).reshape(1, 128, 128, 3)
    img = img / mx.array(1 * 128 * 128 * 3, dtype=mx.float32)
    spec = _manual_spec()

    reference = apply_cutout_spec(img, spec, cut_size=64)
    candidate = apply_cutout_spec_weighted(img, spec, cut_size=64)
    mx.eval(reference, candidate)

    assert np.array_equal(np.array(candidate), np.array(reference))


def test_weighted_cutouts_chunk_slice_matches_existing_path_exactly():
    img = mx.arange(1 * 128 * 128 * 3, dtype=mx.float32).reshape(1, 128, 128, 3)
    img = img / mx.array(1 * 128 * 128 * 3, dtype=mx.float32)
    spec = _manual_spec()

    reference = apply_cutout_spec(img, spec, cut_size=64, start=1, end=3)
    candidate = apply_cutout_spec_weighted(img, spec, cut_size=64, start=1, end=3)
    mx.eval(reference, candidate)

    assert np.array_equal(np.array(candidate), np.array(reference))
```

- [ ] **Step 2: Run tests and confirm they fail because the function is missing**

Run:

```bash
.venv/bin/pytest -q --timeout 120 tests/test_mlx_weighted_cutouts.py
```

Expected:

```text
ImportError: cannot import name 'apply_cutout_spec_weighted'
```

- [ ] **Step 3: Extract the existing bilinear resize weight helper**

In `localvqgan/pipeline/backends/mlx_backend/cutouts.py`, replace the nested `weights` function inside `_resize_bilinear` with a module-level helper:

```python
def _resize_weights(src: int, dst: int) -> mx.array:
    s = (mx.arange(dst) + 0.5) * (src / dst) - 0.5
    s = mx.clip(s, 0, src - 1)
    i0 = mx.clip(mx.floor(s), 0, src - 1).astype(mx.int32)
    i1 = mx.minimum(i0 + 1, src - 1)
    frac = s - i0.astype(s.dtype)
    cols = mx.arange(src)[None, :]
    return (
        (cols == i0[:, None]).astype(s.dtype) * (1 - frac)[:, None]
        + (cols == i1[:, None]).astype(s.dtype) * frac[:, None]
    )
```

Then update `_resize_bilinear` to call it:

```python
def _resize_bilinear(img: mx.array, size: int) -> mx.array:
    n, h, w, c = img.shape
    del n, c
    out = mx.einsum("iy,nyxc->nixc", _resize_weights(h, size), img)
    return mx.einsum("jx,nixc->nijc", _resize_weights(w, size), out)
```

- [ ] **Step 4: Add the first weighted implementation**

Append these helpers below `_resize_bilinear`:

```python
def _padded_resize_weights(src_total: int, offset: int, src_cut: int, dst: int) -> mx.array:
    weights = _resize_weights(src_cut, dst)
    before = mx.zeros((dst, offset), dtype=weights.dtype)
    after = mx.zeros((dst, src_total - offset - src_cut), dtype=weights.dtype)
    return mx.concatenate([before, weights, after], axis=1)


def _weighted_resize_cutouts(
    img: mx.array,
    spec: CutoutSpec,
    cut_size: int,
    start: int,
    end: int,
) -> mx.array:
    _, h, w, _ = img.shape
    row_weights = []
    col_weights = []
    for i in range(start, end):
        size = spec.sizes[i]
        oy, ox = spec.offsets[i]
        row_weights.append(_padded_resize_weights(h, oy, size, cut_size))
        col_weights.append(_padded_resize_weights(w, ox, size, cut_size))
    rows = mx.stack(row_weights, axis=0)
    cols = mx.stack(col_weights, axis=0)
    y_resized = mx.einsum("bih,nhwc->bniwc", rows, img)
    xy_resized = mx.einsum("bjw,bniwc->bnijc", cols, y_resized)
    b, n, i, j, c = xy_resized.shape
    return xy_resized.reshape(b * n, i, j, c)
```

Then add the opt-in equivalent of `apply_cutout_spec`:

```python
def apply_cutout_spec_weighted(
    img: mx.array,
    spec: CutoutSpec,
    cut_size: int,
    start: int = 0,
    end: int | None = None,
) -> mx.array:
    end = len(spec.sizes) if end is None else end
    batch = _weighted_resize_cutouts(img, spec, cut_size, start, end)

    flip = spec.flip[start:end]
    batch = mx.where(flip, batch[:, :, ::-1, :], batch)

    batch = _sharpness(batch, spec.factor[start:end])

    mean = batch.mean(axis=-1, keepdims=True)
    batch = mx.clip(mean + (batch - mean) * spec.sat[start:end], 0, 1)

    batch = batch + spec.fac[start:end] * spec.noise[start:end]
    return batch
```

- [ ] **Step 5: Run forward parity tests**

Run:

```bash
.venv/bin/pytest -q --timeout 120 tests/test_mlx_weighted_cutouts.py
```

Expected:

```text
2 passed
```

If either assertion is not byte-exact, stop. Do not loosen the test to `allclose`; record the candidate as rejected.

## Task 2: Add Gradient Parity Tests

**Files:**
- Modify: `tests/test_mlx_weighted_cutouts.py`

- [ ] **Step 1: Add exact gradient parity test**

Append this test:

```python
def test_weighted_cutouts_gradient_matches_existing_path_exactly():
    img = mx.arange(1 * 128 * 128 * 3, dtype=mx.float32).reshape(1, 128, 128, 3)
    img = img / mx.array(1 * 128 * 128 * 3, dtype=mx.float32)
    spec = _manual_spec()

    def reference_loss(x):
        return apply_cutout_spec(x, spec, cut_size=64).sum()

    def candidate_loss(x):
        return apply_cutout_spec_weighted(x, spec, cut_size=64).sum()

    reference_value, reference_grad = mx.value_and_grad(reference_loss)(img)
    candidate_value, candidate_grad = mx.value_and_grad(candidate_loss)(img)
    mx.eval(reference_value, candidate_value, reference_grad, candidate_grad)

    assert np.array_equal(np.array(candidate_value), np.array(reference_value))
    assert np.array_equal(np.array(candidate_grad), np.array(reference_grad))
```

- [ ] **Step 2: Run gradient parity test**

Run:

```bash
.venv/bin/pytest -q --timeout 120 tests/test_mlx_weighted_cutouts.py::test_weighted_cutouts_gradient_matches_existing_path_exactly
```

Expected:

```text
1 passed
```

If exact gradient parity fails, stop and reject the candidate. Do not benchmark a non-exact gradient path.

## Task 3: Route Weighted Cutouts Behind An Opt-In Flag

**Files:**
- Modify: `localvqgan/pipeline/backends/mlx_backend/cutouts.py`
- Modify: `localvqgan/pipeline/backends/mlx_backend/generator.py`
- Modify: `tests/test_mlx_generator.py`

- [ ] **Step 1: Add flag helper and weighted make_cutouts wrapper**

In `cutouts.py`, add imports and helpers near the top:

```python
import os
```

```python
def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {raw!r}")


def weighted_cutouts_enabled() -> bool:
    return _env_flag("LOCALVQGAN_MLX_WEIGHTED_CUTOUTS", False)
```

Then add a wrapper:

```python
def apply_cutout_spec_selected(
    img: mx.array,
    spec: CutoutSpec,
    cut_size: int,
    start: int = 0,
    end: int | None = None,
) -> mx.array:
    if weighted_cutouts_enabled():
        return apply_cutout_spec_weighted(img, spec, cut_size, start, end)
    return apply_cutout_spec(img, spec, cut_size, start, end)
```

Update `make_cutouts`:

```python
def make_cutouts(img: mx.array, cutn: int, cut_size: int, cut_pow: float = 1.0) -> mx.array:
    spec = make_cutout_spec(img, cutn, cut_size, cut_pow)
    return apply_cutout_spec_selected(img, spec, cut_size)
```

- [ ] **Step 2: Route generator chunked path through the selected helper**

In `generator.py`, change the cutout import:

```python
from localvqgan.pipeline.backends.mlx_backend.cutouts import (
    apply_cutout_spec_selected,
    make_cutout_spec,
    make_cutouts,
)
```

Then change the chunked branch call:

```python
cutouts = apply_cutout_spec_selected(out_, spec, self.clip.cut_size, start, end)
```

- [ ] **Step 3: Add flag tests**

In `tests/test_mlx_generator.py`, add the env to the existing tuning knob tests:

```python
monkeypatch.delenv("LOCALVQGAN_MLX_WEIGHTED_CUTOUTS", raising=False)
assert not generator.cutouts.weighted_cutouts_enabled()
```

If importing through `generator.cutouts` is not available, import the module directly in the test:

```python
from localvqgan.pipeline.backends.mlx_backend import cutouts
assert not cutouts.weighted_cutouts_enabled()
```

Add override and invalid cases:

```python
monkeypatch.setenv("LOCALVQGAN_MLX_WEIGHTED_CUTOUTS", "1")
assert cutouts.weighted_cutouts_enabled()

monkeypatch.setenv("LOCALVQGAN_MLX_WEIGHTED_CUTOUTS", "maybe")
with pytest.raises(ValueError, match="LOCALVQGAN_MLX_WEIGHTED_CUTOUTS"):
    cutouts.weighted_cutouts_enabled()
```

- [ ] **Step 4: Run focused flag and weighted tests**

Run:

```bash
.venv/bin/pytest -q --timeout 120 tests/test_mlx_weighted_cutouts.py tests/test_mlx_generator.py::test_large_canvas_tuning_knobs_default tests/test_mlx_generator.py::test_large_canvas_tuning_knobs_honor_env tests/test_mlx_generator.py::test_large_canvas_tuning_knobs_reject_invalid_env
```

Expected:

```text
all selected tests pass
```

## Task 4: Add Seeded Generation Parity Gate

**Files:**
- Modify: `tests/test_mlx_generator.py`

- [ ] **Step 1: Add slow tiny generation parity test**

Append this slow test near the other slow parity tests:

```python
@pytest.mark.slow
def test_weighted_cutouts_match_default_generation(monkeypatch):
    monkeypatch.setenv("LOCALVQGAN_MLX_WEIGHTED_CUTOUTS", "0")
    default = _run_tiny_generation(cutouts=4)

    monkeypatch.setenv("LOCALVQGAN_MLX_WEIGHTED_CUTOUTS", "1")
    weighted = _run_tiny_generation(cutouts=4)

    default_losses = np.array([f.loss for f in default], dtype=np.float32)
    weighted_losses = np.array([f.loss for f in weighted], dtype=np.float32)
    assert np.array_equal(weighted_losses, default_losses), (
        weighted_losses,
        default_losses,
    )
    assert np.array_equal(np.asarray(weighted[-1].image), np.asarray(default[-1].image))
```

- [ ] **Step 2: Run slow parity test**

Run:

```bash
.venv/bin/pytest -q --timeout 180 -m slow tests/test_mlx_generator.py::test_weighted_cutouts_match_default_generation
```

Expected:

```text
1 passed
```

If the tiny generation is not byte-identical, stop and reject the candidate.

- [ ] **Step 3: Run real 256x256 parity command**

Run this command exactly:

```bash
.venv/bin/python - <<'PY'
import os
import numpy as np
import mlx.core as mx
from localvqgan.pipeline.backends.mlx_backend.generator import MlxGenerator
from localvqgan.pipeline.settings import GenerationSettings

settings = GenerationSettings(
    prompts="a lighthouse on a cliff at dusk",
    width=256,
    height=256,
    iterations=2,
    cutouts=4,
    seed=123,
    display_freq=1,
    checkpoint="imagenet_16384",
    clip_model="ViT-B-32",
    engine="mlx",
)

def run(label, weighted):
    os.environ["LOCALVQGAN_MLX_WEIGHTED_CUTOUTS"] = "1" if weighted else "0"
    mx.clear_cache()
    g = MlxGenerator()
    g.load(settings.checkpoint, settings.clip_model)
    frames = list(g.generate(settings))
    losses = np.array([f.loss for f in frames], dtype=np.float32)
    img = np.asarray(frames[-1].image)
    print(label, "losses", losses.tolist(), "image_sum", int(img.sum()))
    return losses, img

base_losses, base_img = run("default", weighted=False)
weighted_losses, weighted_img = run("weighted", weighted=True)
print("loss_max_abs", float(np.max(np.abs(weighted_losses - base_losses))))
print("image_equal", bool(np.array_equal(weighted_img, base_img)))
print("image_max_abs", int(np.max(np.abs(weighted_img.astype(np.int16) - base_img.astype(np.int16)))))
PY
```

Expected:

```text
loss_max_abs 0.0
image_equal True
image_max_abs 0
```

If this fails, leave `LOCALVQGAN_MLX_WEIGHTED_CUTOUTS` default-off and document rejection.

## Task 5: Benchmark The Candidate Against Current Default

**Files:**
- Modify: `tools/mlx_512_benchmark.py`
- Modify: `tests/test_benchmark_tools.py`

- [ ] **Step 1: Add weighted-cutouts candidate leg**

In `tools/mlx_512_benchmark.py`, add a `--candidate` argument:

```python
parser.add_argument(
    "--candidate",
    choices=["compiled-optimizer", "weighted-cutouts"],
    default="weighted-cutouts",
)
```

Update the small-canvas branch of `legs(args)` so it returns:

```python
base = Leg(
    "previous-default",
    {
        "LOCALVQGAN_MLX_COMPILE": None,
        "LOCALVQGAN_MLX_ASYNC_CHUNKS": None,
        "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(args.chunk_size),
        "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(args.cache_gb),
        "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": "0",
        "LOCALVQGAN_MLX_WEIGHTED_CUTOUTS": "0",
    },
)
if args.candidate == "compiled-optimizer":
    candidate = Leg(
        "compiled-optimizer",
        {
            "LOCALVQGAN_MLX_COMPILE": None,
            "LOCALVQGAN_MLX_ASYNC_CHUNKS": None,
            "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(args.chunk_size),
            "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(args.cache_gb),
            "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": "1",
            "LOCALVQGAN_MLX_WEIGHTED_CUTOUTS": "0",
        },
    )
else:
    candidate = Leg(
        "weighted-cutouts",
        {
            "LOCALVQGAN_MLX_COMPILE": None,
            "LOCALVQGAN_MLX_ASYNC_CHUNKS": None,
            "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(args.chunk_size),
            "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(args.cache_gb),
            "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": "0",
            "LOCALVQGAN_MLX_WEIGHTED_CUTOUTS": "1",
        },
    )
return [base, candidate]
```

Also add `"LOCALVQGAN_MLX_WEIGHTED_CUTOUTS"` to `ENV_KEYS`.

- [ ] **Step 2: Update benchmark tool tests**

In `tests/test_benchmark_tools.py`, add:

```python
def test_256_legs_compare_weighted_cutouts_against_previous_default():
    args = bench.build_parser().parse_args([
        "--width", "256",
        "--height", "256",
        "--candidate", "weighted-cutouts",
    ])

    legs = bench.legs(args)

    assert [leg.name for leg in legs] == ["previous-default", "weighted-cutouts"]
    assert legs[0].env["LOCALVQGAN_MLX_WEIGHTED_CUTOUTS"] == "0"
    assert legs[1].env["LOCALVQGAN_MLX_WEIGHTED_CUTOUTS"] == "1"
    assert legs[1].env["LOCALVQGAN_MLX_COMPILED_OPTIMIZER"] == "0"
```

- [ ] **Step 3: Run benchmark tool tests**

Run:

```bash
.venv/bin/pytest -q --timeout 120 tests/test_benchmark_tools.py
```

Expected:

```text
all tests pass
```

- [ ] **Step 4: Run canonical 256 benchmark**

First verify swap:

```bash
sysctl vm.swapusage
```

Proceed only if used swap is below 10240 MB. Then run:

```bash
.venv/bin/python tools/mlx_512_benchmark.py --candidate weighted-cutouts
```

Expected benchmark shape:

```text
MLX benchmark: 256x256, 32 cutouts, warmup=5, timed=20, start_swap=<under 10240> MB
previous-default          <number> it/s
weighted-cutouts          <number> it/s
parity vs previous-default: weighted-cutouts loss_diff=0 image_equal=True image_max_abs=0
```

Acceptance gate:

- `image_equal=True`
- `image_max_abs=0`
- `loss_diff=0`
- `weighted-cutouts` is at least 3% faster than `previous-default`
- ending swap stays under 10240 MB

If speed is less than +3%, keep the flag default-off and document the result as rejected-for-default.

## Task 6: Promote Or Reject The Candidate

**Files:**
- Modify: `localvqgan/pipeline/backends/mlx_backend/cutouts.py`
- Modify: `PERFORMANCE.md`
- Modify: `state.yaml`

- [ ] **Step 1: If accepted, flip default to enabled**

Only if Task 5 passes every acceptance gate, change:

```python
def weighted_cutouts_enabled() -> bool:
    return _env_flag("LOCALVQGAN_MLX_WEIGHTED_CUTOUTS", True)
```

If Task 5 does not pass, leave the default as `False`.

- [ ] **Step 2: Update `PERFORMANCE.md`**

If accepted, add:

```markdown
**Enabled weighted cutouts at 256².** Crop + bilinear resize now use a
weighted full-image row/column path for small canvases. Parity proof: real
256×256, imagenet_16384/ViT-B/32, 2 iterations, 4 cutouts, seed 123 compared
default vs weighted with loss diff **0.0** and final image byte-identical.
Benchmark: copy the exact previous-default and weighted-cutouts `it/s` values
from the accepted Task 5 run, including the warmup/timed iteration counts and
the under-10-GB swap statement.
```

If rejected, add:

```markdown
**Rejected with data: weighted cutouts as a default.** The weighted full-image
row/column path was byte-identical on the real 256×256 parity gate, but the
canonical benchmark did not clear the +3% speed gate: previous-default
and weighted-cutouts measured values copied exactly from the Task 5 run. The
flag remains opt-in with `LOCALVQGAN_MLX_WEIGHTED_CUTOUTS=1`.
```

- [ ] **Step 3: Update `state.yaml`**

Add a `round_3.weighted_cutouts_256` entry:

```yaml
  weighted_cutouts_256:
    status: "implemented_default_on"  # or "rejected_for_default_path"
    default_enabled: true             # or false
    env: "LOCALVQGAN_MLX_WEIGHTED_CUTOUTS"
    parity_proof:
      command: "real 256x256 imagenet_16384/ViT-B-32, 2 iterations, 4 cutouts, seed 123, default vs weighted"
      loss_diff: 0.0
      final_image: "byte-identical"
    benchmark_result:
      command: ".venv/bin/python tools/mlx_512_benchmark.py --candidate weighted-cutouts"
      canonical: true
      width: 256
      height: 256
      cutouts: 32
      warmup_iterations: 5
      timed_iterations: 20
      previous_default:
        it_per_sec: 0.0
      weighted_cutouts:
        it_per_sec: 0.0
      parity:
        loss_diff: 0.0
        image_equal: true
        image_max_abs: 0
```

Replace the `0.0` example values with the actual numbers from Task 5 before
running the final verification. A zero value is valid only if the benchmark
actually printed `0.000 it/s`.

- [ ] **Step 4: Run final verification**

Run:

```bash
.venv/bin/pytest -q --timeout 120
.venv/bin/pytest -q --timeout 180 -m slow tests/test_mlx_generator.py::test_weighted_cutouts_match_default_generation
git diff --check
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path("state.yaml").read_text())
print("state ok")
PY
```

Expected:

```text
fast suite passes with only the known StarletteDeprecationWarning
slow weighted parity test passes
git diff --check has no output
state ok
```

## Self-Review

- Spec coverage: the plan targets the next 256x256 speed candidate, preserves quality defaults, requires exact forward/gradient and seeded-image parity, and rejects speed regressions by default.
- Placeholder scan: implementation steps include concrete code and commands. Documentation updates require copying measured values from Task 5 rather than filling fake example numbers.
- Type consistency: all new helpers operate on existing `CutoutSpec`, `mx.array`, `cut_size`, `start`, and `end` shapes. The benchmark flag name is consistently `LOCALVQGAN_MLX_WEIGHTED_CUTOUTS`.

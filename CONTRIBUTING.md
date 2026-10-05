# Contributing

Thanks for your interest! LocalVQGAN has one unusual constraint worth
understanding before you start: **fidelity is the product.** The point of the
project is reproducing the 2021 VQGAN+CLIP aesthetic exactly, so changes are
judged on output preservation first and speed second.

## Ground rules

- **Never change the quality defaults** (300 iterations, 32 cutouts, step
  0.1, default 256x256) to gain speed. Speed features that alter outputs
  (like Fast mode) ship opt-in, never as the default.
- **Any performance claim needs a measurement** recorded in
  [PERFORMANCE.md](PERFORMANCE.md) — warmup iterations, sequential legs,
  swap state, and the commit it applies to. Ideas that were measured and
  rejected are recorded there too; check it before re-proposing one.
- **Exact-output changes need proof**: seeded runs before/after your change
  should produce byte-identical images (the test suite has examples of
  parity gates to copy).
- **Keep torch out of the MLX/server import chain.** The Apple-Silicon
  install has no torch. `tests/test_slim_install.py` runs the server and a
  real generation in a subprocess with torch imports blocked — it will catch
  a stray module-level import, but save yourself the round trip.

## Development setup

Use Python 3.11+ and work from the repository root. The launch scripts
install runtime dependencies only; the `[dev]` extra adds pytest,
pytest-timeout, httpx, Ruff, and the torch engine used by parity tests.
On native Apple-Silicon Python, the normal platform dependency also adds
MLX. The dev environment is intentionally larger than the torch-free
end-user install.

macOS/Linux:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

If PowerShell activation is unavailable, use
`.\.venv\Scripts\python.exe` in place of `python` in the commands below.
On macOS/Linux, `.venv/bin/python` is the corresponding direct path.
Keep using that environment's Python for installation, the app, and tests.

## Tests

```sh
python -m pytest -q --timeout 120    # default selection: not slow
python -m pytest -q -m slow         # parity and real-generation checks
python -m ruff check .
```

The default selection is configured in `pyproject.toml`; it excludes tests
marked `slow`, but is **not guaranteed to be download-free**. Some unmarked
MLX generator tests use the tiny VQGAN fixture with real CLIP weights. On a
cold Apple-Silicon environment, prepare those weights before the 120-second
per-test timeout, matching CI:

```sh
python -c "from localvqgan.pipeline.backends.mlx_backend.convert import cached_clip_weights; cached_clip_weights('ViT-B-32')"
```

This downloads and converts CLIP once if it is not already cached. Run it
only in an MLX-capable Apple-Silicon environment. Slow tests can download
large models, exercise real generation, and consume significant GPU/RAM;
some require locally cached weights and skip when those are absent. Read
the relevant test's prerequisites and report skips rather than treating
them as coverage.

The fast suite and Ruff must pass on your machine before a code PR. For
documentation-only PRs, verify relative links and commands against the
source, and say explicitly if runtime checks were not run. Project notes
record a pre-existing starlette-testclient deprecation warning; don't assume
other warnings are benign. In particular, NumPy "invalid value in cast"
during generation means non-finite image data and should be investigated.

### CI coverage

[tests.yml](.github/workflows/tests.yml) runs the default suite on Ubuntu
with Python 3.11 and Apple-Silicon macOS with Python 3.12, a separate
torch-free macOS install check, Ruff, and wheel/sdist validation. Windows
has a launcher but is not in the current CI matrix. Slow model-generation
tests are not part of the default CI job, so relevant local fidelity checks
still need to be reported when changing generation behavior.

## Running the app from source

With the development environment active:

```sh
python -m localvqgan
python -m localvqgan --no-browser --port 8421 --outputs ./outputs
```

The terminal prints the actual loopback URL; the preferred port falls
forward if occupied. The launch scripts (`./run.sh` on macOS/Linux,
`run.bat` on Windows) are the first-run end-user path and do not forward
CLI arguments. They also skip dependency installation when the `.venv`
interpreter exists; rerun `python -m pip install -e ".[dev]"` after
dependency changes.

Keep the same output folder when checking gallery behavior. Cache/model
files live under `~/.cache/localvqgan/`; gallery runs live under the
selected output root. Don't commit caches, virtual environments, or
generated output.

## Finding the relevant code

- [Generation settings](localvqgan/pipeline/settings.py) and
  [web controls](localvqgan/web/index.html): keep defaults aligned
- [Engine dispatcher](localvqgan/pipeline/backends/__init__.py): capability
  checks, explicit selection, and the measured auto-selection gate
- [Torch engine](localvqgan/pipeline/backends/torch_backend.py) and
  [MLX engine](localvqgan/pipeline/backends/mlx_backend/): generation paths
- [Server app](localvqgan/server/app.py) and
  [job manager](localvqgan/server/jobs.py): API, queue, previews, sidecars
- [Web UI](localvqgan/web/app.js): gallery, batching, reuse, and upscale
- [Tests](tests/): small fixtures, backend parity, replay, API behavior, and
  [torch-free import/generation guards](tests/test_slim_install.py)

## Before opening a pull request

- Describe the problem, affected engine/platform, and the change's scope
- List the exact checks you ran, their results, and any skips or blockers
- For generation changes, include seeded before/after evidence appropriate
  to the fidelity rule above; preserve the default path and test opt-in
  behavior separately
- For speed claims, record the workload, hardware, versions, warmup,
  memory/swap conditions, and commit in [PERFORMANCE.md](PERFORMANCE.md)
- If behavior or setup changes, update the relevant README instructions
  and contributor guidance in the same PR

For bug reports, use the [issue tracker](https://github.com/RasputinKaiser/LocalVQGAN/issues)
and include reproduction steps, OS, Python/dependency versions, engine,
device, checkpoint/CLIP model, canvas size, seed, and the exact error or
warning. A minimal `settings.json` can help; remove private prompt text and
local file paths before posting it publicly. Don't include model weights
or a whole output/cache folder.

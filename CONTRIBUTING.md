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

```
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # includes torch; add mlx on Apple Silicon
```

## Tests

```
.venv/bin/pytest -q --timeout 120      # fast suite, no model downloads
.venv/bin/pytest -q -m slow            # downloads models, runs real generation
```

The fast suite must pass on your machine before a PR. The baseline has
exactly one known warning (a starlette-testclient deprecation) — any new
warning is treated as a finding, especially NumPy "invalid value in cast"
(that means NaN images).

## Running the app from source

```
./run.sh        # macOS/Linux (run.bat on Windows) -> http://127.0.0.1:8420
```

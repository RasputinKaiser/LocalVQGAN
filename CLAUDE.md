# LocalVQGAN — project notes for Claude

Historic VQGAN+CLIP recreation: same aesthetic/math as the 2021 Colab, modernized, fast, installable by anyone. Fidelity is the product — never change quality defaults (300 iterations, 32 cutouts, step 0.1) to gain speed.

## Commands
- venv: `.venv/bin/pytest -q --timeout 120` (fast suite) · `-m slow` downloads models / real generation
- run app: `./run.sh` → http://127.0.0.1:8420 · dev preview server is `.claude/launch.json` name `localvqgan`
- outputs land in `./outputs/<run-id>/` with a `settings.json` sidecar (`engine_used` records resolution)

## Architecture facts
- Two engines behind `localvqgan/pipeline/backends/`: `torch_backend.py` (MPS/CUDA/CPU) and `mlx_backend/` (Apple Silicon). Dispatcher: `backends.resolve_engine(engine, checkpoint, clip_model, width, height)` — auto prefers MLX only ≤256² (`MLX_MAX_AUTO_PIXELS`) and only while `MLX_MEETS_SPEED_GATE` holds (re-measure before flipping).
- MLX runs VQGAN fp32 + CLIP fp16 (fp16 VQGAN overflows to Inf in the decoder up-chain at 256² — do not "optimize" it back to fp16 without instrumenting activation maxima).
- Weight caches: `~/.cache/localvqgan/<ckpt>/` (torch), `<ckpt>/mlx/` + `clip/<model>/mlx/` (converted). faceshq/ade20k/ffhq/celebahq mirrors are dead upstream (`mirror_offline=True`).
- Measured baselines (M1 16GB, 2026-07-07, 256²/32cut): torch 0.680 it/s, MLX 0.929 it/s. Any perf claim goes in PERFORMANCE.md with commit citation.

## Gotchas that already bit us
- NumPy `RuntimeWarning: invalid value encountered in cast` during generation/tests = NaN images. Treat as failure, never benign.
- Torch-MPS: patch-embed conv backward is pathological (FastPatchEmbed replaces it); no grid_sample or antialias-resize backward in hot loops; torch.compile is a measured regression on MPS.
- Benchmarks: sequential legs, 5+ warmup its (mx.compile), no taskpolicy, nothing else on the GPU; per-iteration seconds that climb = memory thrash (check `sysctl vm.swapusage`).
- Heavy local runs otherwise: `taskpolicy -c utility` + `torch.set_num_threads(2)` to keep the Mac usable.
- pytest baseline includes 1 pre-existing starlette-testclient deprecation warning; new warnings are findings.

# LocalVQGAN — project notes for Claude

Historic VQGAN+CLIP recreation: same aesthetic/math as the 2021 Colab, modernized, fast, installable by anyone. Fidelity is the product — never change quality defaults (300 iterations, 32 cutouts, step 0.1) to gain speed.

## Commands
- venv: `.venv/bin/pytest -q --timeout 120` (fast suite) · `-m slow` downloads models / real generation
- run app: `./run.sh` → http://127.0.0.1:8420 · dev preview server is `.claude/launch.json` name `localvqgan`
- outputs land in `./outputs/<run-id>/` with a `settings.json` sidecar (`engine_used` records resolution)

## Architecture facts
- Two engines behind `localvqgan/pipeline/backends/`: `torch_backend.py` (MPS/CUDA/CPU) and `mlx_backend/` (Apple Silicon). Dispatcher: `backends.resolve_engine(engine, checkpoint, clip_model, width, height)` — auto prefers MLX at ALL sizes while available+supported+`MLX_MEETS_SPEED_GATE` (re-measure before flipping the gate).
- Default resolution is 256² (settings.py + web/index.html); it's ~2× faster than 384² and the pristine fp32 path. Don't silently bump it back.
- MLX precision policy: ≤256² VQGAN fp32 (historic bit-stable seeds — bf16 there was measured 1.07× and REJECTED for changing seeds) + CLIP fp16. >256² (`MLX_LARGE_CANVAS_PIXEL_THRESHOLD` in mlx generator): VQGAN bf16 + 4GB `mx.set_cache_limit` bracket. Cutouts chunk (CHUNK=8, exact-math gradient accumulation via two-stage vjp) ONLY above `MLX_UNCHUNKED_MAX_PIXELS` (384²) — 384² runs single-pass (+12%, measured), 512² needs chunking or it thrashes. Never fp16 VQGAN (Inf overflow in decoder up-chain).
- Iteration-speed ceiling (measured round 3, 2026-07-08): 256² is COMPUTE-bound at ~0.93 it/s (~85-90% of M1 8-core GPU roofline). Proven dead: fused mx.fast kernels (+0.1%, work isn't attention-shaped), early-stop (image never converges, only loss does), bf16@256² (1.07× + seed change), fused compiled optimizer (T8 — 0.04% ceiling by probe). No look-preserving 256² *per-iteration* lever remains — don't re-chase these; see PERFORMANCE.md round-3.
- The only wall-clock lever left is `fast_mode` (coarse-to-fine, round 4): first 60% of iterations at half-res, upsample latent, finish at full res. 1.30× at 256² (MLX), look preserved in distribution (blue-cast/grit unchanged over 3 seeds; mild ~10% luminance dip). Opt-in ONLY — breaks per-seed reproducibility. Additive `_generate_coarse_to_fine` path on BOTH engines; pristine defaults untouched. Torch gotcha: the MPS fp16 decoder overflows to NaN on upsampled latents (fine from one-hot inits) — the driver redoes just the fine stage with an fp32 VQGAN, so torch fast mode measures smaller (~1.1× diagnostic); a torch bf16 fine stage is the open follow-up. Torch decoder gradient checkpointing (`fast_mode` unrelated) rescues torch 512² DNF on ≤16GB, exact math, gated >384².
- MLX streams are thread-local; every load/set_dtype must `mx.eval` params or the next job thread dies with "There is no Stream(gpu, N)". The chunked path must stay EAGER — per-chunk `mx.eval` is illegal under `mx.compile`, and the fixed-shape reformulation for compile reuse diverged 0.1% (rejected).
- Weight caches: `~/.cache/localvqgan/<ckpt>/` (torch), `<ckpt>/mlx/` + `clip/<model>/mlx/` (converted). faceshq/ade20k/ffhq/celebahq mirrors are dead upstream (`mirror_offline=True`).
- Measured baselines (M1 16GB, 2026-07-07, 32cut): 256² torch 0.680 / MLX 0.929 it/s (MLX ~85-90% of this GPU's roofline — don't chase 1.5 it/s here); 512² torch 0.010/DNF (>16GB working set) / MLX 0.16–0.25 it/s. Any perf claim goes in PERFORMANCE.md with commit citation.

## Gotchas that already bit us
- NumPy `RuntimeWarning: invalid value encountered in cast` during generation/tests = NaN images. Treat as failure, never benign.
- Torch-MPS: patch-embed conv backward is pathological (FastPatchEmbed replaces it); no grid_sample or antialias-resize backward in hot loops; torch.compile is a measured regression on MPS.
- Benchmarks: sequential legs, 5+ warmup its (mx.compile), no taskpolicy, nothing else on the GPU; per-iteration seconds that climb = memory thrash (check `sysctl vm.swapusage`).
- Heavy local runs otherwise: `taskpolicy -c utility` + `torch.set_num_threads(2)` to keep the Mac usable.
- pytest baseline includes 1 pre-existing starlette-testclient deprecation warning; new warnings are findings.

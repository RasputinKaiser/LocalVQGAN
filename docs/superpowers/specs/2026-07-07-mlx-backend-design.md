# LocalVQGAN MLX Backend — Design Spec

**Date:** 2026-07-07
**Goal:** Add an Apple-silicon-native MLX engine to LocalVQGAN as a selectable backend behind the existing GUI, preserving the exact VQGAN+CLIP aesthetic (same checkpoints, same loss math) while running meaningfully faster than PyTorch-MPS on the M1. PyTorch remains the engine for CUDA/CPU — cross-platform behavior is unchanged.

**Why:** PyTorch-MPS has been optimized to its ceiling (~0.68 it/s at 256²/32 cutouts; profile shows pure compute, and torch.compile/channels_last measured as regressions). Core AI/Core ML is inference-only — VQGAN+CLIP generation requires runtime backprop, so MLX (autograd + native Metal + unified memory + lazy-eval fusion) is the only true M1-native path. Expected 1.5–3× over torch-MPS; this is a speed project, not a quality change.

## Scope

Core-first with automatic fallback (user-approved):
- MLX covers: VQModel checkpoints (imagenet_1024/16384, wikiart_1024/16384, coco, sflckr) + CLIP ViT-B-32 and ViT-B-16; all prompt features (pipe/weight/stop, image prompts, init image, init weight, seed, step size, cutouts); animation mode (free — drives the same iterator).
- Not on MLX initially: Gumbel checkpoint (gumbel_8192) and anything else unsupported → silently handled by the PyTorch backend with a visible "engine: torch (fallback reason)" note.
- Quality defaults (iterations, cutouts) unchanged — per standing user constraint.

## Architecture

```
localvqgan/pipeline/
  generator.py            # becomes thin dispatcher: make_generator(engine) + shared FrameUpdate/GenerationOOM
  backends/
    __init__.py           # engine registry, availability + capability checks
    torch_backend.py      # today's Generator moved verbatim (zero behavior change)
    mlx_backend/
      __init__.py
      vqgan.py            # taming VQGAN encoder/decoder/quantize in mlx.nn (adapted from mlx-examples VAE, MIT)
      clip.py             # CLIP ViT-B/32 + B/16 visual & text towers in mlx.nn
      cutouts.py          # random crops + bilinear resize, hflip, sharpness, color jitter, noise
      convert.py          # one-time torch ckpt -> MLX safetensors, cached ~/.cache/localvqgan/<name>/mlx/
      generator.py        # MlxGenerator: same load()/generate(settings, cancel) iterator contract
```

- Server/API/GUI consume only the dispatcher; existing imports keep working via a shim (`from localvqgan.pipeline.generator import Generator` continues to resolve to the torch class for tests).
- `GenerationSettings.engine: str = "auto"` ("auto" | "mlx" | "torch"), recorded in settings.json sidecars.
- Engine resolution (in dispatcher): explicit engine honored if available (error surfaced if not installed); "auto" = MLX iff platform is Apple Silicon AND `import mlx` succeeds AND `MlxGenerator.supports(checkpoint, clip_model)`; else torch. Resolution result + reason published in WS status and stored in the sidecar (`engine_used`).
- Dependency: optional extra `mlx = ["mlx>=0.21"]` in pyproject; all mlx imports guarded.

## Algorithm parity (the aesthetic contract)

Identical math to the torch engine, reimplemented in MLX:
- z init: one-hot sample @ codebook, or encoder(init image); z and Adam state fp32; model weights fp16.
- Per iteration: vector-quantize z (straight-through gradient via `z_q + mx.stop_gradient`-composition — same as replace_grad), decode, add(1)/div(2), clamp-with-grad, cutouts, CLIP encode, arcsin-squared spherical distance loss with weight sign + stop thresholds, optional init-weight MSE, Adam step, z clamped to codebook min/max.
- Cutouts: same size distribution (cut_pow), bilinear resize to cut_size, hflip p=0.5, sharpness p=0.4, color jitter (hue/sat 0.01, p=0.7), uniform noise fac 0.1. (No affine/perspective — matches the current MPS torch path.)
- `mx.compile` wraps the loss+grad step function; seeds via `mx.random.seed` (MLX runs are self-reproducible; cross-engine bit-identity is not promised — parity is enforced statistically, below).

## Weight conversion

- `convert.py`: loads the cached torch checkpoint (reusing the existing safe-globals loader) + open_clip state dict, maps names to the MLX modules, writes fp16 safetensors to `~/.cache/localvqgan/<name>/mlx/` (and `~/.cache/localvqgan/clip/<model>/mlx/`).
- Triggered lazily on first MLX use of a checkpoint; progress published over the existing WS download-progress channel ("preparing MLX weights"). Conversion failure → clear error + automatic torch fallback.

## GUI

- Engine dropdown (auto / mlx / torch) next to the CLIP picker; disabled states when mlx not installed.
- Status bar shows the engine actually used for the running job.
- No other GUI changes.

## Error handling

- mlx not importable → engine list omits it; "auto" silently resolves to torch.
- Non-finite loss on iteration 1 in fp16 → fp32 retry once (mirror of torch path).
- OOM → same GenerationOOM surfacing.
- Cancel honored between iterations, same as torch.
- `mx.compile` failure at first step → run eager with a logged warning (never fatal).

## Acceptance gate for defaulting

"auto" prefers MLX only if the final benchmark shows ≥1.2× over torch-MPS at 256²/32cut. Below that, MLX ships as opt-in ("mlx" explicit) and "auto" keeps torch, until a later MLX release earns the default.

## Verification

- Unit (fast): converter name-mapping roundtrip on the tiny fixture VQGAN — convert torch tiny model to MLX, decode the same z in both, PSNR > 50 dB. Loss-math parity: identical synthetic embeddings through torch Prompt vs MLX loss → agreement < 1e-4.
- Slow: real CLIP conversion — text+image embedding cosine vs open_clip > 0.999; end-to-end MLX smoke (5 its at 128² on imagenet_16384); visual A/B of a fixed-prompt run per engine (human check).
- API: engine field accepted, sidecar records engine_used, fallback path exercised with a fake unsupported checkpoint.
- Benchmark task at the end: 256² and 384² at 32 cutouts, mlx vs torch, reported honestly against the 1-min/256² goal.

## Non-goals

- Gumbel/segmentation-conditioned checkpoints on MLX (torch fallback covers them).
- MLX-Swift / native Mac app.
- Any change to PyTorch-path behavior, defaults, or cross-platform support.
- Bit-identical outputs across engines (same aesthetic and statistics, different RNG streams).

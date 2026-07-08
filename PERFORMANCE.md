# Performance Log — LocalVQGAN

Every optimization applied to this project, with measured numbers and the
commit that introduced it. All measurements on the reference machine unless
stated otherwise: **Apple M1 MacBook Pro, 16 GB unified memory, macOS,
Python 3.14, torch 2.12.1 (MPS), mlx ≥ 0.21**. Standard workload:
imagenet_16384 checkpoint + CLIP ViT-B/32, 32 cutouts, fp16 weights.

## Headline

| configuration | 256×256 | 384×384 |
|---|---|---|
| Original notebook code, faithful M1 port (pre-optimization) | **0.023 it/s** (43.5 s/iteration) | unusable |
| LocalVQGAN torch engine (MPS) | **0.680 it/s** | 0.283 it/s |
| LocalVQGAN MLX engine | **0.929 it/s** | 0.015 it/s (memory thrash — auto-routed to torch) |

**Net: ~40× faster than the original code running on the same machine**
(0.023 → 0.929 it/s at 256²). The original Colab notebook is CUDA-only and
does not run on Apple Silicon at all without porting; the 0.023 it/s baseline
is our faithful port before any optimization. A free-tier Colab T4 (the
original's typical home) ran this workload at roughly 1–2 it/s *after* a
3–6 minute per-session setup tax; LocalVQGAN starts generating in ~2 seconds
with models kept warm in a local server.

## The optimization chain (PyTorch/MPS engine)

Each row lists the change, its measured effect, and the commit on `main`.

| # | change | effect | commit |
|---|---|---|---|
| 1 | **Eliminate per-session setup.** Models load once into a warm FastAPI server instead of re-cloning repos + reinstalling per run (the Colab pattern). | minutes → ~2 s to first iteration | `2c9403d` |
| 2 | **MPS-native augmentations.** kornia RandomAffine/RandomPerspective backward (`grid_sampler_2d_backward`) has no Metal kernel; the CPU fallback saturated every core each iteration (10 iterations took 485 s on a tiny test model and lagged the whole machine). Dropped those two augs on MPS only; random cutout offsets already supply translation diversity. | removed the CPU-fallback stall; machine stays responsive | `43157d3` |
| 3 | **`FastPatchEmbed` — the big one.** torch 2.12.1's MPS kernel for stride-32/16 conv *input gradients* (CLIP ViT patch embedding) is pathological: **43.1 s of a 43.6 s iteration** was this single backward op (measured by module-level bisection). A stride-k kernel-k conv is exactly a patch reshape + matmul, which is bit-exact (max diff 0.0) and has a fast backward. Swapped in on MPS. | 43.6 s/it → ~1.6 s/it (**~27×**) | `19c7fd9` |
| 4 | **fp16 weights instead of autocast.** autocast re-casts fp32 weights per op every iteration — pure memory-bandwidth waste on unified memory. Models convert to fp16 once at load; latent, Adam state, and loss math stay fp32 (with an iteration-1 non-finite → fp32 retry guard). | bandwidth halved on the hot path | `28821b7` |
| 5 | **GPU→CPU sync elimination.** `float(loss)` forced a device sync every iteration; loss now materializes only on preview frames. | removed a per-iteration pipeline stall | `28821b7` |
| 6 | **MPS-native bilinear cutouts.** antialiased resize backward falls back to a CPU kernel (fp32-only); plain bilinear is Metal-native and lets the cutout stage run fp16. | removed the last CPU-fallback op in the loop | `1903625` |
| 7 | **Preview decode reuse.** Preview frames reuse the iteration's already-computed decode instead of a second full VQGAN forward every `display_freq` iterations. | ~20 % fewer decodes at display_freq=5 | `1f12768` |
| 8 | **Rejected with data: `torch.compile` and channels_last.** Measured on the `compile-experiment` branch: compile crashes standalone on MPS (inductor stride assertion in `convolution_backward`); compile+channels_last runs but is **25 % slower** than eager (0.51 vs 0.68 it/s); channels_last alone 12 % slower (0.60). Not merged. | negative result, documented | branch `compile-experiment` (`8c93208`) |

Post-chain steady state (torch/MPS): **0.68 it/s at 256²/32cut** — the
measured profile at that point is pure, well-distributed compute (24 % VQGAN
decode, 16 % CLIP forward, 48 % backward), i.e. no pathological op remains.

## The MLX engine (Apple-native, branch `mlx-backend`)

MLX has real autograd on native Metal with unified memory and lazy-eval
kernel fusion — the only true Apple-native path for an algorithm that
backpropagates at generation time (Core ML/ANE are inference-only).

**Fidelity proofs** (the aesthetic is the product; every port step is gated
by a numeric parity test in CI):

| component | parity evidence | commit |
|---|---|---|
| VQGAN encoder/decoder | decode PSNR **128.4 dB**, encode **136.3 dB** vs torch (gate: >50) | `5053865` |
| CLIP ViT-B/32 & B/16 | text/image embedding cosine **≈ 1 − 1e-12** vs open_clip (gate: >0.999) | `592cb5a`, `4f44f93` |
| Loss math (arcsin-squared spherical distance, straight-through VQ, clamp-with-grad) | forward+gradient agreement < 1e-4 vs torch reference (measured ~1e-9) | `634b060` |
| Bilinear cutout resize | matches `torch.nn.functional.interpolate` < 1e-5, both directions; deterministic vjp (matmul form avoids Metal scatter-add atomics) | `2c51d72`, `4e02937` |
| Adam trajectory | bias-corrected to match torch.optim.Adam step-for-step (mlx default omits bias correction → uncorrected steps are 2–3× larger) | `5b0edaa` |

**Benchmark** (5 warmup + 20 timed iterations, sequential legs, one process;
384² MLX result independently reproduced in an isolated mlx-only process to
rule out contention): torch 0.680 / 0.283 it/s, MLX **0.929** / 0.015 it/s at
256²/384². Ratio at 256² = **1.37×** → the `auto` engine prefers MLX (gate
threshold 1.2×), and size-aware routing keeps ≥384² jobs on torch where MLX
exceeds this machine's 16 GB working set. Commit `039fdf3`.

**The fp16 NaN discovery and fix**: The original 1.021 it/s benchmark was
measured while the engine was silently producing NaN losses from iteration 2
onward due to two issues: `clamp_with_grad`'s custom-gradient dtype promotion
bug when the VQGAN output was not exactly float32, and fp16 dynamic-range
overflow in the VQGAN decoder's ResNet/Upsample chain at 256², where
activations grew and hit Inf near the final upsample before decode. Direct
layer-by-layer instrumentation confirmed the overflow path. The fix casts
`clamp_with_grad` bounds to the input dtype, and VQGAN now loads and runs in
float32 while CLIP stays float16, where no range or precision issue was
observed. Re-measured at 256² with the fix: 0.873 / 0.929 / 0.985 it/s across
3 runs (avg 0.929, torch reference 0.680 unchanged), ratio ~1.37, still
clearing the 1.2x gate.

## Round 2: large canvases (512²) — 2026-07-07 evening

Target sizes per the user: 256² and 512². Findings, each measured on the
reference M1 16 GB (ambient desktop load noted where it matters):

**512² was unusable on both engines.** torch's full-loop working set exceeds
16 GB: three attempts measured **0.010 / 0.011 it/s or DNF** (per-stage
profile shows 0.284 it/s of pure compute — the gap is swap death, not
arithmetic). MLX fp32 thrashed the same way (74–123 s/it, +4.5 GB swap).

**The fix that shipped (`569cb35`..`4c56ec3`, merge `7b7bc96`)** — three
parts, all gated:

1. **Chunked cutout evaluation** above 256²: decode once per iteration
   (two-stage `mx.vjp`), run the 32-cutout CLIP branch in chunks of 8 with
   per-chunk `mx.eval` to bound peak memory. Equivalence vs the unchunked
   path: loss diff **0.0** (cutn=4), 6e-8 (remainder case), images
   pixel-identical.
2. **bf16 VQGAN decode** above 256² only: bf16 has fp32's exponent range (the
   fp16 Inf-overflow cannot recur); decode PSNR vs fp32 = **55.1 dB**
   (gate >50). At 256² bf16 was measured **1.07×** and REJECTED — it changes
   seeded outputs, and the historic fp32 behavior is the product there.
3. **4 GB `mx.set_cache_limit`** during large-canvas generations (restored +
   `mx.clear_cache()` after): without it the MLX buffer cache grows without
   bound across iterations — 3.7 s/it climbing to 143 s/it with +11 GB swap
   over 13 iterations; with it, stable ~4.0–6.2 s/it.

**Result: MLX 512² = 0.16–0.25 it/s across four runs** (spread is ambient
memory pressure; loss bit-identical across all runs — the path is
deterministic), vs torch 0.010/DNF. **`auto` now prefers MLX at every
supported size** (`4c56ec3` removed the 256² auto cap).

**Rejected with data, round 2:**
- bf16 at 256² (1.07×, changes seeds — see above).
- Compiled chunk reuse: the fixed-shape reformulation needed to make
  `mx.compile` amortize across chunks diverged from the eager reference by
  ~0.1 % relative loss at iteration 1 — a real math change, not float
  reassociation — caught by the seeded equivalence test and reverted. The
  chunked path ships eager.

**256² is near its ceiling on this GPU.** Per-stage MLX profile (compiled
production step 1171 ms ≈ 0.85 it/s): decode fwd 336 ms, CLIP fwd 189 ms,
cutouts 77 ms, backward ≈ 843 ms — no pathological op, and `mx.compile` is
already worth 19 % (1446 → 1173 ms eager→compiled). Roofline math with the
fp32 decoder puts this M1 (8-core GPU, ~2.6 TFLOPS fp32) at ~1.0–1.3 it/s
best case: the measured 0.85–1.0 it/s is 80–90 % of ceiling. A 1.5 it/s
256² target needs an M-Pro/Max-class GPU or math changes that would alter
outputs; neither is applied silently.

## Reproducing any number

```
# torch/MPS steady state (after warmup):
.venv/bin/pytest -m slow tests/test_smoke.py     # end-to-end on real weights
# per-stage profile and engine benchmark scripts are quoted verbatim in
# docs/superpowers/plans/*.md (Tasks 8 of each plan); every parity gate
# above runs in the test suite: .venv/bin/pytest -m slow
```

Numbers were collected 2026-07-06/07; exact methodology (warmups, thread
caps, sequential legs) is recorded next to each measurement's commit.

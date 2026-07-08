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

## Round 3: first findings — 2026-07-08

**Rejected with data: `mx.fast.scaled_dot_product_attention` as a drop-in
replacement.** The fused kernel is tempting because CLIP/VQGAN attention is
still hand-rolled, but it is not bit-exact against the current formula on this
MLX build. A direct VQGAN-shape probe measured max absolute deltas of
4.77e-7 for fp32, 9.77e-4 for fp16, and 8.79e-3 for bf16. Those are small
numerically, but the project gate is seeded-output parity, not "visually close",
so no fused-attention substitution has shipped.

**Added a backward-attribution profiler.** `tools/mlx_backward_profile.py`
times synth forward, CLIP+cutout backward, synth pullback, two-stage backward,
and full eager `value_and_grad` for the 256² MLX path. It is measurement-only
and refuses larger canvases so it cannot accidentally exercise the unchunked
512² path.

**Added sweep knobs for the large-canvas chunk path.** Defaults remain chunk
size 8 and 4 GB MLX cache limit, but benchmark runs can now set
`LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE=16` or `32` and
`LOCALVQGAN_MLX_CACHE_LIMIT_GB=2`, `4`, or `6` without editing source. Invalid
values fail fast instead of silently producing an unusable measurement.

**Compiled large-canvas synth and synth-pullback separately.** The chunked
CLIP branch remains eager because per-chunk `mx.eval` cannot be traced, but the
fixed-shape VQGAN synth and its VJP now compile once per large-canvas
generation when `LOCALVQGAN_MLX_COMPILE` is not `0`. Parity proof: real
512×512, imagenet_16384/ViT-B-32, 2 iterations, 4 cutouts, seed 123 compared
eager vs compiled with loss diff **0.0** and final image byte-identical. No
speed claim yet; rerun under the benchmark protocol after memory pressure is
clean.

**Reused the large-canvas loss decode for preview/final frame emission.** The
chunked path already decodes the pre-update latent before computing the CLIP
branch, so preview frames now reuse that exact decoded array instead of running
a second `_synth(preview_z)`. The VJP still legitimately replays synth for the
gradient; this removes only the extra display decode. Parity proof remained the
same on the real 512×512 check above: loss diff **0.0** and final image
byte-identical.

**Enabled bounded async scheduling for large-canvas chunks.** The chunked CLIP
branch now uses `mx.async_eval(chunk_loss, chunk_grad)` with at most two chunks
in flight, then drains in original chunk order before accumulating. Set
`LOCALVQGAN_MLX_ASYNC_CHUNKS=0` to force the old synchronous scheduling. Parity
proof: real 512×512, imagenet_16384/ViT-B-32, 2 iterations, 4 cutouts, seed 123
compared sync chunks vs async chunks with loss diff **0.0** and final image
byte-identical. No speed claim yet; rerun under the benchmark protocol after
memory pressure is clean.

**Rejected with data: compiled small-canvas Adam + z-clamp as a default.** The
existing compiled `value_and_grad` boundary stayed unchanged, while an
experimental optimizer step ran the bias-corrected Adam update and codebook
clamp through a separate compiled function. Parity proof cleared: real
256×256, imagenet_16384/ViT-B/32, 2 iterations, 4 cutouts, seed 123 compared
previous default vs compiled optimizer with loss diff **0.0** and final image
byte-identical. But the canonical 256² benchmark regressed:
previous-default **0.977 it/s** vs compiled-optimizer **0.651 it/s** (5 warmup
+ 20 timed iterations, 32 cutouts, start/end swap under the 10 GB gate, final
image byte-identical). The compiled optimizer remains opt-in for experiments
with `LOCALVQGAN_MLX_COMPILED_OPTIMIZER=1`, but the default stays on the faster
previous route. A mathematically equivalent Adam rewrite was also rejected
during implementation because it produced a one-level final-pixel drift; the
kept helper mirrors MLX Adam's source operation order.

**Rejected before benchmarking: weighted cutouts as a default.** The proposed
full-image row/column matrix path for composing crop + bilinear resize matched
the existing cutout path in forward evaluation, but failed the exact gradient
gate: max gradient delta **4.77e-7** across 27 elements on the controlled
128×128 test fixture. That is small numerically, but not acceptable for the
256² seeded-output policy, so the candidate was removed before any speed run.

**Rejected at parity probe: MLX FastPatchEmbed.** Replacing CLIP ViT-B/32's
stride-32 `nn.Conv2d` patch embedding with the same reshape+matmul trick that
helped torch/MPS was not exact on MLX: direct output max absolute delta was
**1.91e-5** against the current Conv2d. No generation path shipped.

**Enabled small-canvas intermediate preview decode reuse.** For 256² runs that
display intermediate previews (`display_freq < iterations`), the MLX path now
reuses the VQGAN decode already computed inside the loss graph for non-final
preview frames. The final/saved frame still uses the historical second `_synth`
decode, preserving final output parity. Parity proof: real 256×256,
imagenet_16384/ViT-B/32, 2 iterations, 4 cutouts, seed 123, `display_freq=1`,
final image byte-identical and loss diff **0.0**. Benchmark with previews:
previous-default **0.893 it/s** vs small-preview-reuse **0.968 it/s** (5 warmup
+ 20 timed iterations, 32 cutouts, `display_freq=5`, start/end swap under the
10 GB gate, final image byte-identical). Final-only benchmark remained
byte-identical and did not show a regression in the same harness shape.

**Rejected at parity gate: using aux decode only on preview iterations.** A
more selective variant tried to run the aux-output compiled step only for
non-final preview frames and the scalar-loss step otherwise. It failed the
seeded loss trajectory gate on the tiny generator fixture and on the real
256×256 probe (`loss_max_abs` about 9.5e-3), so the shipped preview-reuse path
continues to use one consistent aux-output step whenever intermediate previews
are enabled.

**Rejected in probe: cached CLIP normalization constants.** Caching the
per-dtype mean/std arrays used by `MlxClip.encode_cutouts` preserved loss and
final image exactly in a short 256² probe, but regressed timing badly:
current **0.958 it/s** vs cached-norm **0.673 it/s** over 10 timed iterations.
No production path changed.

**Enabled thread-local sharpness kernel caching.** The cutout sharpness filter
uses a constant 3×3 per-channel kernel. Rebuilding that tiny MLX array inside
every cutout batch was exact but wasteful, and sharing MLX arrays across
generation threads is unsafe, so the cache is thread-local and materializes the
kernel once per channel count. Benchmark with previews, holding small-preview
reuse enabled on both legs: no-sharpness-cache **0.773 it/s** vs
sharpness-cache **0.982 it/s** (256², 32 cutouts, 5 warmup + 20 timed
iterations, `display_freq=5`, start/end swap under the 10 GB gate, loss diff
**0.0**, final image byte-identical). Final-only regression check was also
byte-identical and slightly positive: **1.028 → 1.036 it/s**. Disable with
`LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE=0` for comparison runs.

**Rejected in benchmark: cached bilinear resize weights.** The cutout resize
path also rebuilds deterministic row/column interpolation matrices, but caching
those MLX arrays per thread was slower in the 256² previewed lane even though
it was exact: no-resize-weight-cache **0.986 it/s** vs resize-weight-cache
**0.931 it/s** (5 warmup + 20 timed iterations, 32 cutouts,
`display_freq=5`, loss diff **0.0**, final image byte-identical). The
candidate was removed from the shipped path.

**Rejected in benchmark: pre-normalized prompt targets.** Precomputing the
normalized text/image target embeddings once per generation preserved exact
loss and final pixels, but did not clear the no-regression bar. The first
previewed 256² run showed a small win (**1.003 → 1.024 it/s**), while the
same unconditional path regressed final-only timing (**0.860 → 0.838 it/s**).
A conditional preview-only rerun then measured slower as well (**0.770 →
0.756 it/s**), with exact parity. The candidate was removed from the shipped
path.

**Rejected in benchmark: single-target prompt-loss specialization.** The common
one-target prompt path can avoid the broadcasted `(cutouts, 1, dim)` distance
tensor, and the specialized formula matched value and gradient bit-for-bit in
unit probes. The previewed 256² benchmark looked strong
(**0.736 → 0.957 it/s**), but the same path regressed final-only timing
(**0.949 → 0.850 it/s**) with exact parity. The candidate was removed from the
shipped path.

**Rejected in benchmark: score-argmax VQ lookup without cached codebook
constants.** A GPT-5.5 Pro report correctly identified that the `x²` term in
`vector_quantize` is constant across codebook entries, so an equivalent
`argmax(x @ codebook.T - 0.5 * ||codebook||²)` lookup is worth testing. The
low-risk in-function rewrite preserved exact output, and the previewed 256²
run improved (**0.724 → 0.862 it/s**), but final-only timing regressed
(**0.990 → 0.963 it/s**). The candidate was removed from the shipped path.
If revisited, test a stronger version with codebook transpose/norm materialized
once per load rather than recomputed inside each quantize call.

**Rejected in benchmark: cached-codebook distance VQ.** The stronger VQ follow-up
kept the exact distance-argmin formula and only precomputed `codebook.T` plus
`||codebook||²` once per generation attempt. Tiny-generator parity and
value/gradient probes were exact, and previewed 256² improved slightly
(**0.732 → 0.756 it/s**), but final-only timing regressed (**0.968 →
0.943 it/s**). The candidate was removed from the shipped path.

**Enabled identity-resize bypass for cutouts already at CLIP size.** The MLX
bilinear resize helper now returns the input directly when a sampled cutout is
already `224×224`, avoiding two identity interpolation matrix builds and two
`einsum`s without changing the math. A value+gradient unit probe matched the
old resize path exactly. Benchmark with previews, holding small-preview reuse
and sharpness kernel caching enabled on both legs: no-identity-skip
**0.928 it/s** vs identity-skip **0.946 it/s** (256², 32 cutouts, 5 warmup +
20 timed iterations, `display_freq=5`, loss diff **0.0**, final image
byte-identical). Final-only regression check was also byte-identical and
positive: **0.945 → 0.971 it/s**. Disable with
`LOCALVQGAN_MLX_RESIZE_IDENTITY_SKIP=0` for comparison runs.

**Rejected at parity probe: manual norm rewrite in prompt loss.** Replacing
`mx.linalg.norm` with explicit `sqrt(sum(x*x))` in the MLX prompt loss matched
forward loss on a controlled fixture, but did not match gradients exactly:
fp32 max gradient delta **1.16e-10** and fp16 max gradient delta **9.5e-7**.
No production path changed.

**Rejected in benchmark: hoisting the resize identity env flag.** Moving
`LOCALVQGAN_MLX_RESIZE_IDENTITY_SKIP` evaluation out of the per-cutout resize
helper preserved exact loss and final pixels, but regressed the final-only 256²
lane badly: per-cutout flag check **0.357 it/s** vs hoisted flag
**0.308 it/s**. The previewed lane was exact and relatively faster
(**0.524 → 0.566 it/s**), but both legs were far below the accepted
identity-resize path and the final-only gate failed. The candidate was removed.

**Rejected in benchmark: selective sharpness for only augmented cutouts.** The
sharpness augmentation leaves about 60 % of cutouts with `factor == 1`, so a
candidate gathered only the actually-sharpened rows, ran the same valid conv on
that smaller batch, and concatenated rows back in original order. Controlled
value+gradient probes and seeded full-cutout probes were exact. The previewed
256² benchmark improved in a paired run (**0.477 → 0.734 it/s**), but the
final-only lane regressed (**0.702 → 0.639 it/s**) with exact loss and final
pixels. The candidate was removed.

**Rejected in benchmark: materializing cutout sizes once.** The cutout spec
generator draws all crop sizes as one MLX vector, then converts each scalar
size to Python while choosing per-cutout offsets. A candidate materialized that
size vector once with NumPy before the offset loop, preserving seeded sizes,
offsets, all later random tensors, cutout output, and input gradients exactly
across multiple seeds. The previewed 256² benchmark improved (**0.664 →
0.754 it/s**), but the final-only lane regressed badly (**0.718 → 0.392 it/s**)
with exact loss and final pixels. The candidate was removed.

**Enabled single-loss sum fast path.** The common one-prompt/no-init path used
`sum([loss], mx.array(0.0))`, adding a scalar zero array to the only loss.
Returning the lone loss directly matched value and gradient exactly in a
controlled probe, while multi-loss cases keep the existing sum order. Benchmark
with previews, holding preview reuse, sharpness kernel caching, and
identity-resize bypass enabled on both legs: list-sum loss **0.794 it/s** vs
single-loss sum **0.907 it/s** (256², 32 cutouts, 5 warmup + 20 timed
iterations, `display_freq=5`, loss diff **0.0**, final image byte-identical).
Final-only regression check was also byte-identical and slightly positive:
**0.949 → 0.952 it/s**. Disable with
`LOCALVQGAN_MLX_SINGLE_LOSS_SUM_FASTPATH=0` for comparison runs.

**Rejected in benchmark: direct single-prompt loss call.** A follow-up tried to
skip the remaining one-element Python `losses` list and call `prompt_loss`
directly when there is exactly one target and no init loss. Controlled
value+gradient probes were exact, and init-loss cases kept the existing
aggregation order. The previewed 256² benchmark regressed anyway:
single-loss list path **0.666 it/s** vs direct prompt loss **0.653 it/s** with
loss diff **0.0** and final image byte-identical. The candidate was removed
without running a final-only claim because it already failed the previewed gate.

**Rejected in benchmark: skipping redundant fp32 cast during image export.** At
256² the decoded preview/final image is already fp32, so a candidate skipped
the defensive `.astype(mx.float32)` in `_to_pil` when the export buffer was
already fp32 while keeping the cast for fp16/bf16 paths. Byte-level image tests
matched exactly for fp32 and the half fallback. The previewed 256² benchmark
improved slightly (**0.953 → 0.967 it/s**), but the final-only lane regressed
(**0.927 → 0.847 it/s**) with exact loss and final pixels. The candidate was
removed.

**Rejected in benchmark: scalar-only small-canvas final loss function.** The
final-only small-canvas path does not need the preview image from
`loss_and_out_fn`, so a candidate split out a scalar-only loss closure for
runs where previews are not reused. A focused slow parity test matched the
tuple-output path exactly, including final pixels. The previewed 256² gate
regressed badly anyway: tuple-loss path **0.996 it/s** vs scalar-only loss
**0.594 it/s**, with exact loss and final pixels, and swap climbed
**7537 → 7912 MB** during the candidate leg. The candidate was removed without
running a final-only claim because it already failed the previewed gate.

**Rejected in benchmark: cached CLIP normalization constants.** A narrower
constant-cache candidate reused CLIP image normalization `mean`/`std` arrays
per dtype/thread instead of constructing them inside every `encode_cutouts`
call. A focused value+gradient probe was exact. The previewed 256² gate
improved (**0.713 → 0.761 it/s**) with exact loss and final pixels, but the
final-only lane regressed (**0.926 → 0.839 it/s**) with the same exact parity.
The candidate was removed.

**Rejected in benchmark: eager materialization of clip bounds.** The per-run
`z_min`/`z_max` bounds used to clip the latent were explicitly `mx.eval`'d once
after deriving them from the codebook, to avoid carrying a lazy min/max graph
into repeated clips. This preserved exact loss and final pixels. The previewed
256² gate improved (**0.728 → 0.867 it/s**), but the final-only lane regressed
(**0.727 → 0.686 it/s**). The candidate was removed.

**Added a canonical MLX benchmark harness.** `tools/mlx_512_benchmark.py`
defaults to the 256² convenience lane and can compare previous-default against
small-canvas candidates such as `small-preview-reuse`, `compiled-optimizer`,
`sharpness-cache`, `resize-identity-skip`, and `single-loss-sum`; at >256² it
runs the large-canvas baseline sync/eager and current-default legs with swap
guards, parity checks, and optional chunk/cache sweeps. The timed leg is
conservative: it includes per-generation compile overhead instead of reporting a
warmed-kernel microbenchmark. Use `--sweep` after the machine is below the swap
threshold to pick a chunk/cache configuration before recording a 512² speed
claim.

Canonical 256² attribution rerun (swap stayed below the 10 GB gate): 256²,
32 cutouts, 5 warmups, 20 samples measured synth forward **279 ms**,
CLIP+cutout backward **573 ms**, synth pullback **685 ms**, two-stage backward
**1256 ms**, full eager `value_and_grad` **1375 ms**. The remaining real
headroom is in decoder pullback and CLIP/cutout backward kernels; micro-probes
that did not preserve exactness or speed, such as weighted cutouts and MLX
FastPatchEmbed, remain rejected.

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

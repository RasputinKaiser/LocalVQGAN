# LocalVQGAN — Design Spec

**Date:** 2026-07-06
**Goal:** Port the VQGAN+CLIP Colab notebook (justinjohn0306/VQGAN-CLIP, "VQGAN+CLIP(Updated).ipynb") to a fast local app with a polished web GUI. Preserve the authentic VQGAN+CLIP aesthetic and full feature set; eliminate the slowness. Cross-platform Python (macOS/Windows/Linux), performance-tuned for Apple Silicon first (target machine: M1 MacBook Pro, 16 GB).

## Why the Colab was slow

1. Cloned repos + pip-installed dependencies every session (minutes of setup per run).
2. 64 CLIP cutout passes per iteration, unbatched.
3. float32 everywhere.
4. CUDA-only code; models reloaded per run.

## Architecture

One Python package, `localvqgan`, in this repo, three layers:

### `localvqgan/pipeline/` — generation engine (pure PyTorch, no web deps)
- Minimal VQGAN model definition: only the encoder/decoder/quantize classes needed to load taming-transformers checkpoints. No taming repo clone.
- CLIP via `open_clip` (pip package, provides OpenAI ViT-B/32 and ViT-B/16 weights).
- Batched cutout + kornia augmentation module.
- `Generator` class: loads a checkpoint + CLIP once, exposes `generate(settings)` as an iterator yielding `(iteration, image, loss)` so any frontend can stream progress.
- Device auto-select: `mps` → `cuda` → `cpu`.

### `localvqgan/server/` — FastAPI app
- Holds one warm `Generator`; models stay in RAM between runs; reload only on checkpoint/CLIP-variant switch.
- Single-job queue (16 GB cannot run concurrent generations).
- REST: start job, cancel job, list/download checkpoints (with progress), gallery history, export timelapse MP4.
- WebSocket: streams preview JPEGs, iteration count, it/s, loss.

### `localvqgan/web/` — static frontend
- Vanilla JS + CSS, no build step, no framework. Served by FastAPI at `http://localhost:8420`.
- Launch UX: one command starts server + opens browser.

### Storage
- Checkpoints: `~/.cache/localvqgan/`, downloaded on demand from the notebook's mirrors, GUI download progress. ImageNet-16384 (~1 GB) is the default.
- Outputs: `outputs/<run-id>/` — final PNG, per-frame PNGs (for timelapse), and a JSON settings sidecar making every result reproducible.

## Pipeline: algorithm and optimizations

Same algorithm as the notebook — Adam optimizing the VQGAN latent against CLIP similarity loss over augmented cutouts — so the aesthetic is unchanged. Optimizations:

1. **fp16 autocast** for CLIP forward and VQGAN decode on MPS and CUDA; latent + optimizer state stay fp32 (stability).
2. **Cutouts default 32** (slider 8–64), fully batched: single interpolate call, kornia augs applied batch-wise.
3. **Prompt/target embeddings computed once** before the loop.
4. **Warm models** — never load per job.
5. **Memory guardrails** — GUI grays out size/settings combinations projected to exceed available memory (~640×640 cap with ImageNet-16384 on 16 GB); OOM during a run is caught and stops the job gracefully.
6. Expected speed on M1: ~1.5–4 it/s at 384×384 (vs ~1 it/s on a free Colab T4), with zero per-run setup.

## Feature parity

Kept (everything from the notebook):
- All checkpoints: ImageNet 1024 & 16384, WikiArt 1024 & 16384, COCO-Stuff, FacesHQ, S-FLCKR, ADE20K, FFHQ, CelebA-HQ, Gumbel-8192.
- Pipe-separated multi-prompts with `prompt:weight` syntax; image prompts with weights; init image; seed; step size; cutout count; ViT-B/32 or B/16; width/height.
- Timelapse MP4 assembled from saved frames at chosen FPS (imageio-ffmpeg).
- **Animation mode** (the notebook's zoom/story video): prompt keyframes over time plus per-frame zoom/pan transforms, rendered as a background job with GUI progress.

Dropped:
- Steganography/XMP metadata embedding (stegano, python-xmp-toolkit, imgtag) — three fragile dependencies; replaced by the JSON settings sidecar.

## GUI

Dark, gallery-style, single page:
- **Left panel:** prompt box with inline syntax hints, checkpoint picker (downloaded vs. downloadable with sizes), size presets + custom, iterations, cutouts, step size, seed, CLIP variant, init/target image drag-and-drop.
- **Center (hero):** live preview streaming as the image evolves; iteration counter, it/s, loss sparkline; Stop and Save-now buttons.
- **Bottom:** session gallery — thumbnails of finished runs; click for full-res, settings JSON, reuse-settings, timelapse export, rendered animation videos.
- **Animation mode:** keyframe timeline; each keyframe holds prompts + zoom/pan values + frame count.

## Error handling

- Startup device probe; clear banner if running on CPU fallback.
- Checkpoint download resume on network failure.
- OOM → graceful job stop with actionable message (smaller size / fewer cutouts).
- WebSocket reconnect: a refreshed tab reattaches to the running job.

## Testing

- Unit: cutout tensor shapes, prompt parsing (pipes/weights), seed reproducibility (same seed + settings ⇒ identical latents on CPU).
- Smoke: 5 iterations at 128×128 on the smallest checkpoint end-to-end.
- API: job lifecycle — start → stream frames → cancel.
- GUI: manual verification via browser preview.

## Non-goals

- MLX backend (possible later optimization, not in scope).
- Multi-user / remote serving; this is localhost-only.
- Steganographic metadata.

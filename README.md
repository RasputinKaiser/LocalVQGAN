# LocalVQGAN

The classic VQGAN+CLIP (the 2021 Colab aesthetic), rebuilt to run fast and
locally with a web GUI. Apple Silicon (MPS), NVIDIA (CUDA), and CPU.

## Quick start

    ./run.sh

Opens http://127.0.0.1:8420. First use downloads the checkpoint you pick
(ImageNet-16384 is the default, ~934 MB, one time) plus CLIP ViT-B/32.

## What's different from the Colab

- No per-session setup: models stay loaded in a local server.
- fp16 autocast on MPS/CUDA; batched cutouts (default 32, not 64).
- MPS-native augmentation set — no CPU-fallback ops in the hot loop.
- Live preview streaming, gallery with reusable settings, timelapse MP4
  export, and keyframed zoom/pan animation mode.
- Every image gets a `settings.json` sidecar for exact reproduction.
- On Apple Silicon, an optional MLX engine (`pip install -e ".[mlx]"`) runs the
  same checkpoints natively; pick the engine in the GUI (auto/mlx/torch).

## Checkpoints

imagenet_1024/16384, gumbel_8192, coco, sflckr, and wikiart_1024/16384
download on demand. faceshq, ade20k, ffhq, and celebahq have no surviving
public mirrors (the original notebook is equally broken for them) — if you
have the files, drop `config.yaml` + `model.ckpt` into
`~/.cache/localvqgan/<name>/` and they'll be picked up.

## Engines

Two generation backends share the same checkpoints and math:

- **torch** — MPS/CUDA/CPU via PyTorch. Always available.
- **mlx** — Apple Silicon only, native Metal via MLX. Optional
  (`pip install -e ".[mlx]"`). Not all checkpoints are supported yet (see
  the capability map in `localvqgan/pipeline/backends/mlx_backend/`).

Pick `torch`, `mlx`, or `auto` per generation in the GUI/API. `auto` uses
mlx only when it's installed, supports the requested checkpoint, and meets
the measured speed gate below; otherwise it falls back to torch.
Auto engine selection uses mlx only up to 256x256 based on this measurement,
and larger sizes automatically fall back to torch; explicit mlx selection has no size limit.

Measured on an Apple M1 (imagenet_16384/ViT-B-32, 32 cutouts, steady state:
5 warmup its + 20 timed its, `torch.set_num_threads(2)`, 2026-07-07):

| size    | torch    | mlx      | ratio (mlx/torch) |
|---------|----------|----------|--------------------|
| 256x256 | 0.680 it/s | 1.021 it/s | 1.50 |
| 384x384 | 0.283 it/s | 0.015 it/s | 0.05 |

The speed gate (`MLX_MEETS_SPEED_GATE`) is evaluated at 256x256/32cut per
spec (ratio >= 1.2 to prefer mlx in `auto`); it currently passes. At 384x384
this M1's unified memory is not enough to keep mlx's working set resident
and it thrashes — torch stays the better choice at that size on this
hardware, which is why the gate is pinned to the 256x256 measurement rather
than a blanket "mlx is faster" claim.

## Dev

    .venv/bin/pytest            # fast suite
    .venv/bin/pytest -m slow    # downloads models, runs real generation

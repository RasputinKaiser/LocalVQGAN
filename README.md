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

## Checkpoints

imagenet_1024/16384, gumbel_8192, coco, sflckr, and wikiart_1024/16384
download on demand. faceshq, ade20k, ffhq, and celebahq have no surviving
public mirrors (the original notebook is equally broken for them) — if you
have the files, drop `config.yaml` + `model.ckpt` into
`~/.cache/localvqgan/<name>/` and they'll be picked up.

## Dev

    .venv/bin/pytest            # fast suite
    .venv/bin/pytest -m slow    # downloads models, runs real generation

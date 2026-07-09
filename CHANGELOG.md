# Changelog

Notable changes to LocalVQGAN. Format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [SemVer](https://semver.org/).

## [0.2.0] - 2026-07-09

First release prepared for PyPI.

### Added
- **Slim Apple-Silicon install**: `pip install localvqgan` on an M-series Mac
  now installs the native MLX engine only — no ~2 GB PyTorch stack. The
  historic torch checkpoints (VQGAN `model.ckpt`, CLIP `pytorch_model.bin`)
  are read by a numpy-only pickle loader whose output is parity-pinned
  byte-for-byte against `torch.load` in the test suite. `localvqgan[torch]`
  adds the torch engine back on any platform.
- **Fast mode on the torch engine** (was MLX-only): the same opt-in
  coarse-to-fine schedule — 60% of iterations at half resolution, latent
  upsample, finish at full res. Includes an automatic fp32 fine-stage
  fallback for an MPS fp16 decoder overflow on upsampled latents.
- CI now covers Apple Silicon (mlx + torch), the torch-free slim install,
  and a wheel build check; releases publish to PyPI from `v*` tags.

### Changed
- The server warms whichever engine `auto` resolves for the default settings
  instead of always warming torch; `/api/system` reports only installed
  engines and the running version.
- Checkpoints with converted MLX weights count as downloaded — the original
  torch ckpt is only fetched when something actually needs it.

### Fixed
- Explicitly selecting the torch engine now always runs a torch generator
  (previously it could return whatever engine the server had warmed first).
- MPS fp16 NaN at the fast-mode stage boundary (decoder overflow on
  upsampled latents) — the fine stage retries with an fp32 VQGAN, keeping
  the finished coarse stage.

## [0.1.0] - 2026-07-08

Initial version: the 2021 VQGAN+CLIP Colab recipe rebuilt as a local web app.
Two engines (PyTorch MPS/CUDA/CPU and native MLX on Apple Silicon) sharing
the same checkpoints and math, live preview streaming, gallery with settings
sidecars, timelapse export, keyframed animation mode, and the performance
work recorded in [PERFORMANCE.md](PERFORMANCE.md) (~40x the faithful port,
with the aesthetic preserved).

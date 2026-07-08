# LocalVQGAN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the VQGAN+CLIP Colab notebook to a fast local Python app (MPS/CUDA/CPU) with a FastAPI backend and a polished no-build-step web GUI, per the spec at `docs/superpowers/specs/2026-07-06-localvqgan-design.md`.

**Architecture:** Three layers in one package: `localvqgan/pipeline/` (pure PyTorch generation engine exposing a frame-yielding iterator), `localvqgan/server/` (FastAPI + WebSocket, warm models, single-job queue), `localvqgan/web/` (vanilla JS/CSS static frontend). Checkpoints cached in `~/.cache/localvqgan/`, outputs in `outputs/<run-id>/` with JSON settings sidecars.

**Tech Stack:** Python ≥3.11, PyTorch ≥2.4, open-clip-torch, kornia, omegaconf, FastAPI + uvicorn, imageio-ffmpeg, pytest + httpx.

## Global Constraints

- Python ≥ 3.11; PyTorch ≥ 2.4 (needed for stable MPS autocast).
- Device selection order: `mps` → `cuda` → `cpu`. Never hard-code a device.
- fp16 autocast applies ONLY to CLIP forward and VQGAN decode; the latent `z` and Adam state stay fp32.
- Default cutouts 32 (range 8–64); default size 384×384; default step size 0.1; default checkpoint `imagenet_16384`; default CLIP `ViT-B-32`.
- Server binds `127.0.0.1:8420` only (localhost-only, no remote serving).
- Frontend: vanilla JS + CSS, no build step, no framework.
- Checkpoint cache dir: `~/.cache/localvqgan/`. Output dir: `./outputs/`.
- No steganography/XMP deps (stegano, python-xmp-toolkit, imgtag are banned). Reproducibility via `settings.json` sidecar.
- Tests that download models or run real generation are marked `@pytest.mark.slow` and excluded by default (`addopts = "-m 'not slow'"`).
- All algorithm math (arcsin distance loss, replace-grad/clamp-with-grad, z clamping to codebook min/max) must match the notebook exactly — the aesthetic is the product.
- This directory is not yet a git repo: Task 1 runs `git init`. Per-task commits are part of this approved plan.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `localvqgan/__init__.py`, `localvqgan/pipeline/__init__.py`, `localvqgan/server/__init__.py`, `localvqgan/web/.gitkeep`, `tests/__init__.py`, `tests/test_import.py`

**Interfaces:**
- Produces: installable `localvqgan` package; `pytest` runs with slow-marker filtering.

- [ ] **Step 1: git init and gitignore**

```bash
cd LocalVQGAN && git init
```

`.gitignore`:
```
.venv/
__pycache__/
*.egg-info/
outputs/
.pytest_cache/
.DS_Store
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[project]
name = "localvqgan"
version = "0.1.0"
description = "Fast local VQGAN+CLIP with a web GUI"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4",
    "torchvision",
    "open-clip-torch>=2.24",
    "kornia>=0.7",
    "einops",
    "omegaconf",
    "numpy",
    "pillow",
    "fastapi",
    "uvicorn[standard]",
    "imageio[ffmpeg]",
    "requests",
    "psutil",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-timeout", "httpx"]

[project.scripts]
localvqgan = "localvqgan.server.main:run"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["localvqgan*"]

[tool.setuptools.package-data]
"localvqgan.web" = ["**/*"]

[tool.pytest.ini_options]
addopts = "-m 'not slow'"
markers = ["slow: downloads models or runs real generation"]
```

- [ ] **Step 3: Package skeleton + import test**

Create empty `__init__.py` files listed above. `tests/test_import.py`:
```python
def test_import():
    import localvqgan  # noqa: F401
```

- [ ] **Step 4: Create venv, install, run test**

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -v
```
Expected: `test_import PASSED`. (Torch install is ~2 GB; one time.)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: scaffold localvqgan package"
```

---

### Task 2: Core utilities — device selection, settings, prompt parsing

**Files:**
- Create: `localvqgan/pipeline/device.py`, `localvqgan/pipeline/settings.py`, `localvqgan/pipeline/prompts.py`
- Test: `tests/test_device.py`, `tests/test_prompts.py`

**Interfaces:**
- Produces: `pick_device(prefer: str | None = None) -> torch.device`; `GenerationSettings` dataclass; `parse_prompts(s: str) -> list[ParsedPrompt]` with `ParsedPrompt(text, weight, stop)`.

- [ ] **Step 1: Write failing tests**

`tests/test_prompts.py`:
```python
from localvqgan.pipeline.prompts import parse_prompts

def test_single_prompt():
    p = parse_prompts("a sunset over the ocean")
    assert len(p) == 1
    assert p[0].text == "a sunset over the ocean"
    assert p[0].weight == 1.0

def test_pipe_and_weights():
    p = parse_prompts("a cat:1.5 | watercolor:0.5")
    assert [x.text for x in p] == ["a cat", "watercolor"]
    assert p[0].weight == 1.5 and p[1].weight == 0.5

def test_weight_and_stop():
    p = parse_prompts("dog:2:-0.5")
    assert p[0].weight == 2.0 and p[0].stop == -0.5

def test_empty_segments_skipped():
    assert len(parse_prompts("a | | b")) == 2
```

`tests/test_device.py`:
```python
import torch
from localvqgan.pipeline.device import pick_device

def test_prefer_overrides():
    assert pick_device("cpu") == torch.device("cpu")

def test_auto_returns_device():
    assert pick_device().type in ("mps", "cuda", "cpu")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_prompts.py tests/test_device.py -v` — Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

`localvqgan/pipeline/device.py`:
```python
import torch


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

`localvqgan/pipeline/prompts.py` (same `text:weight:stop` rsplit semantics as the notebook):
```python
from dataclasses import dataclass


@dataclass
class ParsedPrompt:
    text: str
    weight: float = 1.0
    stop: float = float("-inf")


def parse_prompts(s: str) -> list[ParsedPrompt]:
    out = []
    for chunk in s.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = chunk.rsplit(":", 2)
        # A trailing :number is a weight; two are weight:stop. Bare colons
        # inside text (no numeric suffix) stay part of the text.
        text, weight, stop = vals[0], 1.0, float("-inf")
        try:
            if len(vals) == 3:
                weight, stop = float(vals[1]), float(vals[2])
            elif len(vals) == 2:
                weight = float(vals[1])
        except ValueError:
            text = chunk
        out.append(ParsedPrompt(text.strip(), weight, stop))
    return out
```

`localvqgan/pipeline/settings.py`:
```python
from dataclasses import dataclass, field, asdict


@dataclass
class GenerationSettings:
    prompts: str = ""
    width: int = 384
    height: int = 384
    iterations: int = 300
    cutouts: int = 32
    step_size: float = 0.1
    seed: int = -1
    init_image: str | None = None
    init_weight: float = 0.0
    image_prompts: list[str] = field(default_factory=list)
    checkpoint: str = "imagenet_16384"
    clip_model: str = "ViT-B-32"
    display_freq: int = 5
    precision: str = "auto"  # auto | fp16 | fp32

    def to_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Run tests — Expected: all PASS**

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: device selection, settings, prompt parsing"`

---

### Task 3: Vendored VQGAN model + loader

**Files:**
- Create: `localvqgan/pipeline/vendor/__init__.py`, `localvqgan/pipeline/vendor/taming_model.py`, `localvqgan/pipeline/vendor/taming_quantize.py`, `localvqgan/pipeline/vqgan.py`
- Test: `tests/test_vqgan.py`, fixture `tests/fixtures/tiny_vqgan.yaml`

**Interfaces:**
- Produces: `load_vqgan(config_path: Path, ckpt_path: Path | None, device: torch.device) -> VQGANWrapper`. `VQGANWrapper` has `.model` (VQModel), `.is_gumbel: bool`, `.codebook` property (embedding weight tensor), `.e_dim: int`, `.n_toks: int`, `.f: int` (downsampling factor), `.encode(img_tensor) -> z`, `.decode(z_q) -> tensor`.

- [ ] **Step 1: Vendor upstream model files (MIT-licensed)**

```bash
mkdir -p localvqgan/pipeline/vendor
curl -fsSL https://raw.githubusercontent.com/CompVis/taming-transformers/master/taming/modules/diffusionmodules/model.py -o localvqgan/pipeline/vendor/taming_model.py
curl -fsSL https://raw.githubusercontent.com/CompVis/taming-transformers/master/taming/modules/vqvae/quantize.py -o localvqgan/pipeline/vendor/taming_quantize.py
```

Then edit the two vendored files so they import cleanly without the taming package:
- In `taming_quantize.py`: delete `from taming.modules.util import ...` style imports if present and any classes other than `VectorQuantizer2` and `GumbelQuantize` that require them (keep `VectorQuantizer2` aliased as `VectorQuantizer` if the file does so). If `GumbelQuantize` imports `einsum` from `einops`, keep it — einops is a dependency.
- In `taming_model.py`: this file is self-contained (torch + numpy only); verify with `python -c "from localvqgan.pipeline.vendor import taming_model"`.
- Add a header comment to both: `# Vendored from CompVis/taming-transformers (MIT). Trimmed: no lightning, no training code.`

- [ ] **Step 2: Write failing test with a tiny config**

`tests/fixtures/tiny_vqgan.yaml` (structure mirrors real taming configs; tiny sizes so CPU construction is instant):
```yaml
model:
  target: taming.models.vqgan.VQModel
  params:
    embed_dim: 8
    n_embed: 32
    ddconfig:
      double_z: false
      z_channels: 8
      resolution: 64
      in_channels: 3
      out_ch: 3
      ch: 16
      ch_mult: [1, 2]
      num_res_blocks: 1
      attn_resolutions: []
      dropout: 0.0
```

`tests/test_vqgan.py`:
```python
from pathlib import Path
import torch
from localvqgan.pipeline.vqgan import load_vqgan

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_vqgan.yaml"

def test_roundtrip_shapes():
    w = load_vqgan(FIXTURE, ckpt_path=None, device=torch.device("cpu"))
    assert w.f == 2  # len(ch_mult)=2 -> one downsample
    img = torch.rand(1, 3, 64, 64) * 2 - 1
    z = w.encode(img)
    assert z.shape == (1, 8, 32, 32)
    out = w.decode(z)
    assert out.shape == (1, 3, 64, 64)

def test_codebook():
    w = load_vqgan(FIXTURE, ckpt_path=None, device=torch.device("cpu"))
    assert w.codebook.shape == (32, 8)
    assert w.n_toks == 32 and w.e_dim == 8
```

Run: `.venv/bin/pytest tests/test_vqgan.py -v` — Expected: FAIL (no module `localvqgan.pipeline.vqgan`).

- [ ] **Step 3: Implement `localvqgan/pipeline/vqgan.py`**

```python
from pathlib import Path

import torch
from torch import nn
from omegaconf import OmegaConf

from localvqgan.pipeline.vendor.taming_model import Encoder, Decoder
from localvqgan.pipeline.vendor.taming_quantize import VectorQuantizer2, GumbelQuantize


class VQModel(nn.Module):
    def __init__(self, ddconfig, n_embed, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer2(n_embed, embed_dim, beta=0.25,
                                         remap=None, sane_index_shape=False)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)


class GumbelVQModel(nn.Module):
    def __init__(self, ddconfig, n_embed, embed_dim, kl_weight=1e-8):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = GumbelQuantize(ddconfig["z_channels"], embed_dim,
                                       n_embed, kl_weight=kl_weight, temp_init=1.0)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)


class VQGANWrapper:
    def __init__(self, model: nn.Module, is_gumbel: bool, f: int, device: torch.device):
        self.model = model.eval().requires_grad_(False).to(device)
        self.is_gumbel = is_gumbel
        self.f = f
        self.device = device

    @property
    def codebook(self) -> torch.Tensor:
        q = self.model.quantize
        return q.embed.weight if self.is_gumbel else q.embedding.weight

    @property
    def n_toks(self) -> int:
        return self.codebook.shape[0]

    @property
    def e_dim(self) -> int:
        return self.codebook.shape[1]

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        h = self.model.encoder(img.to(self.device))
        return self.model.quant_conv(h)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.model.decoder(self.model.post_quant_conv(z_q))


def load_vqgan(config_path: Path, ckpt_path: Path | None, device: torch.device) -> VQGANWrapper:
    config = OmegaConf.load(config_path)
    is_gumbel = "Gumbel" in config.model.target
    params = config.model.params
    cls = GumbelVQModel if is_gumbel else VQModel
    kwargs = dict(ddconfig=params.ddconfig, n_embed=params.n_embed, embed_dim=params.embed_dim)
    if is_gumbel and "kl_weight" in params:
        kwargs["kl_weight"] = params.kl_weight
    model = cls(**kwargs)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        # checkpoints include loss-network weights we don't define; ignore them
        model.load_state_dict(sd, strict=False)
    f = 2 ** (len(params.ddconfig.ch_mult) - 1)
    return VQGANWrapper(model, is_gumbel, f, device)
```

Note for implementer: `ddconfig` from OmegaConf must be converted for `**` expansion — use `OmegaConf.to_container(params.ddconfig)` when passing. Adjust the `kwargs` line accordingly:
```python
kwargs = dict(ddconfig=OmegaConf.to_container(params.ddconfig),
              n_embed=params.n_embed, embed_dim=params.embed_dim)
```

- [ ] **Step 4: Run tests — Expected: PASS.** If `GumbelQuantize.__init__` signature differs in the vendored file (arg names vary by commit), match the vendored signature — the vendored file is the source of truth.

- [ ] **Step 5: Commit** — `git commit -am "feat: vendored VQGAN model and loader"`

---

### Task 4: Checkpoint registry + downloader with resume

**Files:**
- Create: `localvqgan/pipeline/checkpoints.py`
- Test: `tests/test_checkpoints.py`

**Interfaces:**
- Produces: `CHECKPOINTS: dict[str, CheckpointSpec]` (`CheckpointSpec(name, config_url, ckpt_url, size_mb)`); `cache_dir() -> Path`; `checkpoint_paths(name) -> tuple[Path, Path]` (config, ckpt); `is_downloaded(name) -> bool`; `download(name, progress_cb: Callable[[str, int, int], None]) -> None` (cb args: name, bytes_done, bytes_total); resumes partial downloads via HTTP Range.

- [ ] **Step 1: Extract the checkpoint URLs from the notebook**

```bash
curl -fsSL "https://raw.githubusercontent.com/justinjohn0306/VQGAN-CLIP/main/VQGAN%2BCLIP(Updated).ipynb" -o /tmp/vqgan_nb.ipynb
python3 - <<'EOF'
import json, re
nb = json.load(open('/tmp/vqgan_nb.ipynb'))
src = "\n".join("".join(c["source"]) for c in nb["cells"])
for m in sorted(set(re.findall(r'https?://[^\s\'"\\)]+\.(?:ckpt|yaml)[^\s\'"\\)]*', src))):
    print(m)
EOF
```

Build `CHECKPOINTS` from the printed URLs, covering ALL spec checkpoints: `imagenet_1024`, `imagenet_16384`, `wikiart_1024`, `wikiart_16384`, `coco`, `faceshq`, `sflckr`, `ade20k`, `ffhq`, `celebahq`, `gumbel_8192`. Then verify each URL is alive (`curl -sIL -o /dev/null -w "%{http_code} " <url>`); for any dead mirror, search the nerdyrodent/VQGAN-CLIP `download_models.sh` for an alternate and note the substitution in a comment.

- [ ] **Step 2: Write failing tests (no network — monkeypatched)**

`tests/test_checkpoints.py`:
```python
from pathlib import Path
import localvqgan.pipeline.checkpoints as cp

def test_registry_complete():
    for name in ["imagenet_1024", "imagenet_16384", "wikiart_1024", "wikiart_16384",
                 "coco", "faceshq", "sflckr", "ade20k", "ffhq", "celebahq", "gumbel_8192"]:
        assert name in cp.CHECKPOINTS

def test_paths_and_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "cache_dir", lambda: tmp_path)
    cfg, ckpt = cp.checkpoint_paths("imagenet_16384")
    assert not cp.is_downloaded("imagenet_16384")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("x"); ckpt.write_bytes(b"x")
    assert cp.is_downloaded("imagenet_16384")

def test_download_resumes(tmp_path, monkeypatch):
    calls = []
    def fake_stream(url, dest, cb):
        calls.append((url, dest.name))
        dest.write_bytes(b"data")
    monkeypatch.setattr(cp, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(cp, "_stream_to_file", fake_stream)
    cp.download("imagenet_16384", progress_cb=lambda *a: None)
    assert len(calls) == 2  # config + ckpt
```

Run: FAIL (module missing).

- [ ] **Step 3: Implement `localvqgan/pipeline/checkpoints.py`**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    config_url: str
    ckpt_url: str
    size_mb: int


# URLs extracted from justinjohn0306/VQGAN-CLIP notebook (Step 1); verify liveness.
CHECKPOINTS: dict[str, CheckpointSpec] = {
    # e.g. "imagenet_16384": CheckpointSpec("imagenet_16384", "<config_url>", "<ckpt_url>", 980),
    # ... one entry per spec checkpoint, filled from Step 1 output ...
}


def cache_dir() -> Path:
    return Path.home() / ".cache" / "localvqgan"


def checkpoint_paths(name: str) -> tuple[Path, Path]:
    d = cache_dir() / name
    return d / "config.yaml", d / "model.ckpt"


def is_downloaded(name: str) -> bool:
    cfg, ckpt = checkpoint_paths(name)
    return cfg.exists() and ckpt.exists() and ckpt.stat().st_size > 0


def _stream_to_file(url: str, dest: Path, cb: Callable[[int, int], None]) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, headers=headers, timeout=30) as r:
        if r.status_code == 416:  # already complete
            part.rename(dest)
            return
        r.raise_for_status()
        if r.status_code != 206:
            done = 0  # server ignored Range; restart
        total = done + int(r.headers.get("content-length", 0))
        mode = "ab" if done else "wb"
        with open(part, mode) as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                cb(done, total)
    part.rename(dest)


def download(name: str, progress_cb: Callable[[str, int, int], None]) -> None:
    spec = CHECKPOINTS[name]
    cfg, ckpt = checkpoint_paths(name)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.exists():
        _stream_to_file(spec.config_url, cfg, lambda d, t: None)
    if not (ckpt.exists() and ckpt.stat().st_size > 0):
        _stream_to_file(spec.ckpt_url, ckpt, lambda d, t: progress_cb(name, d, t))
```

The `CHECKPOINTS` dict MUST be fully populated with the real URLs from Step 1 — the commented example is scaffolding shape, not the deliverable.

- [ ] **Step 4: Run tests — Expected: PASS.**

- [ ] **Step 5: Manual liveness spot-check** — `python -c` snippet to HEAD each URL, print status. Any 404 → substitute mirror per Step 1.

- [ ] **Step 6: Commit** — `git commit -am "feat: checkpoint registry and resumable downloader"`

---

### Task 5: Batched cutouts + augmentations

**Files:**
- Create: `localvqgan/pipeline/cutouts.py`
- Test: `tests/test_cutouts.py`

**Interfaces:**
- Produces: `MakeCutouts(nn.Module)` — `__init__(cut_size: int, cutn: int, cut_pow: float = 1.0)`, `forward(input: Tensor[1,3,H,W]) -> Tensor[cutn,3,cut_size,cut_size]`. Augs applied to the whole batch in one call (this is the vectorization win).

- [ ] **Step 1: Failing tests**

`tests/test_cutouts.py`:
```python
import torch
from localvqgan.pipeline.cutouts import MakeCutouts

def test_output_shape():
    mc = MakeCutouts(cut_size=224, cutn=8)
    out = mc(torch.rand(1, 3, 384, 384))
    assert out.shape == (8, 3, 224, 224)

def test_small_input_still_works():
    mc = MakeCutouts(cut_size=224, cutn=4)
    out = mc(torch.rand(1, 3, 128, 128))
    assert out.shape == (4, 3, 224, 224)

def test_gradients_flow():
    mc = MakeCutouts(cut_size=32, cutn=4)
    x = torch.rand(1, 3, 64, 64, requires_grad=True)
    mc(x).sum().backward()
    assert x.grad is not None
```

Run: FAIL.

- [ ] **Step 2: Implement `localvqgan/pipeline/cutouts.py`**

```python
import torch
from torch import nn
from torch.nn import functional as F
import kornia.augmentation as K


class MakeCutouts(nn.Module):
    def __init__(self, cut_size: int, cutn: int, cut_pow: float = 1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        side_y, side_x = input.shape[2:4]
        max_size = min(side_x, side_y)
        min_size = min(side_x, side_y, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            ox = int(torch.randint(0, side_x - size + 1, ()))
            oy = int(torch.randint(0, side_y - size + 1, ()))
            cut = input[:, :, oy:oy + size, ox:ox + size]
            cutouts.append(F.adaptive_avg_pool2d(cut, self.cut_size))
        batch = self.augs(torch.cat(cutouts))
        if self.noise_fac:
            facs = batch.new_empty([batch.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch
```

- [ ] **Step 3: Run tests — Expected: PASS**

- [ ] **Step 4: Commit** — `git commit -am "feat: batched cutouts with kornia augmentations"`

---

### Task 6: CLIP wrapper + prompt loss modules

**Files:**
- Create: `localvqgan/pipeline/clip_guide.py`
- Test: `tests/test_clip_guide.py`

**Interfaces:**
- Produces: `replace_grad`, `clamp_with_grad`, `vector_quantize(x, codebook)`; `Prompt(nn.Module)` with `__init__(embed: Tensor, weight: float, stop: float)` and `forward(image_embeds: Tensor) -> Tensor` (scalar loss); `ClipGuide` with `__init__(model_name: str, device)`, `.cut_size: int`, `.embed_text(s: str) -> Tensor`, `.embed_image(pil_img) -> Tensor`, `.encode_cutouts(batch: Tensor) -> Tensor` (normalizes with CLIP mean/std internally, returns fp32).

- [ ] **Step 1: Failing tests (fast tests use fake embeds — no CLIP download)**

`tests/test_clip_guide.py`:
```python
import pytest
import torch
from localvqgan.pipeline.clip_guide import Prompt, vector_quantize, clamp_with_grad

def test_prompt_loss_scalar_and_grad():
    embed = torch.randn(1, 512)
    p = Prompt(embed, weight=1.0, stop=float("-inf"))
    x = torch.randn(8, 512, requires_grad=True)
    loss = p(x)
    assert loss.dim() == 0
    loss.backward()
    assert x.grad is not None

def test_negative_weight_flips_sign():
    embed = torch.randn(1, 512)
    x = torch.randn(4, 512)
    a = Prompt(embed, 1.0, float("-inf"))(x)
    b = Prompt(embed, -1.0, float("-inf"))(x)
    assert torch.sign(a) != torch.sign(b) or a == 0

def test_vector_quantize_snaps_to_codebook():
    codebook = torch.eye(4)
    x = torch.tensor([[[0.9, 0.1, 0.0, 0.0]]])
    q = vector_quantize(x, codebook)
    assert torch.allclose(q, torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))

def test_clamp_with_grad_range():
    x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    y = clamp_with_grad(x, 0.0, 1.0)
    assert y.min() >= 0 and y.max() <= 1

@pytest.mark.slow
def test_real_clip_text_embed():
    from localvqgan.pipeline.clip_guide import ClipGuide
    g = ClipGuide("ViT-B-32", torch.device("cpu"))
    e = g.embed_text("a photograph of a cat")
    assert e.shape[-1] == 512
```

Run fast set: FAIL.

- [ ] **Step 2: Implement `localvqgan/pipeline/clip_guide.py`**

```python
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import open_clip


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed: torch.Tensor, weight: float = 1.0, stop: float = float("-inf")):
        super().__init__()
        self.register_buffer("embed", embed)
        self.register_buffer("weight", torch.as_tensor(weight))
        self.register_buffer("stop", torch.as_tensor(stop))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


_CLIP_NORM = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                  std=(0.26862954, 0.26130258, 0.27577711))


class ClipGuide:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai")
        self.model = self.model.eval().requires_grad_(False).to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.cut_size = self.model.visual.image_size[0]

    def embed_text(self, s: str) -> torch.Tensor:
        toks = self.tokenizer([s]).to(self.device)
        return self.model.encode_text(toks).float()

    def embed_image(self, pil_img) -> torch.Tensor:
        t = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        return self.model.encode_image(t).float()

    def encode_cutouts(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(_CLIP_NORM(batch)).float()
```

- [ ] **Step 3: Run fast tests — Expected: PASS.** Optionally run the slow one once: `.venv/bin/pytest tests/test_clip_guide.py -m slow -v` (downloads ViT-B/32, ~350 MB).

- [ ] **Step 4: Commit** — `git commit -am "feat: CLIP guide and notebook-faithful loss modules"`

---

### Task 7: Generator core loop

**Files:**
- Create: `localvqgan/pipeline/generator.py`
- Test: `tests/test_generator.py`

**Interfaces:**
- Consumes: everything from Tasks 2–6.
- Produces: `FrameUpdate` dataclass (`iteration: int`, `total: int`, `image: PIL.Image.Image | None`, `loss: float`); `Generator` with `__init__(device: torch.device | None = None)`, `.load(checkpoint: str, clip_model: str)` (no-op if already loaded; uses `checkpoints.checkpoint_paths`), `.load_from_paths(config_path, ckpt_path, clip_model)` (test seam), `.generate(settings: GenerationSettings, cancel: threading.Event | None = None) -> Iterator[FrameUpdate]`. Image is populated every `display_freq` iterations and on the final iteration; other updates carry `image=None`.

- [ ] **Step 1: Failing tests (tiny random VQGAN + real CLIP on CPU, marked slow; pure-logic test fast)**

`tests/test_generator.py`:
```python
from pathlib import Path
import threading
import pytest
import torch
from localvqgan.pipeline.settings import GenerationSettings
from localvqgan.pipeline.generator import Generator

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_vqgan.yaml"

@pytest.mark.slow
def test_seed_reproducibility_cpu():
    def run():
        g = Generator(torch.device("cpu"))
        g.load_from_paths(FIXTURE, None, "ViT-B-32")
        s = GenerationSettings(prompts="a red square", width=64, height=64,
                               iterations=3, cutouts=4, seed=42, display_freq=1)
        return [u for u in g.generate(s)]
    a, b = run(), run()
    assert len(a) == len(b) == 3
    assert list(a[-1].image.getdata()) == list(b[-1].image.getdata())

@pytest.mark.slow
def test_cancel_stops_early():
    g = Generator(torch.device("cpu"))
    g.load_from_paths(FIXTURE, None, "ViT-B-32")
    cancel = threading.Event()
    s = GenerationSettings(prompts="x", width=64, height=64, iterations=100,
                           cutouts=4, seed=1, display_freq=1)
    seen = 0
    for u in g.generate(s, cancel=cancel):
        seen += 1
        if seen == 2:
            cancel.set()
    assert seen < 100
```

- [ ] **Step 2: Implement `localvqgan/pipeline/generator.py`**

```python
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import optim
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image

from localvqgan.pipeline import checkpoints
from localvqgan.pipeline.clip_guide import ClipGuide, Prompt, vector_quantize, clamp_with_grad
from localvqgan.pipeline.cutouts import MakeCutouts
from localvqgan.pipeline.device import pick_device
from localvqgan.pipeline.prompts import parse_prompts
from localvqgan.pipeline.settings import GenerationSettings
from localvqgan.pipeline.vqgan import load_vqgan, VQGANWrapper


@dataclass
class FrameUpdate:
    iteration: int
    total: int
    image: Image.Image | None
    loss: float


class GenerationOOM(RuntimeError):
    pass


class Generator:
    def __init__(self, device: torch.device | None = None):
        self.device = device or pick_device()
        self.vqgan: VQGANWrapper | None = None
        self.clip: ClipGuide | None = None
        self._loaded: tuple[str, str] | None = None

    def load(self, checkpoint: str, clip_model: str) -> None:
        if self._loaded == (checkpoint, clip_model):
            return
        cfg, ckpt = checkpoints.checkpoint_paths(checkpoint)
        self.load_from_paths(cfg, ckpt, clip_model)
        self._loaded = (checkpoint, clip_model)

    def load_from_paths(self, config_path: Path, ckpt_path: Path | None, clip_model: str) -> None:
        self.vqgan = load_vqgan(config_path, ckpt_path, self.device)
        if self.clip is None or self.clip.model_name != clip_model:
            self.clip = ClipGuide(clip_model, self.device)
        self._loaded = None

    def _autocast(self, precision: str):
        use_fp16 = precision == "fp16" or (precision == "auto" and self.device.type in ("mps", "cuda"))
        if use_fp16:
            return torch.autocast(device_type=self.device.type, dtype=torch.float16)
        return nullcontext()

    def _synth(self, z: torch.Tensor) -> torch.Tensor:
        z_q = vector_quantize(z.movedim(1, 3), self.vqgan.codebook).movedim(3, 1)
        return clamp_with_grad(self.vqgan.decode(z_q).add(1).div(2), 0, 1)

    def _init_z(self, s: GenerationSettings) -> torch.Tensor:
        f = self.vqgan.f
        toks_x, toks_y = s.width // f, s.height // f
        if s.init_image:
            img = Image.open(s.init_image).convert("RGB").resize(
                (toks_x * f, toks_y * f), Image.LANCZOS)
            t = TF.to_tensor(img).unsqueeze(0).to(self.device) * 2 - 1
            return self.vqgan.encode(t)
        one_hot = F.one_hot(
            torch.randint(self.vqgan.n_toks, [toks_y * toks_x], device=self.device),
            self.vqgan.n_toks).float()
        z = one_hot @ self.vqgan.codebook
        return z.view([-1, toks_y, toks_x, self.vqgan.e_dim]).permute(0, 3, 1, 2)

    @staticmethod
    def _to_pil(out: torch.Tensor) -> Image.Image:
        arr = out[0].detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr)

    def generate(self, s: GenerationSettings,
                 cancel: threading.Event | None = None) -> Iterator[FrameUpdate]:
        assert self.vqgan is not None and self.clip is not None, "call load() first"
        seed = s.seed if s.seed >= 0 else int(torch.randint(0, 2**31 - 1, ()).item())
        torch.manual_seed(seed)

        codebook = self.vqgan.codebook
        z_min = codebook.min(dim=0).values[None, :, None, None]
        z_max = codebook.max(dim=0).values[None, :, None, None]

        z = self._init_z(s)
        z_orig = z.clone()
        z.requires_grad_(True)
        opt = optim.Adam([z], lr=s.step_size)

        make_cutouts = MakeCutouts(self.clip.cut_size, s.cutouts).to(self.device)
        prompt_modules = [
            Prompt(self.clip.embed_text(p.text), p.weight, p.stop).to(self.device)
            for p in parse_prompts(s.prompts)
        ]
        for path in s.image_prompts:
            embed = self.clip.embed_image(Image.open(path).convert("RGB"))
            prompt_modules.append(Prompt(embed, 1.0, float("-inf")).to(self.device))

        precision = s.precision
        for i in range(1, s.iterations + 1):
            if cancel is not None and cancel.is_set():
                return
            try:
                opt.zero_grad(set_to_none=True)
                with self._autocast(precision):
                    out = self._synth(z)
                    embeds = self.clip.encode_cutouts(make_cutouts(out))
                losses = [pm(embeds) for pm in prompt_modules]
                if s.init_weight:
                    losses.append(F.mse_loss(z, z_orig) * s.init_weight / 2)
                loss = sum(losses)
                loss.backward()
                opt.step()
                with torch.inference_mode():
                    z.copy_(z.maximum(z_min).minimum(z_max))
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    raise GenerationOOM(
                        "Out of memory — try a smaller size or fewer cutouts.") from e
                if precision != "fp32" and i == 1:
                    precision = "fp32"  # fp16 unsupported for some op; retry in fp32
                    continue
                raise
            want_image = i % s.display_freq == 0 or i == s.iterations
            img = self._to_pil(self._synth(z)) if want_image else None
            yield FrameUpdate(i, s.iterations, img, float(loss))
```

Add `self.model_name = model_name` in `ClipGuide.__init__` (Task 6 file) — `generate`'s reload check uses it.

- [ ] **Step 3: Run** `.venv/bin/pytest tests/test_generator.py -m slow -v` — Expected: both PASS (first run downloads CLIP once). Fast suite still green: `.venv/bin/pytest -v`.

- [ ] **Step 4: Commit** — `git commit -am "feat: generator core loop with fp16 autocast and cancel"`

---

### Task 8: Run outputs — frames, sidecar, timelapse MP4

**Files:**
- Create: `localvqgan/pipeline/outputs.py`
- Test: `tests/test_outputs.py`

**Interfaces:**
- Produces: `RunWriter` — `__init__(outputs_root: Path, settings: GenerationSettings)`, attrs `.run_id: str` (e.g. `run-0001`, next free index), `.dir: Path`; methods `.save_frame(iteration: int, img: PIL.Image)` (PNG to `frames/%05d.png`), `.save_final(img)` (`final.png`), `.write_sidecar(extra: dict | None = None)` (`settings.json` = settings dict + seed/extra), `.make_timelapse(fps: int) -> Path` (`timelapse.mp4` via imageio-ffmpeg from frames/ in order). Also `list_runs(outputs_root) -> list[dict]` (run_id, has final, settings, has timelapse) sorted newest-first.

- [ ] **Step 1: Failing tests**

```python
from pathlib import Path
from PIL import Image
from localvqgan.pipeline.outputs import RunWriter, list_runs
from localvqgan.pipeline.settings import GenerationSettings

def _img():
    return Image.new("RGB", (64, 64), (200, 30, 30))

def test_run_lifecycle(tmp_path):
    w = RunWriter(tmp_path, GenerationSettings(prompts="x"))
    for i in range(1, 4):
        w.save_frame(i, _img())
    w.save_final(_img())
    w.write_sidecar({"seed_used": 42})
    mp4 = w.make_timelapse(fps=10)
    assert mp4.exists() and mp4.stat().st_size > 0
    runs = list_runs(tmp_path)
    assert runs[0]["run_id"] == w.run_id
    assert runs[0]["settings"]["prompts"] == "x"

def test_run_ids_increment(tmp_path):
    a = RunWriter(tmp_path, GenerationSettings())
    b = RunWriter(tmp_path, GenerationSettings())
    assert a.run_id != b.run_id
```

Run: FAIL.

- [ ] **Step 2: Implement `localvqgan/pipeline/outputs.py`**

```python
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from localvqgan.pipeline.settings import GenerationSettings


class RunWriter:
    def __init__(self, outputs_root: Path, settings: GenerationSettings):
        outputs_root.mkdir(parents=True, exist_ok=True)
        existing = sorted(p.name for p in outputs_root.glob("run-*"))
        nxt = int(existing[-1].split("-")[1]) + 1 if existing else 1
        self.run_id = f"run-{nxt:04d}"
        self.dir = outputs_root / self.run_id
        (self.dir / "frames").mkdir(parents=True)
        self.settings = settings

    def save_frame(self, iteration: int, img: Image.Image) -> None:
        img.save(self.dir / "frames" / f"{iteration:05d}.png")

    def save_final(self, img: Image.Image) -> None:
        img.save(self.dir / "final.png")

    def write_sidecar(self, extra: dict | None = None) -> None:
        data = self.settings.to_dict() | (extra or {})
        (self.dir / "settings.json").write_text(json.dumps(data, indent=2))

    def make_timelapse(self, fps: int = 30) -> Path:
        out = self.dir / "timelapse.mp4"
        frames = sorted((self.dir / "frames").glob("*.png"))
        with imageio.get_writer(out, fps=fps, codec="libx264",
                                pixelformat="yuv420p") as w:
            for f in frames:
                w.append_data(np.asarray(Image.open(f)))
        return out


def list_runs(outputs_root: Path) -> list[dict]:
    runs = []
    for d in sorted(outputs_root.glob("run-*"), reverse=True):
        sidecar = d / "settings.json"
        runs.append({
            "run_id": d.name,
            "final": (d / "final.png").exists(),
            "timelapse": (d / "timelapse.mp4").exists(),
            "settings": json.loads(sidecar.read_text()) if sidecar.exists() else {},
        })
    return runs
```

(x264 requires even dimensions; frames from VQGAN are multiples of 16, always even.)

- [ ] **Step 3: Run tests — Expected: PASS**

- [ ] **Step 4: Commit** — `git commit -am "feat: run outputs, sidecar, timelapse"`

---

### Task 9: Animation engine (keyframed zoom/pan)

**Files:**
- Create: `localvqgan/pipeline/animation.py`
- Test: `tests/test_animation.py`

**Interfaces:**
- Consumes: `Generator` internals (`_synth`, `_init_z` pattern), `VQGANWrapper.encode`.
- Produces: `Keyframe` dataclass (`prompts: str`, `frames: int`, `zoom: float = 1.0`, `pan_x: int = 0`, `pan_y: int = 0`, `iterations_per_frame: int = 8`); `render_animation(generator, base_settings: GenerationSettings, keyframes: list[Keyframe], writer: RunWriter, cancel, progress_cb: Callable[[int, int, PIL.Image | None], None]) -> None`. Algorithm: run N optimization iterations per output frame; save frame; apply zoom/pan to the frame image (PIL affine crop-resize); re-encode transformed image to `z` (like the notebook's zoom video mode); switch prompt set at each keyframe boundary.

- [ ] **Step 1: Failing test (slow, tiny model)**

```python
from pathlib import Path
import pytest
import torch
from localvqgan.pipeline.animation import Keyframe, render_animation, transform_image
from localvqgan.pipeline.generator import Generator
from localvqgan.pipeline.outputs import RunWriter
from localvqgan.pipeline.settings import GenerationSettings
from PIL import Image

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_vqgan.yaml"

def test_transform_zoom_keeps_size():
    img = Image.new("RGB", (64, 64))
    out = transform_image(img, zoom=1.05, pan_x=2, pan_y=0)
    assert out.size == (64, 64)

@pytest.mark.slow
def test_animation_produces_frames(tmp_path):
    g = Generator(torch.device("cpu"))
    g.load_from_paths(FIXTURE, None, "ViT-B-32")
    s = GenerationSettings(width=64, height=64, cutouts=4, seed=7)
    w = RunWriter(tmp_path, s)
    kfs = [Keyframe(prompts="a forest", frames=2, zoom=1.02, iterations_per_frame=2),
           Keyframe(prompts="a city", frames=2, zoom=1.02, iterations_per_frame=2)]
    seen = []
    render_animation(g, s, kfs, w, cancel=None,
                     progress_cb=lambda done, total, img: seen.append((done, total)))
    assert len(list((w.dir / "frames").glob("*.png"))) == 4
    assert seen[-1] == (4, 4)
```

Run fast test: FAIL.

- [ ] **Step 2: Implement `localvqgan/pipeline/animation.py`**

```python
import threading
from dataclasses import dataclass
from typing import Callable

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from localvqgan.pipeline.generator import Generator
from localvqgan.pipeline.outputs import RunWriter
from localvqgan.pipeline.settings import GenerationSettings


@dataclass
class Keyframe:
    prompts: str
    frames: int
    zoom: float = 1.0
    pan_x: int = 0
    pan_y: int = 0
    iterations_per_frame: int = 8


def transform_image(img: Image.Image, zoom: float, pan_x: int, pan_y: int) -> Image.Image:
    w, h = img.size
    cw, ch = int(w / zoom), int(h / zoom)
    left = (w - cw) // 2 + pan_x
    top = (h - ch) // 2 + pan_y
    left = max(0, min(w - cw, left))
    top = max(0, min(h - ch, top))
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)


def render_animation(g: Generator, base: GenerationSettings, keyframes: list[Keyframe],
                     writer: RunWriter, cancel: threading.Event | None,
                     progress_cb: Callable[[int, int, Image.Image | None], None]) -> None:
    total = sum(k.frames for k in keyframes)
    done = 0
    current: Image.Image | None = None
    if base.init_image:
        current = Image.open(base.init_image).convert("RGB")
    for kf in keyframes:
        for _ in range(kf.frames):
            if cancel is not None and cancel.is_set():
                return
            s = GenerationSettings(**{**base.to_dict(),
                                      "prompts": kf.prompts,
                                      "iterations": kf.iterations_per_frame,
                                      "display_freq": kf.iterations_per_frame,
                                      "seed": base.seed if done == 0 else -1,
                                      "init_image": None})
            if current is not None:
                s.init_image = _stash(writer, current)
            last = None
            for update in g.generate(s, cancel=cancel):
                last = update
            if last is None or last.image is None:
                return
            done += 1
            writer.save_frame(done, last.image)
            progress_cb(done, total, last.image)
            current = transform_image(last.image, kf.zoom, kf.pan_x, kf.pan_y)
    writer.save_final(current)
    writer.write_sidecar({"animation": [k.__dict__ for k in keyframes]})


def _stash(writer: RunWriter, img: Image.Image) -> str:
    p = writer.dir / "_carry.png"
    img.save(p)
    return str(p)
```

- [ ] **Step 3: Run** fast test then `pytest tests/test_animation.py -m slow -v` — Expected: PASS.

- [ ] **Step 4: Commit** — `git commit -am "feat: keyframed zoom/pan animation engine"`

---

### Task 10: Server — job manager and REST API

**Files:**
- Create: `localvqgan/server/jobs.py`, `localvqgan/server/app.py`, `localvqgan/server/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Generator`, `RunWriter`, `list_runs`, `checkpoints`, `render_animation`.
- Produces:
  - `JobManager` — `__init__(generator_factory: Callable[[], Generator], outputs_root: Path)`; `.start_still(settings_dict) -> str` / `.start_animation(settings_dict, keyframes: list[dict]) -> str` (returns run_id; raises `Busy`); `.status() -> dict` (`state: idle|running|done|error`, run_id, iteration, total, its_per_sec, loss, error); `.cancel()`; `.subscribe() -> asyncio.Queue` / `.unsubscribe(q)`; `.latest_preview: bytes | None` (JPEG).
  - `create_app(manager: JobManager) -> FastAPI` with routes: `GET /api/status`, `POST /api/jobs` (body: `{"type": "still"|"animation", "settings": {...}, "keyframes": [...]}`, 409 if busy), `POST /api/jobs/cancel`, `GET /api/checkpoints` (name, size_mb, downloaded), `POST /api/checkpoints/{name}/download` (background thread, progress via WS), `GET /api/gallery`, `GET /api/gallery/{run_id}/final.png`, `GET /api/gallery/{run_id}/settings.json`, `GET /api/gallery/{run_id}/timelapse.mp4?fps=30` (builds on demand then serves), `GET /api/system` (device, total_ram_gb, max_recommended_side), static mount of `localvqgan/web` at `/`.
  - `main.run()` — uvicorn on `127.0.0.1:8420`, opens browser after startup.

- [ ] **Step 1: Failing API tests with a fake generator (no models)**

`tests/test_api.py`:
```python
import time
from pathlib import Path
import threading
from fastapi.testclient import TestClient
from PIL import Image
from localvqgan.pipeline.generator import FrameUpdate
from localvqgan.server.jobs import JobManager
from localvqgan.server.app import create_app


class FakeGenerator:
    device = type("D", (), {"type": "cpu"})()
    def load(self, checkpoint, clip_model):
        pass
    def generate(self, settings, cancel=None):
        for i in range(1, 6):
            if cancel is not None and cancel.is_set():
                return
            time.sleep(0.01)
            yield FrameUpdate(i, 5, Image.new("RGB", (32, 32)), 0.5)


def make_client(tmp_path):
    mgr = JobManager(lambda: FakeGenerator(), tmp_path)
    return TestClient(create_app(mgr)), mgr


def _wait_idle(client, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get("/api/status").json()
        if st["state"] in ("done", "error", "idle"):
            return st
        time.sleep(0.05)
    raise TimeoutError


def test_job_lifecycle(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/api/jobs", json={"type": "still", "settings": {"prompts": "x", "iterations": 5}})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    st = _wait_idle(client)
    assert st["state"] == "done"
    assert client.get(f"/api/gallery/{run_id}/final.png").status_code == 200
    gallery = client.get("/api/gallery").json()
    assert gallery[0]["run_id"] == run_id

def test_busy_returns_409(tmp_path):
    client, _ = make_client(tmp_path)
    client.post("/api/jobs", json={"type": "still", "settings": {"prompts": "x"}})
    r2 = client.post("/api/jobs", json={"type": "still", "settings": {"prompts": "y"}})
    assert r2.status_code == 409
    _wait_idle(client)

def test_cancel(tmp_path):
    client, _ = make_client(tmp_path)
    client.post("/api/jobs", json={"type": "still", "settings": {"prompts": "x", "iterations": 5}})
    assert client.post("/api/jobs/cancel").status_code == 200
    st = _wait_idle(client)
    assert st["state"] in ("done", "idle")

def test_system_info(tmp_path):
    client, _ = make_client(tmp_path)
    info = client.get("/api/system").json()
    assert "device" in info and "max_recommended_side" in info
```

Run: FAIL.

- [ ] **Step 2: Implement `localvqgan/server/jobs.py`**

```python
import asyncio
import io
import threading
import time
from pathlib import Path
from typing import Callable

from localvqgan.pipeline.animation import Keyframe, render_animation
from localvqgan.pipeline.generator import Generator, GenerationOOM
from localvqgan.pipeline.outputs import RunWriter
from localvqgan.pipeline.settings import GenerationSettings


class Busy(Exception):
    pass


class JobManager:
    def __init__(self, generator_factory: Callable[[], Generator], outputs_root: Path):
        self._factory = generator_factory
        self._generator: Generator | None = None
        self.outputs_root = Path(outputs_root)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._subs: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.latest_preview: bytes | None = None
        self._state = {"state": "idle"}

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = self._factory()
        return self._generator

    def status(self) -> dict:
        return dict(self._state)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def _publish(self, msg: dict) -> None:
        self._state = {**self._state, **{k: v for k, v in msg.items() if k != "image_jpeg"}}
        if self._loop is None:
            return
        def push():
            for q in list(self._subs):
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(msg)
        self._loop.call_soon_threadsafe(push)

    def _begin(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise Busy()
            self._cancel = threading.Event()

    def start_still(self, settings_dict: dict) -> str:
        self._begin()
        settings = GenerationSettings(**settings_dict)
        writer = RunWriter(self.outputs_root, settings)
        self._thread = threading.Thread(
            target=self._run_still, args=(settings, writer), daemon=True)
        self._thread.start()
        return writer.run_id

    def start_animation(self, settings_dict: dict, keyframes: list[dict]) -> str:
        self._begin()
        settings = GenerationSettings(**settings_dict)
        writer = RunWriter(self.outputs_root, settings)
        kfs = [Keyframe(**k) for k in keyframes]
        self._thread = threading.Thread(
            target=self._run_animation, args=(settings, writer, kfs), daemon=True)
        self._thread.start()
        return writer.run_id

    def cancel(self) -> None:
        self._cancel.set()

    def _preview_jpeg(self, img) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        self.latest_preview = buf.getvalue()
        return self.latest_preview

    def _run_still(self, settings: GenerationSettings, writer: RunWriter) -> None:
        try:
            self._publish({"state": "running", "run_id": writer.run_id,
                           "iteration": 0, "total": settings.iterations, "phase": "loading"})
            self.generator.load(settings.checkpoint, settings.clip_model)
            t0, last_img = time.time(), None
            for u in self.generator.generate(settings, cancel=self._cancel):
                msg = {"state": "running", "run_id": writer.run_id, "phase": "generating",
                       "iteration": u.iteration, "total": u.total, "loss": u.loss,
                       "its_per_sec": round(u.iteration / max(time.time() - t0, 1e-6), 2)}
                if u.image is not None:
                    writer.save_frame(u.iteration, u.image)
                    last_img = u.image
                    msg["image_jpeg"] = self._preview_jpeg(u.image)
                self._publish(msg)
            if last_img is not None:
                writer.save_final(last_img)
            writer.write_sidecar()
            self._publish({"state": "done", "run_id": writer.run_id})
        except GenerationOOM as e:
            self._publish({"state": "error", "error": str(e)})
        except Exception as e:  # surface, don't kill the server
            self._publish({"state": "error", "error": f"{type(e).__name__}: {e}"})

    def _run_animation(self, settings: GenerationSettings, writer: RunWriter,
                       kfs: list) -> None:
        try:
            self._publish({"state": "running", "run_id": writer.run_id,
                           "iteration": 0, "total": sum(k.frames for k in kfs),
                           "phase": "loading"})
            self.generator.load(settings.checkpoint, settings.clip_model)
            def cb(done, total, img):
                msg = {"state": "running", "run_id": writer.run_id, "phase": "animating",
                       "iteration": done, "total": total}
                if img is not None:
                    msg["image_jpeg"] = self._preview_jpeg(img)
                self._publish(msg)
            render_animation(self.generator, settings, kfs, writer, self._cancel, cb)
            self._publish({"state": "done", "run_id": writer.run_id})
        except Exception as e:
            self._publish({"state": "error", "error": f"{type(e).__name__}: {e}"})
```

- [ ] **Step 3: Implement `localvqgan/server/app.py`**

```python
import base64
import threading
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from localvqgan.pipeline import checkpoints
from localvqgan.pipeline.outputs import list_runs
from localvqgan.server.jobs import Busy, JobManager

WEB_DIR = Path(__file__).parent.parent / "web"


def _max_side(total_ram_gb: float) -> int:
    if total_ram_gb >= 32:
        return 896
    if total_ram_gb >= 16:
        return 640
    return 448


def create_app(manager: JobManager) -> FastAPI:
    app = FastAPI(title="LocalVQGAN")

    @app.on_event("startup")
    async def _startup():
        import asyncio
        manager.attach_loop(asyncio.get_running_loop())

    @app.get("/api/status")
    def status():
        return manager.status()

    @app.post("/api/jobs")
    def start_job(body: dict):
        try:
            if body.get("type") == "animation":
                run_id = manager.start_animation(body.get("settings", {}),
                                                 body.get("keyframes", []))
            else:
                run_id = manager.start_still(body.get("settings", {}))
        except Busy:
            raise HTTPException(409, "A job is already running")
        except TypeError as e:
            raise HTTPException(422, f"Bad settings: {e}")
        return {"run_id": run_id}

    @app.post("/api/jobs/cancel")
    def cancel():
        manager.cancel()
        return {"ok": True}

    @app.get("/api/checkpoints")
    def list_checkpoints():
        return [{"name": s.name, "size_mb": s.size_mb,
                 "downloaded": checkpoints.is_downloaded(s.name)}
                for s in checkpoints.CHECKPOINTS.values()]

    @app.post("/api/checkpoints/{name}/download")
    def download_checkpoint(name: str):
        if name not in checkpoints.CHECKPOINTS:
            raise HTTPException(404)
        def cb(n, done, total):
            manager._publish({"type": "download", "name": n,
                              "done": done, "total": total})
        threading.Thread(target=checkpoints.download, args=(name, cb),
                         daemon=True).start()
        return {"ok": True}

    @app.get("/api/gallery")
    def gallery():
        return list_runs(manager.outputs_root)

    @app.get("/api/gallery/{run_id}/final.png")
    def final_png(run_id: str):
        p = manager.outputs_root / run_id / "final.png"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/api/gallery/{run_id}/settings.json")
    def sidecar(run_id: str):
        p = manager.outputs_root / run_id / "settings.json"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/api/gallery/{run_id}/timelapse.mp4")
    def timelapse(run_id: str, fps: int = 30):
        d = manager.outputs_root / run_id
        if not d.exists():
            raise HTTPException(404)
        out = d / "timelapse.mp4"
        if not out.exists():
            from localvqgan.pipeline.outputs import RunWriter
            w = RunWriter.__new__(RunWriter)
            w.dir = d
            w.make_timelapse(fps=fps)
        return FileResponse(out, media_type="video/mp4")

    @app.get("/api/system")
    def system():
        ram_gb = psutil.virtual_memory().total / 2**30
        return {"device": manager.generator.device.type,
                "total_ram_gb": round(ram_gb, 1),
                "max_recommended_side": _max_side(ram_gb)}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        snapshot = manager.status()
        if manager.latest_preview:
            snapshot["image_b64"] = base64.b64encode(manager.latest_preview).decode()
        await sock.send_json(snapshot)
        q = manager.subscribe()
        try:
            while True:
                msg = dict(await q.get())
                if "image_jpeg" in msg:
                    msg["image_b64"] = base64.b64encode(msg.pop("image_jpeg")).decode()
                await sock.send_json(msg)
        except WebSocketDisconnect:
            pass
        finally:
            manager.unsubscribe(q)

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
```

Note: `/api/system` touching `manager.generator` would eagerly build the real Generator (loads nothing, just picks device) — fine for the fake too.

- [ ] **Step 4: Implement `localvqgan/server/main.py`**

```python
import threading
import webbrowser
from pathlib import Path

import uvicorn

from localvqgan.pipeline.generator import Generator
from localvqgan.server.app import create_app
from localvqgan.server.jobs import JobManager


def run() -> None:
    manager = JobManager(Generator, Path.cwd() / "outputs")
    app = create_app(manager)
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8420")).start()
    uvicorn.run(app, host="127.0.0.1", port=8420)
```

Placeholder `localvqgan/web/index.html` so StaticFiles mounts (real UI is Task 12):
```html
<!doctype html><title>LocalVQGAN</title><p>UI coming in Task 12</p>
```

- [ ] **Step 5: Run** `.venv/bin/pytest tests/test_api.py -v` — Expected: all PASS.

- [ ] **Step 6: Commit** — `git commit -am "feat: FastAPI server with job manager"`

---

### Task 11: WebSocket streaming test + reattach

**Files:**
- Modify: `tests/test_api.py` (add WS tests)

**Interfaces:**
- Consumes: Task 10's `/ws` endpoint.

- [ ] **Step 1: Write failing/verifying WS tests** (append to `tests/test_api.py`):

```python
def test_ws_snapshot_on_connect(tmp_path):
    client, _ = make_client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["state"] == "idle"

def test_ws_streams_frames_and_reattach(tmp_path):
    client, mgr = make_client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # snapshot
        client.post("/api/jobs", json={"type": "still",
                                       "settings": {"prompts": "x", "iterations": 5,
                                                    "display_freq": 1}})
        got_image = False
        for _ in range(20):
            msg = ws.receive_json()
            if msg.get("image_b64"):
                got_image = True
            if msg.get("state") == "done":
                break
        assert got_image
    # reattach after job: snapshot carries last preview
    with client.websocket_connect("/ws") as ws2:
        snap = ws2.receive_json()
        assert snap["state"] == "done" and snap.get("image_b64")
```

- [ ] **Step 2: Run** — Expected: PASS with Task 10 code (FakeGenerator's `display_freq` handling: FrameUpdate always carries an image, so previews flow). If `test_ws_streams_frames_and_reattach` flakes on queue backpressure, raise queue maxsize to 16 in `subscribe()`.

- [ ] **Step 3: Commit** — `git commit -am "test: websocket streaming and reattach"`

---

### Task 12: Frontend — layout, controls, live preview

**Files:**
- Create: `localvqgan/web/index.html`, `localvqgan/web/style.css`, `localvqgan/web/app.js` (replaces Task 10 placeholder)

**Interfaces:**
- Consumes: all `/api/*` routes and `/ws` messages from Tasks 10–11. WS message fields: `state`, `phase`, `iteration`, `total`, `loss`, `its_per_sec`, `image_b64`, `error`, and `{type:"download", name, done, total}`.

- [ ] **Step 1: `index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LocalVQGAN</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="layout">
  <aside class="panel">
    <h1>LocalVQGAN</h1>
    <label>Prompts <span class="hint">pipe-separated, weight with :1.5</span>
      <textarea id="prompts" rows="3" placeholder="an enchanted forest | trending on artstation:0.5"></textarea>
    </label>
    <label>Checkpoint <select id="checkpoint"></select></label>
    <div id="ckpt-download" class="hidden">
      <button id="download-btn">Download checkpoint</button>
      <progress id="download-progress" max="100" value="0" class="hidden"></progress>
    </div>
    <div class="row">
      <label>Width <input id="width" type="number" value="384" step="16"></label>
      <label>Height <input id="height" type="number" value="384" step="16"></label>
    </div>
    <div class="row">
      <label>Iterations <input id="iterations" type="number" value="300"></label>
      <label>Cutouts <input id="cutouts" type="number" value="32" min="8" max="64"></label>
    </div>
    <div class="row">
      <label>Step size <input id="step_size" type="number" value="0.1" step="0.01"></label>
      <label>Seed <input id="seed" type="number" value="-1"></label>
    </div>
    <label>CLIP <select id="clip_model">
      <option>ViT-B-32</option><option>ViT-B-16</option>
    </select></label>
    <label>Init image <input id="init_image" type="file" accept="image/*"></label>
    <label>Image prompt <input id="image_prompt" type="file" accept="image/*"></label>
    <div id="size-warning" class="warn hidden"></div>
    <button id="generate" class="primary">Generate</button>
    <button id="stop" class="hidden">Stop</button>
    <div id="sysinfo" class="hint"></div>
  </aside>
  <main class="stage">
    <div class="preview-wrap">
      <img id="preview" alt="">
      <div id="idle-hint">Enter a prompt and hit Generate</div>
    </div>
    <div class="statusbar">
      <span id="progress-text"></span>
      <span id="speed"></span>
      <canvas id="loss-spark" width="160" height="28"></canvas>
      <span id="error" class="warn"></span>
    </div>
  </main>
</div>
<section id="gallery-section"><h2>Gallery</h2><div id="gallery"></div></section>
<script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: `style.css`** — dark gallery aesthetic, complete file:

```css
:root {
  --bg: #101014; --panel: #17171d; --line: #26262e;
  --text: #e8e6e0; --dim: #8a8894; --accent: #d8a24a;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 -apple-system, "Segoe UI", sans-serif; }
.layout { display: grid; grid-template-columns: 320px 1fr; min-height: 82vh; }
.panel { background: var(--panel); border-right: 1px solid var(--line);
  padding: 20px; display: flex; flex-direction: column; gap: 12px; }
h1 { font-size: 18px; letter-spacing: 0.06em; margin: 0 0 8px; color: var(--accent); }
label { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--dim); }
textarea, input, select { background: var(--bg); color: var(--text);
  border: 1px solid var(--line); border-radius: 6px; padding: 8px; font: inherit; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.hint { color: var(--dim); font-size: 11px; }
button { border: 1px solid var(--line); background: var(--panel); color: var(--text);
  border-radius: 6px; padding: 10px; cursor: pointer; font: inherit; }
button.primary { background: var(--accent); color: #191919; font-weight: 600; }
button:disabled { opacity: 0.4; cursor: default; }
.warn { color: #e07a5f; font-size: 12px; }
.hidden { display: none !important; }
.stage { display: flex; flex-direction: column; }
.preview-wrap { flex: 1; display: grid; place-items: center; padding: 24px; position: relative; }
#preview { max-width: 100%; max-height: 70vh; border-radius: 8px;
  box-shadow: 0 8px 40px rgba(0,0,0,0.5); }
#idle-hint { position: absolute; color: var(--dim); }
.statusbar { display: flex; align-items: center; gap: 16px; padding: 10px 24px;
  border-top: 1px solid var(--line); color: var(--dim); font-size: 12px; }
#loss-spark { opacity: 0.8; }
#gallery-section { padding: 20px 24px; border-top: 1px solid var(--line); }
#gallery-section h2 { font-size: 14px; color: var(--dim); letter-spacing: 0.08em; }
#gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; }
#gallery .card { position: relative; cursor: pointer; }
#gallery img { width: 100%; border-radius: 6px; display: block; }
#gallery .card .actions { display: flex; gap: 6px; margin-top: 4px; }
#gallery .card .actions a, #gallery .card .actions button {
  font-size: 11px; padding: 3px 6px; }
```

- [ ] **Step 3: `app.js`** — complete file:

```javascript
const $ = (id) => document.getElementById(id);
const losses = [];
let uploadedInit = null, uploadedPrompt = null, sysinfo = null;

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return r.json();
}

async function loadCheckpoints() {
  const cks = await api("/api/checkpoints");
  const sel = $("checkpoint");
  sel.innerHTML = "";
  for (const c of cks) {
    const o = document.createElement("option");
    o.value = c.name;
    o.textContent = `${c.name}${c.downloaded ? "" : ` (download ${c.size_mb} MB)`}`;
    o.dataset.downloaded = c.downloaded;
    sel.appendChild(o);
  }
  updateDownloadUI();
}

function updateDownloadUI() {
  const opt = $("checkpoint").selectedOptions[0];
  $("ckpt-download").classList.toggle("hidden", !opt || opt.dataset.downloaded === "true");
}

async function loadSystem() {
  sysinfo = await api("/api/system");
  $("sysinfo").textContent =
    `device: ${sysinfo.device} · ram: ${sysinfo.total_ram_gb} GB · max side: ${sysinfo.max_recommended_side}px`;
  checkSize();
}

function checkSize() {
  if (!sysinfo) return;
  const w = +$("width").value, h = +$("height").value, m = sysinfo.max_recommended_side;
  const over = w > m || h > m;
  $("size-warning").classList.toggle("hidden", !over);
  $("size-warning").textContent = over ? `Above ${m}px this machine will likely run out of memory.` : "";
  $("generate").disabled = over;
}

async function fileToDataURL(input) {
  const f = input.files[0];
  if (!f) return null;
  return new Promise((res) => {
    const rd = new FileReader();
    rd.onload = () => res(rd.result);
    rd.readAsDataURL(f);
  });
}

async function uploadIfAny() {
  // send data URLs; server writes temp files (see /api/upload note in Task 13)
  uploadedInit = await fileToDataURL($("init_image"));
  uploadedPrompt = await fileToDataURL($("image_prompt"));
}

function settingsFromForm() {
  return {
    prompts: $("prompts").value,
    width: +$("width").value, height: +$("height").value,
    iterations: +$("iterations").value, cutouts: +$("cutouts").value,
    step_size: +$("step_size").value, seed: +$("seed").value,
    checkpoint: $("checkpoint").value, clip_model: $("clip_model").value,
    display_freq: 5,
  };
}

$("generate").onclick = async () => {
  try {
    await uploadIfAny();
    const settings = settingsFromForm();
    const body = { type: "still", settings };
    if (uploadedInit) body.init_image_data = uploadedInit;
    if (uploadedPrompt) body.image_prompt_data = uploadedPrompt;
    losses.length = 0;
    await api("/api/jobs", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    $("generate").classList.add("hidden");
    $("stop").classList.remove("hidden");
    $("error").textContent = "";
  } catch (e) { $("error").textContent = e.message; }
};

$("stop").onclick = () => api("/api/jobs/cancel", { method: "POST" });
$("download-btn").onclick = async () => {
  await api(`/api/checkpoints/${$("checkpoint").value}/download`, { method: "POST" });
  $("download-progress").classList.remove("hidden");
};
$("checkpoint").onchange = updateDownloadUI;
$("width").oninput = checkSize;
$("height").oninput = checkSize;

function drawSpark() {
  const c = $("loss-spark"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (losses.length < 2) return;
  const min = Math.min(...losses), max = Math.max(...losses), span = max - min || 1;
  ctx.strokeStyle = "#d8a24a";
  ctx.beginPath();
  losses.forEach((v, i) => {
    const x = (i / (losses.length - 1)) * c.width;
    const y = c.height - ((v - min) / span) * (c.height - 4) - 2;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

function onMessage(msg) {
  if (msg.type === "download") {
    const p = $("download-progress");
    p.classList.remove("hidden");
    p.value = msg.total ? (100 * msg.done / msg.total) : 0;
    if (msg.total && msg.done >= msg.total) { p.classList.add("hidden"); loadCheckpoints(); }
    return;
  }
  if (msg.image_b64) {
    $("preview").src = "data:image/jpeg;base64," + msg.image_b64;
    $("idle-hint").style.display = "none";
  }
  if (msg.loss !== undefined) { losses.push(msg.loss); drawSpark(); }
  if (msg.iteration !== undefined)
    $("progress-text").textContent = `${msg.phase || ""} ${msg.iteration}/${msg.total}`;
  if (msg.its_per_sec !== undefined) $("speed").textContent = `${msg.its_per_sec} it/s`;
  if (msg.error) $("error").textContent = msg.error;
  if (msg.state === "done" || msg.state === "error" || msg.state === "idle") {
    $("generate").classList.remove("hidden");
    $("stop").classList.add("hidden");
    if (msg.state === "done") loadGallery();
  }
}

function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connectWS, 1500);
}

async function loadGallery() {
  const runs = await api("/api/gallery");
  const g = $("gallery");
  g.innerHTML = "";
  for (const r of runs) {
    if (!r.final) continue;
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <img src="/api/gallery/${r.run_id}/final.png" loading="lazy">
      <div class="actions">
        <a href="/api/gallery/${r.run_id}/final.png" download>PNG</a>
        <a href="/api/gallery/${r.run_id}/timelapse.mp4">MP4</a>
        <a href="/api/gallery/${r.run_id}/settings.json" target="_blank">JSON</a>
        <button data-run="${r.run_id}" class="reuse">Reuse</button>
      </div>`;
    card.querySelector(".reuse").onclick = async (ev) => {
      const s = await api(`/api/gallery/${ev.target.dataset.run}/settings.json`);
      for (const k of ["prompts", "width", "height", "iterations", "cutouts",
                       "step_size", "seed", "clip_model"])
        if (s[k] !== undefined && $(k)) $(k).value = s[k];
      if (s.checkpoint) $("checkpoint").value = s.checkpoint;
      window.scrollTo({ top: 0, behavior: "smooth" });
    };
    g.appendChild(card);
  }
}

loadCheckpoints(); loadSystem(); loadGallery(); connectWS();
```

- [ ] **Step 4: Manual verification** — `.venv/bin/localvqgan` (or `uvicorn` invocation) then open `http://127.0.0.1:8420`; verify layout renders, checkpoint list populates, size warning triggers above the cap. Use the preview browser tools for a screenshot check.

- [ ] **Step 5: Commit** — `git commit -am "feat: web UI with live preview, controls, gallery"`

---

### Task 13: Image uploads + animation UI + gallery polish

**Files:**
- Modify: `localvqgan/server/app.py` (decode data-URL uploads in `POST /api/jobs`), `localvqgan/web/index.html`, `localvqgan/web/app.js`
- Test: `tests/test_api.py` (upload decoding)

**Interfaces:**
- Consumes: Task 10 job start; Task 9 `Keyframe` fields.
- Produces: `POST /api/jobs` accepts optional `init_image_data` / `image_prompt_data` (data URLs); server writes them to `outputs/_uploads/` and sets `settings["init_image"]` / appends to `settings["image_prompts"]`. Animation panel in UI posts `{"type": "animation", "settings": {...}, "keyframes": [...]}`.

- [ ] **Step 1: Failing test** (append to `tests/test_api.py`):

```python
import base64 as b64

def _data_url():
    from io import BytesIO
    buf = BytesIO()
    Image.new("RGB", (16, 16), (0, 255, 0)).save(buf, format="PNG")
    return "data:image/png;base64," + b64.b64encode(buf.getvalue()).decode()

def test_init_image_upload_decoded(tmp_path):
    client, mgr = make_client(tmp_path)
    r = client.post("/api/jobs", json={"type": "still",
        "settings": {"prompts": "x", "iterations": 5},
        "init_image_data": _data_url()})
    assert r.status_code == 200
    _wait_idle(client)
    uploads = list((tmp_path / "_uploads").glob("*.png"))
    assert len(uploads) == 1
```

Run: FAIL.

- [ ] **Step 2: Implement upload decoding in `app.py`** — replace the body of `start_job`:

```python
    @app.post("/api/jobs")
    def start_job(body: dict):
        settings = body.get("settings", {})
        try:
            for key, target in (("init_image_data", "init_image"),
                                ("image_prompt_data", "image_prompts")):
                data_url = body.get(key)
                if not data_url:
                    continue
                header, b64data = data_url.split(",", 1)
                updir = manager.outputs_root / "_uploads"
                updir.mkdir(parents=True, exist_ok=True)
                n = len(list(updir.glob("*"))) + 1
                p = updir / f"upload-{n:04d}.png"
                p.write_bytes(base64.b64decode(b64data))
                if target == "image_prompts":
                    settings.setdefault("image_prompts", []).append(str(p))
                else:
                    settings[target] = str(p)
            if body.get("type") == "animation":
                run_id = manager.start_animation(settings, body.get("keyframes", []))
            else:
                run_id = manager.start_still(settings)
        except Busy:
            raise HTTPException(409, "A job is already running")
        except (TypeError, ValueError) as e:
            raise HTTPException(422, f"Bad request: {e}")
        return {"run_id": run_id}
```

- [ ] **Step 3: Animation UI** — add below the Generate/Stop buttons in `index.html`:

```html
    <details id="anim">
      <summary>Animation mode</summary>
      <div id="keyframes"></div>
      <button id="add-kf">+ keyframe</button>
      <button id="animate" class="primary">Render animation</button>
    </details>
```

Append to `app.js`:

```javascript
function kfRow(kf = { prompts: "", frames: 30, zoom: 1.02, pan_x: 0, pan_y: 0,
                     iterations_per_frame: 8 }) {
  const div = document.createElement("div");
  div.className = "kf";
  div.innerHTML = `
    <textarea class="kf-prompts" rows="2" placeholder="prompts for this keyframe">${kf.prompts}</textarea>
    <div class="row">
      <label>Frames <input class="kf-frames" type="number" value="${kf.frames}"></label>
      <label>Iters/frame <input class="kf-ipf" type="number" value="${kf.iterations_per_frame}"></label>
    </div>
    <div class="row">
      <label>Zoom <input class="kf-zoom" type="number" step="0.01" value="${kf.zoom}"></label>
      <label>Pan x/y <span><input class="kf-px" type="number" value="${kf.pan_x}" style="width:45%">
        <input class="kf-py" type="number" value="${kf.pan_y}" style="width:45%"></span></label>
    </div>
    <button class="kf-del">remove</button><hr>`;
  div.querySelector(".kf-del").onclick = () => div.remove();
  return div;
}

$("add-kf").onclick = () => $("keyframes").appendChild(kfRow());

$("animate").onclick = async () => {
  const keyframes = [...document.querySelectorAll("#keyframes .kf")].map((d) => ({
    prompts: d.querySelector(".kf-prompts").value,
    frames: +d.querySelector(".kf-frames").value,
    iterations_per_frame: +d.querySelector(".kf-ipf").value,
    zoom: +d.querySelector(".kf-zoom").value,
    pan_x: +d.querySelector(".kf-px").value,
    pan_y: +d.querySelector(".kf-py").value,
  }));
  if (!keyframes.length) { $("error").textContent = "Add at least one keyframe"; return; }
  try {
    await uploadIfAny();
    const body = { type: "animation", settings: settingsFromForm(), keyframes };
    if (uploadedInit) body.init_image_data = uploadedInit;
    await api("/api/jobs", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    $("generate").classList.add("hidden");
    $("stop").classList.remove("hidden");
  } catch (e) { $("error").textContent = e.message; }
};
$("keyframes").appendChild(kfRow());
```

And `style.css` addition:
```css
#anim { border-top: 1px solid var(--line); padding-top: 10px; }
.kf textarea { width: 100%; }
```

- [ ] **Step 4: Run tests + manual check** — `.venv/bin/pytest tests/test_api.py -v` PASS; browser: keyframe rows add/remove, animation job posts.

- [ ] **Step 5: Commit** — `git commit -am "feat: uploads and animation UI"`

---

### Task 14: Launcher polish, README, end-to-end smoke

**Files:**
- Create: `run.sh`, `README.md`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: everything.

- [ ] **Step 1: `run.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -e ".[dev]"
fi
exec .venv/bin/localvqgan
```
`chmod +x run.sh`

- [ ] **Step 2: End-to-end smoke test (slow; real checkpoint + CLIP)**

`tests/test_smoke.py`:
```python
import pytest
import torch
from localvqgan.pipeline import checkpoints
from localvqgan.pipeline.generator import Generator
from localvqgan.pipeline.settings import GenerationSettings

@pytest.mark.slow
def test_end_to_end_small():
    name = "imagenet_16384"
    if not checkpoints.is_downloaded(name):
        checkpoints.download(name, progress_cb=lambda *a: None)
    g = Generator()
    g.load(name, "ViT-B-32")
    s = GenerationSettings(prompts="a matte painting of a lighthouse at dusk",
                           width=128, height=128, iterations=5, cutouts=8,
                           seed=123, display_freq=5)
    frames = list(g.generate(s))
    assert frames[-1].image is not None
    assert frames[-1].image.size == (128, 128)
```

Run: `.venv/bin/pytest tests/test_smoke.py -m slow -v` — Expected: PASS on MPS (first run downloads ~1 GB checkpoint). Record the observed it/s in the commit message.

- [ ] **Step 3: `README.md`**

```markdown
# LocalVQGAN

The classic VQGAN+CLIP (the 2021 Colab aesthetic), rebuilt to run fast and
locally with a web GUI. Apple Silicon (MPS), NVIDIA (CUDA), and CPU.

## Quick start

    ./run.sh

Opens http://127.0.0.1:8420. First use downloads the checkpoint you pick
(ImageNet-16384 is the default, ~1 GB, one time) plus CLIP ViT-B/32.

## What's different from the Colab

- No per-session setup: models stay loaded in a local server.
- fp16 autocast on MPS/CUDA; batched cutouts (default 32, not 64).
- Live preview streaming, gallery with reusable settings, timelapse MP4
  export, and keyframed zoom/pan animation mode.
- Every image gets a `settings.json` sidecar for exact reproduction.

## Dev

    .venv/bin/pytest            # fast suite
    .venv/bin/pytest -m slow    # downloads models, runs real generation
```

- [ ] **Step 4: Full fast suite green** — `.venv/bin/pytest -v` — Expected: all PASS.

- [ ] **Step 5: Manual GUI acceptance** — `./run.sh`; generate a 384×384 still on `imagenet_16384`; confirm live preview updates, stop works, gallery card appears with PNG/MP4/JSON/Reuse; render a 10-frame animation; confirm MP4 plays.

- [ ] **Step 6: Commit** — `git commit -am "feat: launcher, README, smoke test"`

---

## Self-review notes (already applied)

- **Spec coverage:** all spec sections map to tasks — pipeline (2–7), checkpoints/downloads (4), outputs/timelapse (8), animation (9), server/WS/OOM/reattach (10–11), GUI incl. guardrails + gallery + animation timeline (12–13), launcher/README/smoke (14). Steganography intentionally absent per spec.
- **Type consistency:** `FrameUpdate`, `GenerationSettings`, `Keyframe`, WS message fields, and route paths are used with identical names across tasks.
- **Known risk:** checkpoint mirror URLs rot; Task 4 Steps 1/5 make liveness verification an explicit deliverable, with nerdyrodent's `download_models.sh` as the fallback source.
- **Known deviation risk:** vendored taming files vary slightly by commit; Task 3 declares the vendored file the source of truth for constructor signatures.

import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import torch
from PIL import Image
from torch import optim
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from localvqgan.pipeline import checkpoints
from localvqgan.pipeline.clip_guide import ClipGuide, Prompt, clamp_with_grad, vector_quantize
from localvqgan.pipeline.cutouts import MakeCutouts
from localvqgan.pipeline.device import pick_device
from localvqgan.pipeline.prompts import parse_prompts
from localvqgan.pipeline.settings import GenerationSettings
from localvqgan.pipeline.vqgan import VQGANWrapper, load_vqgan


# Above this canvas, checkpoint the decoder backward (recompute activations)
# so 512²+ stops swap-dying / OOM-ing on ≤16GB Macs and low-VRAM CUDA. 384²
# and below keep the full-activation fast path (they fit, and checkpointing
# would only add recompute cost). Exact math either way — outputs unchanged.
TORCH_DECODE_CHECKPOINT_MIN_PIXELS = 384 * 384

# Coarse-to-fine ("Fast mode"): same recipe as the MLX engine (see
# mlx_backend/generator.py) — spend this fraction of the iteration budget at
# half resolution, upsample the latent 2x, finish at full res. Opt-in only:
# it changes the exact pixels for a given seed.
FAST_COARSE_FRACTION = 0.6
FAST_MIN_COARSE_TOKENS = 8


@dataclass
class FrameUpdate:
    iteration: int
    total: int
    image: Image.Image | None
    loss: float | None


class GenerationOOM(RuntimeError):
    pass


class _NonFiniteLoss(RuntimeError):
    """Signals a non-finite loss at a given iteration so the coarse-to-fine
    driver can retry the whole run in fp32 (matching the pristine path)."""

    def __init__(self, iteration: int):
        super().__init__("non-finite loss")
        self.iteration = iteration


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

    def load_from_paths(self, config_path: Path, ckpt_path: Path | None,
                        clip_model: str) -> None:
        self.vqgan = load_vqgan(config_path, ckpt_path, self.device)
        if self.clip is None or self.clip.model_name != clip_model:
            self.clip = ClipGuide(clip_model, self.device)
        self._loaded = None

    def _model_dtype(self, precision: str) -> torch.dtype:
        use_fp16 = (
            self.device.type in ("mps", "cuda")
            and precision in ("auto", "fp16")
        )
        return torch.float16 if use_fp16 else torch.float32

    def _set_model_dtype(self, precision: str) -> None:
        dtype = self._model_dtype(precision)
        self.vqgan.set_dtype(dtype)
        self.clip.set_dtype(dtype)

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

    # --- Coarse-to-fine ("Fast mode") -------------------------------------
    def _fast_mode_eligible(self, s: GenerationSettings) -> bool:
        if not getattr(s, "fast_mode", False) or self.vqgan is None:
            return False
        f = self.vqgan.f
        ftx, fty = s.width // f, s.height // f
        if ftx % 2 or fty % 2:
            return False  # need an exact 2x coarse->fine token grid
        return (ftx // 2 >= FAST_MIN_COARSE_TOKENS
                and fty // 2 >= FAST_MIN_COARSE_TOKENS)

    def _fast_coarse_dims(self, s: GenerationSettings) -> tuple[int, int, int]:
        f = self.vqgan.f
        coarse_w = (s.width // f // 2) * f
        coarse_h = (s.height // f // 2) * f
        n_coarse = round(FAST_COARSE_FRACTION * s.iterations)
        n_coarse = max(1, min(s.iterations - 1, n_coarse))
        return coarse_w, coarse_h, n_coarse

    def _build_prompt_modules(self, s: GenerationSettings) -> list[Prompt]:
        prompt_modules = [
            Prompt(self.clip.embed_text(p.text), p.weight, p.stop).to(self.device)
            for p in parse_prompts(s.prompts)
        ]
        for path in s.image_prompts:
            embed = self.clip.embed_image(Image.open(path).convert("RGB"))
            prompt_modules.append(Prompt(embed, 1.0, float("-inf")).to(self.device))
        return prompt_modules

    def _optimize_stage(self, stage_s: GenerationSettings,
                        prompt_modules: list[Prompt], z: torch.Tensor,
                        z_min: torch.Tensor, z_max: torch.Tensor,
                        n_iters: int, global_start: int, total: int,
                        cancel: threading.Event | None):
        """Optimize z for n_iters at the resolution encoded in stage_s, yielding
        FrameUpdates numbered from global_start. Returns the final z via the
        generator return. Decoder checkpointing follows the stage canvas, so a
        512² fine stage checkpoints while its 256² coarse stage does not."""
        self.vqgan.set_decoder_checkpointing(
            stage_s.width * stage_s.height > TORCH_DECODE_CHECKPOINT_MIN_PIXELS)
        z = z.detach().clone().float()  # clone: opt.step() must not mutate the
        # caller's tensor (the driver reuses the upsampled latent on fp32 retry)
        z_orig = z.clone()  # init_weight anchor for this stage (default 0 -> unused)
        z.requires_grad_(True)
        opt = optim.Adam([z], lr=stage_s.step_size)
        make_cutouts = MakeCutouts(self.clip.cut_size, stage_s.cutouts,
                                   stage_s.cut_pow,
                                   device_type=self.device.type).to(self.device)
        for k in range(n_iters):
            if cancel is not None and cancel.is_set():
                return z
            i = global_start + k
            try:
                opt.zero_grad(set_to_none=True)
                out = self._synth(z)
                embeds = self.clip.encode_cutouts(make_cutouts(out))
                losses = [pm(embeds) for pm in prompt_modules]
                if stage_s.init_weight:
                    losses.append(F.mse_loss(z, z_orig) * stage_s.init_weight / 2)
                loss = sum(losses)
                if not bool(torch.isfinite(loss.detach()).item()):
                    raise _NonFiniteLoss(i)
                loss.backward()
                opt.step()
                with torch.inference_mode():
                    z.copy_(z.maximum(z_min).minimum(z_max))
            except RuntimeError as e:
                e.lvq_iteration = i  # lets the fast-mode driver limit fp32
                raise                # retries to first-iteration failures
            want_image = i % stage_s.display_freq == 0 or i == total
            img = self._to_pil(out) if want_image else None
            loss_value = float(loss.detach()) if want_image else None
            yield FrameUpdate(i, total, img, loss_value)
        return z

    def _generate_coarse_to_fine(
        self, s: GenerationSettings, cancel: threading.Event | None
    ) -> Iterator[FrameUpdate]:
        seed = s.seed if s.seed >= 0 else int(torch.randint(0, 2**31 - 1, ()).item())
        precision = s.precision
        did_fp32_retry = False
        while True:
            self._set_model_dtype(precision)
            torch.manual_seed(seed)

            codebook = self.vqgan.codebook
            z_min = codebook.min(dim=0).values[None, :, None, None]
            z_max = codebook.max(dim=0).values[None, :, None, None]
            coarse_w, coarse_h, n_coarse = self._fast_coarse_dims(s)
            n_fine = s.iterations - n_coarse
            coarse_s = replace(s, width=coarse_w, height=coarse_h)
            prompt_modules = self._build_prompt_modules(s)
            try:
                z = self._init_z(coarse_s)
                z = yield from self._optimize_stage(
                    coarse_s, prompt_modules, z, z_min, z_max,
                    n_coarse, 1, s.iterations, cancel)
                if cancel is not None and cancel.is_set():
                    return
                with torch.no_grad():
                    # latent -> full res (same upsample as the MLX engine)
                    z = F.interpolate(z.detach(), scale_factor=2.0,
                                      mode="bilinear", align_corners=False)
                    z = z.maximum(z_min).minimum(z_max)
                did_fine_fp32_fallback = False
                while True:
                    try:
                        yield from self._optimize_stage(
                            s, prompt_modules, z, z_min, z_max,
                            n_fine, n_coarse + 1, s.iterations, cancel)
                        return
                    except _NonFiniteLoss as nf:
                        if (nf.iteration != n_coarse + 1
                                or did_fine_fp32_fallback
                                or self._model_dtype(precision) == torch.float32):
                            raise
                        did_fine_fp32_fallback = True
                        # The fp16 decoder overflows on upsampled latents (same
                        # Inf-in-decoder-up-chain failure the MLX engine
                        # documented; one-hot inits happen to survive it). Redo
                        # just the fine stage with an fp32 VQGAN — CLIP stays
                        # fp16, matching the MLX precision policy — instead of
                        # throwing away the finished coarse stage.
                        self.vqgan.set_dtype(torch.float32)
                        codebook = self.vqgan.codebook
                        z_min = codebook.min(dim=0).values[None, :, None, None]
                        z_max = codebook.max(dim=0).values[None, :, None, None]
                        with torch.no_grad():
                            z = z.float().maximum(z_min).minimum(z_max)
            except _NonFiniteLoss as nf:
                if (nf.iteration == 1 and precision != "fp32"
                        and not did_fp32_retry):
                    precision = "fp32"
                    did_fp32_retry = True
                    continue
                raise RuntimeError("generation produced non-finite loss")
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    raise GenerationOOM(
                        "Out of memory — try a smaller size or fewer cutouts."
                    ) from e
                if (precision != "fp32" and not did_fp32_retry
                        and getattr(e, "lvq_iteration", None) == 1):
                    precision = "fp32"  # fp16 unsupported for some op; retry in fp32
                    did_fp32_retry = True
                    continue
                raise

    def generate(self, s: GenerationSettings,
                 cancel: threading.Event | None = None) -> Iterator[FrameUpdate]:
        assert self.vqgan is not None and self.clip is not None, "call load() first"
        if self._fast_mode_eligible(s):
            yield from self._generate_coarse_to_fine(s, cancel)
            return
        seed = s.seed if s.seed >= 0 else int(torch.randint(0, 2**31 - 1, ()).item())

        precision = s.precision
        did_fp32_retry = False
        while True:
            self._set_model_dtype(precision)
            self.vqgan.set_decoder_checkpointing(
                s.width * s.height > TORCH_DECODE_CHECKPOINT_MIN_PIXELS)
            torch.manual_seed(seed)

            codebook = self.vqgan.codebook
            z_min = codebook.min(dim=0).values[None, :, None, None]
            z_max = codebook.max(dim=0).values[None, :, None, None]

            z = self._init_z(s).detach().float()
            z_orig = z.clone()
            z.requires_grad_(True)
            opt = optim.Adam([z], lr=s.step_size)

            make_cutouts = MakeCutouts(self.clip.cut_size, s.cutouts, s.cut_pow,
                                       device_type=self.device.type).to(self.device)
            prompt_modules = [
                Prompt(self.clip.embed_text(p.text), p.weight, p.stop).to(self.device)
                for p in parse_prompts(s.prompts)
            ]
            for path in s.image_prompts:
                embed = self.clip.embed_image(Image.open(path).convert("RGB"))
                prompt_modules.append(Prompt(embed, 1.0, float("-inf")).to(self.device))

            i = 1
            try:
                while i <= s.iterations:
                    if cancel is not None and cancel.is_set():
                        return
                    opt.zero_grad(set_to_none=True)
                    out = self._synth(z)
                    embeds = self.clip.encode_cutouts(make_cutouts(out))
                    losses = [pm(embeds) for pm in prompt_modules]
                    if s.init_weight:
                        losses.append(F.mse_loss(z, z_orig) * s.init_weight / 2)
                    loss = sum(losses)
                    if (i == 1 and precision != "fp32" and not did_fp32_retry
                            and not bool(torch.isfinite(loss.detach()).item())):
                        precision = "fp32"
                        did_fp32_retry = True
                        break
                    loss.backward()
                    opt.step()
                    with torch.inference_mode():
                        z.copy_(z.maximum(z_min).minimum(z_max))
                    want_image = i % s.display_freq == 0 or i == s.iterations
                    # reuse this iteration's decode for preview frames; a re-decode
                    # after opt.step would cost a full extra VQGAN forward
                    img = self._to_pil(out) if want_image else None
                    loss_value = float(loss.detach()) if want_image else None
                    yield FrameUpdate(i, s.iterations, img, loss_value)
                    i += 1
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    raise GenerationOOM(
                        "Out of memory — try a smaller size or fewer cutouts.") from e
                if precision != "fp32" and i == 1 and not did_fp32_retry:
                    precision = "fp32"  # fp16 unsupported for some op; retry in fp32
                    continue
                raise
            if i > s.iterations:
                return

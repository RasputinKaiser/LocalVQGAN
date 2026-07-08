import logging
import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterator

import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image

from localvqgan.pipeline import checkpoints
from localvqgan.pipeline.backends.mlx_backend import convert
from localvqgan.pipeline.backends.mlx_backend.clip import MlxClip
from localvqgan.pipeline.backends.mlx_backend.cutouts import (
    apply_cutout_spec,
    make_cutout_spec,
    make_cutouts,
)
from localvqgan.pipeline.backends.mlx_backend.losses import (
    clamp_with_grad,
    prompt_loss,
    vector_quantize,
)
from localvqgan.pipeline.backends.mlx_backend.vqgan import (
    MlxVQGAN,
    load_mlx_vqgan_from_arrays,
)
from localvqgan.pipeline.backends.torch_backend import FrameUpdate, GenerationOOM
from localvqgan.pipeline.prompts import parse_prompts
from localvqgan.pipeline.settings import GenerationSettings

logger = logging.getLogger(__name__)

# Decode and cutout intermediates scale with canvas pixels. Above the verified
# 256x256 MLX comfort zone, fp32 512x512 decode+cutouts thrashes 4.5GB+ into
# swap at 74-123s/it, while bf16 decode+chunked cutouts runs steady at ~3.5s/it
# with no swap growth; bf16 decode PSNR vs fp32 measured 55.1dB (gate >50dB).
MLX_LARGE_CANVAS_PIXEL_THRESHOLD = 256 * 256
CUTOUT_CHUNK_SIZE = 8
# Chunking bounds peak memory but adds per-chunk eval + Python-loop overhead.
# It is only needed once the working set would thrash: measured 2026-07-08,
# 384x384 runs a single pass (all cutouts at once) at 0.526 it/s vs 0.469 when
# chunked by 8 (+12%) with NO swap growth, while 512x512 still thrashes without
# chunking. So chunk only above this pixel count; at/below it, one pass. This is
# a float-reassociation change (identical gradient math); VQGAN+CLIP is chaotic,
# so a given seed renders a different but equal-quality image (loss unchanged).
MLX_UNCHUNKED_MAX_PIXELS = 384 * 384
MLX_LARGE_CANVAS_CACHE_LIMIT_GB = 4
ASYNC_CHUNK_WINDOW = 2
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Coarse-to-fine ("Fast mode"): run this fraction of the iteration budget at half
# resolution, then upsample the latent and finish at full res. 0.6 measured
# ~1.30x at 256² over 3 seeds (2026-07-08) with the color-cast/grit distribution
# unchanged vs the pristine path (only a mild ~10% luminance dip). The coarse
# grid must stay at least this many tokens per side or the composition is too
# blocky to upsample cleanly.
FAST_COARSE_FRACTION = 0.6
FAST_MIN_COARSE_TOKENS = 8


class _NonFiniteLoss(RuntimeError):
    """Signals a non-finite loss at a given iteration so the coarse-to-fine
    driver can retry the whole run in fp32 (matching the pristine path)."""

    def __init__(self, iteration: int):
        super().__init__("non-finite loss")
        self.iteration = iteration


def _is_large_canvas(width: int, height: int, cutn: int) -> bool:
    return cutn > 0 and width * height > MLX_LARGE_CANVAS_PIXEL_THRESHOLD


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number, got {raw!r}")
    return value


def _cutout_chunk_size(
    width: int | None = None, height: int | None = None, cutn: int | None = None
) -> int:
    # An explicit env override always wins (used to force a chunk size in tests
    # and for benchmarking, regardless of canvas).
    if "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE" in os.environ:
        return _positive_int_env("LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE", CUTOUT_CHUNK_SIZE)
    # Otherwise chunk only when the canvas is large enough to need it: a single
    # pass over all cutouts avoids per-chunk overhead where memory allows.
    if width and height and cutn and width * height <= MLX_UNCHUNKED_MAX_PIXELS:
        return cutn
    return CUTOUT_CHUNK_SIZE


def _large_canvas_cache_limit_bytes() -> int:
    gb = _positive_float_env(
        "LOCALVQGAN_MLX_CACHE_LIMIT_GB",
        MLX_LARGE_CANVAS_CACHE_LIMIT_GB,
    )
    return int(gb * 1024**3)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {raw!r}")


def _async_chunks_enabled() -> bool:
    return _env_flag("LOCALVQGAN_MLX_ASYNC_CHUNKS", True)


def _small_canvas_compiled_optimizer_enabled() -> bool:
    return _env_flag("LOCALVQGAN_MLX_COMPILED_OPTIMIZER", False)


def _small_canvas_preview_reuse_enabled() -> bool:
    return _env_flag("LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE", True)


def _single_loss_sum_fastpath_enabled() -> bool:
    return _env_flag("LOCALVQGAN_MLX_SINGLE_LOSS_SUM_FASTPATH", True)


def _sum_losses(losses: list[mx.array]) -> mx.array:
    if _single_loss_sum_fastpath_enabled() and len(losses) == 1:
        return losses[0]
    return sum(losses, mx.array(0.0))


def make_adam(learning_rate):
    return optim.Adam(
        learning_rate=learning_rate,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPS,
        bias_correction=True,
    )


def _adam_update(
    z: mx.array,
    grad: mx.array,
    m: mx.array,
    v: mx.array,
    step: mx.array,
    learning_rate: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    next_step = step + mx.array(1, dtype=mx.uint64)
    next_m = ADAM_BETA1 * m + (1 - ADAM_BETA1) * grad
    next_v = ADAM_BETA2 * v + (1 - ADAM_BETA2) * (grad * grad)
    lr = mx.array(learning_rate).astype(grad.dtype)
    c1 = (lr / (1 - ADAM_BETA1**next_step)).astype(grad.dtype)
    c2 = mx.rsqrt(1 - ADAM_BETA2**next_step).astype(grad.dtype)
    numerator = c1 * next_m
    denominator = mx.sqrt(next_v) * c2 + ADAM_EPS
    next_z = z - numerator / denominator
    return next_z, next_m, next_v, next_step


class MlxGenerator:
    def __init__(self):
        self.vqgan: MlxVQGAN | None = None
        self.clip: MlxClip | None = None
        self._loaded: tuple[str, str] | None = None
        self.device = type("MlxDevice", (), {"type": "mlx"})()
        self.compile_engaged = False

    def load(self, checkpoint: str, clip_model: str) -> None:
        if self._loaded == (checkpoint, clip_model):
            return
        weights_path = convert.cached_vqgan_weights(checkpoint)
        cfg, _ = checkpoints.checkpoint_paths(checkpoint)
        weights = dict(mx.load(str(weights_path)).items())
        # The VQGAN decoder's ResNet/Upsample chain overflows fp16's ~65504 max
        # value at 256-degree resolution as generation progresses (measured:
        # activations grow through the up-blocks and hit Inf at the final
        # Upsample before decode, independent of GroupNorm/attention precision)
        # -- run it in fp32. CLIP has no observed range/precision issue, so it
        # stays fp16 for speed.
        self.vqgan = load_mlx_vqgan_from_arrays(cfg, weights, dtype=mx.float32)
        if self.clip is None or self.clip.model_name != clip_model:
            self.clip = MlxClip.load(clip_model)
        self.clip.set_dtype(mx.float16)
        self._loaded = (checkpoint, clip_model)

    def load_from_paths(
        self,
        config_path: Path,
        ckpt_path: Path | None,
        clip_model: str,
    ) -> None:
        import torch

        if ckpt_path is None:
            # The tiny fixture path constructs random Torch weights before conversion;
            # seed it so generator tests compare stable images across fresh loads.
            torch.manual_seed(0)
        weights = convert.torch_vqgan_to_mlx_weights(config_path, ckpt_path)
        self.vqgan = load_mlx_vqgan_from_arrays(config_path, weights, dtype=mx.float32)
        if self.clip is None or self.clip.model_name != clip_model:
            self.clip = MlxClip.load(clip_model)
        self._loaded = None

    def _synth(self, z: mx.array) -> mx.array:
        assert self.vqgan is not None
        z_q = vector_quantize(z, self.vqgan.codebook)
        out = self.vqgan.decode(z_q)
        return clamp_with_grad((out + 1) / 2, mx.array(0.0), mx.array(1.0))

    def _init_z(self, s: GenerationSettings,
                width: int | None = None, height: int | None = None) -> mx.array:
        assert self.vqgan is not None
        f = self.vqgan.f
        width = s.width if width is None else width
        height = s.height if height is None else height
        toks_x, toks_y = width // f, height // f
        if s.init_image:
            img = Image.open(s.init_image).convert("RGB").resize(
                (toks_x * f, toks_y * f), Image.LANCZOS
            )
            arr = mx.array(np.asarray(img, dtype=np.float32)[None] / 127.5 - 1.0)
            return self.vqgan.encode(arr)
        idx = mx.random.randint(0, self.vqgan.n_toks, shape=(1, toks_y, toks_x))
        return self.vqgan.codebook[idx]

    @staticmethod
    def _to_pil(out: mx.array) -> Image.Image:
        # float32 first: numpy cannot read fp16/bf16 MLX buffers (PEP 3118)
        arr = np.array((mx.clip(out[0], 0, 1) * 255).astype(mx.float32)).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _raise_oom_if_needed(exc: Exception) -> None:
        if "memory" in str(exc).lower():
            raise GenerationOOM(
                "Out of memory - try a smaller size or fewer cutouts."
            ) from exc

    @staticmethod
    def _loss_is_finite(loss: mx.array) -> bool:
        return bool(np.array(mx.isfinite(loss)).item())

    def _compile_step(
        self,
        value_and_grad: Callable[[mx.array], tuple[mx.array, mx.array]],
    ) -> Callable[[mx.array], tuple[mx.array, mx.array]]:
        self.compile_engaged = False
        if os.environ.get("LOCALVQGAN_MLX_COMPILE", "1") == "0":
            return value_and_grad
        try:
            compiled = mx.compile(value_and_grad)
        except Exception as exc:
            logger.warning(
                "MLX compile unavailable; falling back to eager generation",
                exc_info=True,
            )
            return value_and_grad

        first_call = True
        compile_engaged = False

        def step(z: mx.array) -> tuple[mx.array, mx.array]:
            nonlocal first_call, compile_engaged
            if first_call:
                try:
                    out = compiled(z)
                    compile_engaged = True
                    self.compile_engaged = True
                    first_call = False
                    return out
                except Exception as exc:
                    logger.warning(
                        "MLX compiled generation step failed; falling back to eager",
                        exc_info=True,
                    )
                    first_call = False
                    compile_engaged = False
                    self.compile_engaged = False
                    return value_and_grad(z)
            return compiled(z) if compile_engaged else value_and_grad(z)

        return step

    def _compile_small_canvas_optimizer_step(self, optimizer_step: Callable) -> Callable:
        if os.environ.get("LOCALVQGAN_MLX_COMPILE", "1") == "0":
            return optimizer_step
        try:
            compiled = mx.compile(optimizer_step)
        except Exception:
            logger.warning(
                "MLX compile unavailable for optimizer step; falling back to eager",
                exc_info=True,
            )
            return optimizer_step

        first_call = True
        compile_engaged = False

        def step(*args):
            nonlocal first_call, compile_engaged
            if first_call:
                try:
                    out = compiled(*args)
                    first_call = False
                    compile_engaged = True
                    return out
                except Exception:
                    logger.warning(
                        "MLX compiled optimizer step failed; falling back to eager",
                        exc_info=True,
                    )
                    first_call = False
                    compile_engaged = False
                    return optimizer_step(*args)
            return compiled(*args) if compile_engaged else optimizer_step(*args)

        return step

    def _compile_large_canvas_fn(self, name: str, fn: Callable) -> Callable:
        if os.environ.get("LOCALVQGAN_MLX_COMPILE", "1") == "0":
            return fn
        try:
            compiled = mx.compile(fn)
        except Exception:
            logger.warning(
                "MLX compile unavailable for %s; falling back to eager",
                name,
                exc_info=True,
            )
            return fn

        first_call = True
        compile_engaged = False

        def wrapped(*args):
            nonlocal first_call, compile_engaged
            if first_call:
                try:
                    out = compiled(*args)
                    mx.eval(out)
                    first_call = False
                    compile_engaged = True
                    return out
                except Exception:
                    logger.warning(
                        "MLX compiled %s failed; falling back to eager",
                        name,
                        exc_info=True,
                    )
                    first_call = False
                    compile_engaged = False
                    return fn(*args)
            return compiled(*args) if compile_engaged else fn(*args)

        return wrapped

    def _large_canvas_fns(
        self,
    ) -> tuple[Callable[[mx.array], mx.array], Callable[[mx.array, mx.array], mx.array]]:
        return (
            self._compile_large_canvas_fn("large-canvas synth", self._synth),
            self._compile_large_canvas_fn(
                "large-canvas synth pullback",
                self._synth_pullback,
            ),
        )

    def _select_vqgan_dtype(self, large_canvas: bool, did_fp32_retry: bool) -> None:
        assert self.vqgan is not None
        if did_fp32_retry:
            self.vqgan.set_dtype(mx.float32)
        elif large_canvas:
            self.vqgan.set_dtype(mx.bfloat16)
        else:
            self.vqgan.set_dtype(mx.float32)

    def _synth_pullback(
        self,
        z: mx.array,
        grad_out: mx.array,
    ) -> mx.array:
        synth = lambda z_: self._synth(z_)
        pulled = None
        for primals in (z, (z,)):
            try:
                _, pullback = mx.vjp(synth, primals)
                pulled = pullback(grad_out)
                break
            except TypeError:
                pass
        if pulled is None:
            # MLX releases that expose vjp(fun, primals, cotangents) cannot return
            # a reusable pullback. This preserves the same gradient contract, but
            # may replay the decode while computing the VJP on those versions.
            _, pulled = mx.vjp(synth, (z,), (grad_out,))
        if isinstance(pulled, (tuple, list)):
            return pulled[0]
        return pulled

    def _chunked_value_grad_and_out(
        self,
        z: mx.array,
        z_orig: mx.array,
        s: GenerationSettings,
        targets: list[tuple[mx.array, float, float]],
        synth: Callable[[mx.array], mx.array] | None = None,
        synth_pullback: Callable[[mx.array, mx.array], mx.array] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        synth = self._synth if synth is None else synth
        synth_pullback = self._synth_pullback if synth_pullback is None else synth_pullback
        out = synth(z)
        spec = make_cutout_spec(out, s.cutouts, self.clip.cut_size, s.cut_pow)
        total_loss = mx.array(0.0)
        grad_out = mx.zeros_like(out)
        chunk_size = _cutout_chunk_size(s.width, s.height, s.cutouts)
        async_chunks = _async_chunks_enabled()
        pending_chunks: list[tuple[mx.array, mx.array]] = []

        def drain_oldest_chunk() -> None:
            nonlocal total_loss, grad_out
            chunk_loss_, chunk_grad_ = pending_chunks.pop(0)
            mx.eval(chunk_loss_, chunk_grad_)
            total_loss = total_loss + chunk_loss_
            grad_out = grad_out + chunk_grad_

        for start in range(0, s.cutouts, chunk_size):
            end = min(start + chunk_size, s.cutouts)
            chunk_weight = (end - start) / s.cutouts

            def chunk_loss_fn(out_: mx.array) -> mx.array:
                cutouts = apply_cutout_spec(out_, spec, self.clip.cut_size, start, end)
                embeds = self.clip.encode_cutouts(cutouts)
                losses = [prompt_loss(embeds, t, w, stop) for t, w, stop in targets]
                return _sum_losses(losses) * chunk_weight

            chunk_loss, chunk_grad = mx.value_and_grad(chunk_loss_fn)(out)
            if async_chunks:
                mx.async_eval(chunk_loss, chunk_grad)
                pending_chunks.append((chunk_loss, chunk_grad))
                if len(pending_chunks) >= ASYNC_CHUNK_WINDOW:
                    drain_oldest_chunk()
            else:
                mx.eval(chunk_loss, chunk_grad)
                total_loss = total_loss + chunk_loss
                grad_out = grad_out + chunk_grad

        while pending_chunks:
            drain_oldest_chunk()

        grad_z = synth_pullback(z, grad_out)

        if s.init_weight:
            def init_loss_fn(z_: mx.array) -> mx.array:
                return ((z_ - z_orig) ** 2).mean() * s.init_weight / 2

            init_loss, init_grad = mx.value_and_grad(init_loss_fn)(z)
            mx.eval(init_loss, init_grad)
            total_loss = total_loss + init_loss
            grad_z = grad_z + init_grad

        return total_loss, grad_z, out

    def _chunked_value_and_grad(
        self,
        z: mx.array,
        z_orig: mx.array,
        s: GenerationSettings,
        targets: list[tuple[mx.array, float, float]],
    ) -> tuple[mx.array, mx.array]:
        loss, grad_z, _ = self._chunked_value_grad_and_out(z, z_orig, s, targets)
        return loss, grad_z

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

    def _build_targets(
        self, s: GenerationSettings
    ) -> list[tuple[mx.array, float, float]]:
        targets = [
            (self.clip.embed_text(p.text), p.weight, p.stop)
            for p in parse_prompts(s.prompts)
        ]
        for path in s.image_prompts:
            img = Image.open(path).convert("RGB").resize(
                (self.clip.cut_size, self.clip.cut_size), Image.LANCZOS
            )
            arr = mx.array(np.asarray(img, dtype=np.float32)[None] / 255.0)
            targets.append((self.clip.encode_cutouts(arr), 1.0, float("-inf")))
        return targets

    def _optimize_resolution(
        self,
        stage_s: GenerationSettings,
        targets: list[tuple[mx.array, float, float]],
        z: mx.array,
        z_min: mx.array,
        z_max: mx.array,
        n_iters: int,
        global_start: int,
        cancel: threading.Event | None,
    ):
        """Optimize z for n_iters at the resolution encoded in stage_s (its
        width/height set the output size and cutout chunking), yielding
        FrameUpdates numbered from global_start. Returns the final z via the
        generator return. Uses the chunked large-canvas machinery when the stage
        canvas needs it, the compiled single-pass path otherwise — so the coarse
        stage runs the cheap in-RAM path and only a >256² fine stage chunks."""
        total = stage_s.iterations
        large = _is_large_canvas(stage_s.width, stage_s.height, stage_s.cutouts)
        z_orig = z  # init_weight anchor for this stage (default weight 0 -> unused)
        if large:
            synth, synth_pullback = self._large_canvas_fns()
        else:
            def loss_and_out_fn(z_: mx.array) -> tuple[mx.array, mx.array]:
                out = self._synth(z_)
                embeds = self.clip.encode_cutouts(
                    make_cutouts(out, stage_s.cutouts, self.clip.cut_size,
                                 stage_s.cut_pow)
                )
                losses = [prompt_loss(embeds, t, w, stop) for t, w, stop in targets]
                return _sum_losses(losses), out

            step = self._compile_step(mx.value_and_grad(loss_and_out_fn))
        opt = make_adam(stage_s.step_size)
        params = {"z": z}
        for k in range(n_iters):
            if cancel is not None and cancel.is_set():
                return params["z"]
            i = global_start + k
            try:
                if large:
                    loss, grad, out = self._chunked_value_grad_and_out(
                        params["z"], z_orig, stage_s, targets, synth, synth_pullback)
                else:
                    (loss, out), grad = step(params["z"])
                params = opt.apply_gradients({"z": grad}, params)
                params["z"] = mx.clip(params["z"], z_min, z_max)
                mx.eval(loss, params["z"])
            except Exception as exc:
                self._raise_oom_if_needed(exc)
                raise
            if not self._loss_is_finite(loss):
                raise _NonFiniteLoss(i)
            want_image = i % stage_s.display_freq == 0 or i == total
            img = self._to_pil(out) if want_image else None
            loss_value = float(np.array(loss).item()) if want_image else None
            yield FrameUpdate(i, total, img, loss_value)
        return params["z"]

    def _generate_coarse_to_fine(
        self, s: GenerationSettings, cancel: threading.Event | None
    ):
        assert self.vqgan is not None and self.clip is not None
        seed = s.seed if s.seed >= 0 else int.from_bytes(os.urandom(4), "little") % (2**31)
        codebook = self.vqgan.codebook
        z_min, z_max = codebook.min(axis=0), codebook.max(axis=0)
        coarse_w, coarse_h, n_coarse = self._fast_coarse_dims(s)
        n_fine = s.iterations - n_coarse
        coarse_s = replace(s, width=coarse_w, height=coarse_h)
        coarse_large = _is_large_canvas(coarse_w, coarse_h, s.cutouts)
        fine_large = _is_large_canvas(s.width, s.height, s.cutouts)
        upsample = mlx_nn.Upsample(scale_factor=2.0, mode="linear",
                                   align_corners=False)

        did_fp32_retry = False
        while True:
            cache_limit_prev = None
            try:
                mx.random.seed(seed)
                self.clip.set_dtype(mx.float32 if did_fp32_retry else mx.float16)
                targets = self._build_targets(s)

                # Coarse stage: half res (256² for a 512² target) -> fp32, in RAM.
                self._select_vqgan_dtype(coarse_large, did_fp32_retry)
                z = mx.array(self._init_z(s, coarse_w, coarse_h), dtype=mx.float32)
                z = yield from self._optimize_resolution(
                    coarse_s, targets, z, z_min, z_max, n_coarse, 1, cancel)
                if cancel is not None and cancel.is_set():
                    return

                z = mx.clip(upsample(z), z_min, z_max)  # latent -> full res
                mx.eval(z)

                # Fine stage: full res. If large (>256²) this switches VQGAN to
                # bf16 and brackets the MLX cache, matching the pristine large
                # path; the coarse stage already ran without that memory pressure.
                self._select_vqgan_dtype(fine_large, did_fp32_retry)
                if fine_large:
                    cache_limit_prev = mx.set_cache_limit(
                        _large_canvas_cache_limit_bytes())
                yield from self._optimize_resolution(
                    s, targets, z, z_min, z_max, n_fine, n_coarse + 1, cancel)
                return
            except _NonFiniteLoss as nf:
                if nf.iteration == 1 and not did_fp32_retry:
                    did_fp32_retry = True
                    continue
                raise RuntimeError(
                    "generation produced non-finite loss; try the torch engine")
            finally:
                if cache_limit_prev is not None:
                    mx.set_cache_limit(cache_limit_prev)
                    mx.clear_cache()

    def generate(
        self,
        s: GenerationSettings,
        cancel: threading.Event | None = None,
    ) -> Iterator[FrameUpdate]:
        assert self.vqgan is not None and self.clip is not None, "call load() first"
        if self._fast_mode_eligible(s):
            yield from self._generate_coarse_to_fine(s, cancel)
            return
        seed = s.seed if s.seed >= 0 else int.from_bytes(os.urandom(4), "little") % (2**31)

        codebook = self.vqgan.codebook
        z_min = codebook.min(axis=0)
        z_max = codebook.max(axis=0)

        did_fp32_retry = False
        while True:
            retry_fp32 = False
            mx.random.seed(seed)
            z = mx.array(self._init_z(s), dtype=mx.float32)
            z_orig = mx.array(z)
            params = {"z": z}
            opt = make_adam(s.step_size)
            adam_m = mx.zeros_like(z)
            adam_v = mx.zeros_like(z)
            adam_step = mx.array(0, dtype=mx.uint64)
            large_canvas = _is_large_canvas(s.width, s.height, s.cutouts)
            self._select_vqgan_dtype(large_canvas, did_fp32_retry)

            cache_limit_prev = None
            if large_canvas:
                # Measured 512x512 bf16+chunked: unbounded MLX cache climbed
                # 3.7->143s/it with +11GB swap over 13 its; 4GB stayed stable
                # at ~4.0s/it mean over 10 its and releases cleanly afterward.
                cache_limit_prev = mx.set_cache_limit(_large_canvas_cache_limit_bytes())

            try:
                targets = [
                    (self.clip.embed_text(prompt.text), prompt.weight, prompt.stop)
                    for prompt in parse_prompts(s.prompts)
                ]
                for path in s.image_prompts:
                    img = Image.open(path).convert("RGB").resize(
                        (self.clip.cut_size, self.clip.cut_size), Image.LANCZOS
                    )
                    arr = mx.array(np.asarray(img, dtype=np.float32)[None] / 255.0)
                    targets.append((self.clip.encode_cutouts(arr), 1.0, float("-inf")))

                optimizer_in_step = False
                optimizer_step = None
                preview_from_step = False
                if large_canvas:
                    synth, synth_pullback = self._large_canvas_fns()
                    step = lambda z_: self._chunked_value_grad_and_out(
                        z_,
                        z_orig,
                        s,
                        targets,
                        synth,
                        synth_pullback,
                    )
                else:
                    def loss_from_out(z_: mx.array, out_: mx.array) -> mx.array:
                        embeds = self.clip.encode_cutouts(
                            make_cutouts(out_, s.cutouts, self.clip.cut_size, s.cut_pow)
                        )
                        losses = [prompt_loss(embeds, t, w, stop) for t, w, stop in targets]
                        if s.init_weight:
                            losses.append(((z_ - z_orig) ** 2).mean() * s.init_weight / 2)
                        return _sum_losses(losses)

                    def loss_and_out_fn(z_: mx.array) -> tuple[mx.array, mx.array]:
                        out = self._synth(z_)
                        return loss_from_out(z_, out), out

                    def loss_fn(z_: mx.array) -> mx.array:
                        loss_, _ = loss_and_out_fn(z_)
                        return loss_

                    if (
                        os.environ.get("LOCALVQGAN_MLX_COMPILE", "1") == "0"
                        or not _small_canvas_compiled_optimizer_enabled()
                    ):
                        if (
                            _small_canvas_preview_reuse_enabled()
                            and s.display_freq < s.iterations
                        ):
                            preview_from_step = True
                            step = self._compile_step(mx.value_and_grad(loss_and_out_fn))
                        else:
                            step = self._compile_step(mx.value_and_grad(loss_fn))
                    else:
                        optimizer_in_step = True
                        value_and_grad = mx.value_and_grad(loss_fn)
                        step = self._compile_step(value_and_grad)

                        def update_step(
                            z_: mx.array,
                            grad_: mx.array,
                            m_: mx.array,
                            v_: mx.array,
                            adam_step_: mx.array,
                        ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
                            next_z, next_m, next_v, next_step = _adam_update(
                                z_, grad_, m_, v_, adam_step_, s.step_size
                            )
                            next_z = mx.clip(next_z, z_min, z_max)
                            return next_z, next_m, next_v, next_step

                        optimizer_step = self._compile_small_canvas_optimizer_step(
                            update_step
                        )

                i = 1
                while i <= s.iterations:
                    if cancel is not None and cancel.is_set():
                        return
                    try:
                        preview_z = params["z"]
                        step_out = step(preview_z)
                        if large_canvas:
                            loss, grad, preview_out = step_out
                        elif preview_from_step:
                            (loss, preview_out), grad = step_out
                        elif optimizer_in_step:
                            assert optimizer_step is not None
                            loss, grad = step_out
                            next_z, adam_m, adam_v, adam_step = optimizer_step(
                                preview_z,
                                grad,
                                adam_m,
                                adam_v,
                                adam_step,
                            )
                            params["z"] = next_z
                            preview_out = None
                        else:
                            loss, grad = step_out
                            preview_out = None
                        if not optimizer_in_step:
                            params = opt.apply_gradients({"z": grad}, params)
                            params["z"] = mx.clip(params["z"], z_min, z_max)
                        mx.eval(loss, params["z"])
                    except Exception as exc:
                        self._raise_oom_if_needed(exc)
                        raise

                    if not self._loss_is_finite(loss):
                        if i == 1 and not did_fp32_retry:
                            did_fp32_retry = True
                            self.vqgan.set_dtype(mx.float32)
                            self.clip.set_dtype(mx.float32)
                            retry_fp32 = True
                            break
                        raise RuntimeError(
                            "generation produced non-finite loss; try the torch engine"
                        )

                    want_image = i % s.display_freq == 0 or i == s.iterations
                    # Torch previews decode the same pre-update z that produced the reported loss.
                    if want_image and preview_out is None:
                        preview_out = self._synth(preview_z)
                    img = self._to_pil(preview_out) if want_image else None
                    loss_value = float(np.array(loss).item()) if want_image else None
                    yield FrameUpdate(i, s.iterations, img, loss_value)
                    i += 1
                if i > s.iterations:
                    return
            finally:
                if large_canvas:
                    mx.set_cache_limit(cache_limit_prev)
                    mx.clear_cache()
            if retry_fp32:
                continue

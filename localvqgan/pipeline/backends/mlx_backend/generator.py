import logging
import os
import threading
from pathlib import Path
from typing import Callable, Iterator

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from PIL import Image

from localvqgan.pipeline import checkpoints
from localvqgan.pipeline.backends.mlx_backend import convert
from localvqgan.pipeline.backends.mlx_backend.clip import MlxClip
from localvqgan.pipeline.backends.mlx_backend.cutouts import make_cutouts
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


def make_adam(learning_rate):
    return optim.Adam(
        learning_rate=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        bias_correction=True,
    )


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

    def _init_z(self, s: GenerationSettings) -> mx.array:
        assert self.vqgan is not None
        f = self.vqgan.f
        toks_x, toks_y = s.width // f, s.height // f
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
        arr = np.array(mx.clip(out[0], 0, 1) * 255).astype(np.uint8)
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

        def step(z: mx.array) -> tuple[mx.array, mx.array]:
            nonlocal first_call
            if first_call:
                try:
                    out = compiled(z)
                    self.compile_engaged = True
                    first_call = False
                    return out
                except Exception as exc:
                    logger.warning(
                        "MLX compiled generation step failed; falling back to eager",
                        exc_info=True,
                    )
                    first_call = False
                    self.compile_engaged = False
                    return value_and_grad(z)
            return compiled(z) if self.compile_engaged else value_and_grad(z)

        return step

    def generate(
        self,
        s: GenerationSettings,
        cancel: threading.Event | None = None,
    ) -> Iterator[FrameUpdate]:
        assert self.vqgan is not None and self.clip is not None, "call load() first"
        seed = s.seed if s.seed >= 0 else int.from_bytes(os.urandom(4), "little") % (2**31)

        codebook = self.vqgan.codebook
        z_min = codebook.min(axis=0)
        z_max = codebook.max(axis=0)

        did_fp32_retry = False
        while True:
            mx.random.seed(seed)
            z = mx.array(self._init_z(s), dtype=mx.float32)
            z_orig = mx.array(z)
            params = {"z": z}
            opt = make_adam(s.step_size)

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

            def loss_fn(z_: mx.array) -> mx.array:
                out = self._synth(z_)
                embeds = self.clip.encode_cutouts(
                    make_cutouts(out, s.cutouts, self.clip.cut_size, s.cut_pow)
                )
                losses = [prompt_loss(embeds, t, w, stop) for t, w, stop in targets]
                if s.init_weight:
                    losses.append(((z_ - z_orig) ** 2).mean() * s.init_weight / 2)
                return sum(losses, mx.array(0.0))

            step = self._compile_step(mx.value_and_grad(loss_fn))

            i = 1
            while i <= s.iterations:
                if cancel is not None and cancel.is_set():
                    return
                try:
                    preview_z = params["z"]
                    loss, grad = step(preview_z)
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
                        break
                    raise RuntimeError(
                        "generation produced non-finite loss; try the torch engine"
                    )

                want_image = i % s.display_freq == 0 or i == s.iterations
                # Torch previews decode the same pre-update z that produced the reported loss.
                img = self._to_pil(self._synth(preview_z)) if want_image else None
                loss_value = float(np.array(loss).item()) if want_image else None
                yield FrameUpdate(i, s.iterations, img, loss_value)
                i += 1
            if i > s.iterations:
                return

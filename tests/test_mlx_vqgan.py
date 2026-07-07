from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mlx")
import mlx.core as mx

from localvqgan.pipeline.backends.mlx_backend.convert import torch_vqgan_to_mlx_weights
from localvqgan.pipeline.backends.mlx_backend.vqgan import MlxVQGAN, load_mlx_vqgan_from_arrays
from localvqgan.pipeline.vqgan import load_vqgan

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_vqgan.yaml"


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(4.0 / mse)  # range [-1,1] -> peak 2


def test_decode_matches_torch():
    torch.manual_seed(0)
    tw = load_vqgan(FIXTURE, ckpt_path=None, device=torch.device("cpu"))
    weights = torch_vqgan_to_mlx_weights(FIXTURE, None, torch_wrapper=tw)
    mw = load_mlx_vqgan_from_arrays(FIXTURE, weights, dtype=mx.float32)
    assert mw.f == tw.f and mw.n_toks == tw.n_toks and mw.e_dim == tw.e_dim

    z = torch.randn(1, tw.e_dim, 8, 8)
    ref = tw.decode(z).detach().numpy()                      # NCHW
    out = np.array(mw.decode(mx.array(z.numpy().transpose(0, 2, 3, 1))))  # NHWC
    assert _psnr(ref.transpose(0, 2, 3, 1), out) > 50


def test_encode_matches_torch():
    torch.manual_seed(0)
    tw = load_vqgan(FIXTURE, ckpt_path=None, device=torch.device("cpu"))
    weights = torch_vqgan_to_mlx_weights(FIXTURE, None, torch_wrapper=tw)
    mw = load_mlx_vqgan_from_arrays(FIXTURE, weights, dtype=mx.float32)
    img = torch.rand(1, 3, 64, 64) * 2 - 1
    ref = tw.encode(img).detach().numpy()
    out = np.array(mw.encode(mx.array(img.numpy().transpose(0, 2, 3, 1))))
    assert _psnr(ref.transpose(0, 2, 3, 1), out) > 50


def test_gumbel_checkpoint_rejected_clearly():
    from localvqgan.pipeline.backends.mlx_backend import convert as c

    class FakeGumbelWrapper:
        class model:
            @staticmethod
            def state_dict():
                return {"quantize.embed.weight": torch.zeros(4, 2),
                        "quantize.proj.weight": torch.zeros(4, 8, 1, 1)}

    with pytest.raises(NotImplementedError, match="Gumbel"):
        c.torch_vqgan_to_mlx_weights(FIXTURE, None, torch_wrapper=FakeGumbelWrapper())

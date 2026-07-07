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

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


def test_decoder_checkpointing_is_exact():
    # Decoder gradient checkpointing must be pure recompute: identical output
    # AND identical grad w.r.t. z, so 512² seeds/outputs are unchanged.
    w = load_vqgan(FIXTURE, ckpt_path=None, device=torch.device("cpu"))
    z = torch.randn(1, 8, 32, 32)

    def run():
        zc = z.clone().requires_grad_(True)
        out = w.decode(zc)
        out.pow(2).sum().backward()
        return out.detach().clone(), zc.grad.clone()

    w.set_decoder_checkpointing(False)
    out_plain, grad_plain = run()
    w.set_decoder_checkpointing(True)
    out_ckpt, grad_ckpt = run()

    assert torch.equal(out_plain, out_ckpt)
    assert torch.equal(grad_plain, grad_ckpt)

from pathlib import Path

import torch

from localvqgan.pipeline.vendor.taming_model import Decoder
from localvqgan.pipeline.vqgan import load_vqgan

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_vqgan.yaml"


def _decode_exact_under_checkpointing(decode, z_channels):
    """decode(z) must be bit-exact in output AND grad with checkpointing on/off."""
    torch.manual_seed(0)
    z = torch.randn(1, z_channels, 16, 16)

    def run(flag):
        decode.use_checkpoint = flag
        zc = z.clone().requires_grad_(True)
        out = decode(zc)
        out.pow(2).sum().backward()
        return out.detach().clone(), zc.grad.clone()

    o0, g0 = run(False)
    o1, g1 = run(True)
    assert torch.equal(o0, o1)
    assert torch.equal(g0, g1)


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


def test_decoder_checkpointing_exact_with_attention():
    # The real imagenet decoder has AttnBlocks INSIDE the up-levels (the
    # checkpointed segment); the tiny yaml fixture has attn_resolutions=[], so
    # cover the attention path with a decoder that actually has one. z is 16×16
    # at z_channels; curr_res starts at 16 so attn_resolutions=[16] puts an
    # AttnBlock in the first up-level.
    dec = Decoder(ch=32, out_ch=3, ch_mult=(1, 2), num_res_blocks=1,
                  attn_resolutions=[16], in_channels=3, resolution=32,
                  z_channels=8).eval()
    assert any(len(level.attn) > 0 for level in dec.up)  # attn really present
    _decode_exact_under_checkpointing(dec, z_channels=8)

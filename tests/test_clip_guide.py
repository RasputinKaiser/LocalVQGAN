import pytest
import torch

from localvqgan.pipeline.clip_guide import Prompt, clamp_with_grad, vector_quantize


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

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

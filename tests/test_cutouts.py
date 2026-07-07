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


def test_default_cut_pow_matches_explicit_one():
    x = torch.rand(1, 3, 64, 64)
    implicit = MakeCutouts(cut_size=32, cutn=4)
    explicit = MakeCutouts(cut_size=32, cutn=4, cut_pow=1.0)

    torch.manual_seed(123)
    a = implicit(x)
    torch.manual_seed(123)
    b = explicit(x)

    assert torch.allclose(a, b)


def test_mps_config_has_no_grid_sample_augs():
    import kornia.augmentation as K
    mc = MakeCutouts(cut_size=32, cutn=4, device_type="mps")
    kinds = [type(m) for m in mc.augs]
    assert K.RandomAffine not in kinds and K.RandomPerspective not in kinds
    mc_cpu = MakeCutouts(cut_size=32, cutn=4, device_type="cpu")
    kinds_cpu = [type(m) for m in mc_cpu.augs]
    assert K.RandomAffine in kinds_cpu and K.RandomPerspective in kinds_cpu

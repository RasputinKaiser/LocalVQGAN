from localvqgan.pipeline import backends
from localvqgan.pipeline.generator import FrameUpdate, GenerationOOM, Generator  # shim intact


def test_shim_reexports():
    assert Generator is not None and FrameUpdate is not None and GenerationOOM is not None


def test_resolve_explicit_torch():
    assert backends.resolve_engine("torch", "imagenet_16384", "ViT-B-32") == ("torch", "explicit")


def test_resolve_auto_without_mlx(monkeypatch):
    monkeypatch.setattr(backends, "mlx_available", lambda: False)
    name, reason = backends.resolve_engine("auto", "imagenet_16384", "ViT-B-32")
    assert name == "torch" and "mlx" in reason


def test_resolve_auto_with_mlx_supported(monkeypatch):
    monkeypatch.setattr(backends, "mlx_available", lambda: True)
    monkeypatch.setattr(backends, "mlx_supports", lambda c, m: True)
    monkeypatch.setattr(backends, "MLX_MEETS_SPEED_GATE", True)
    assert backends.resolve_engine("auto", "imagenet_16384", "ViT-B-32")[0] == "mlx"


def test_resolve_auto_unsupported_checkpoint(monkeypatch):
    monkeypatch.setattr(backends, "mlx_available", lambda: True)
    monkeypatch.setattr(backends, "mlx_supports", lambda c, m: c != "gumbel_8192")
    name, reason = backends.resolve_engine("auto", "gumbel_8192", "ViT-B-32")
    assert name == "torch" and "unsupported" in reason


def test_resolve_explicit_mlx_unavailable_raises(monkeypatch):
    monkeypatch.setattr(backends, "mlx_available", lambda: False)
    import pytest
    with pytest.raises(RuntimeError):
        backends.resolve_engine("mlx", "imagenet_16384", "ViT-B-32")


def test_settings_engine_default():
    from localvqgan.pipeline.settings import GenerationSettings
    assert GenerationSettings().engine == "auto"

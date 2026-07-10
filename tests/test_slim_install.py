"""The slim Apple-Silicon install has no torch: the server import chain, engine
resolution, and MLX generator construction must all work with torch blocked."""
import subprocess
import sys

import pytest

from localvqgan.pipeline import backends

BLOCK_TORCH_AND_RUN = """
import sys

class _BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked in this slim-install test")
        return None

sys.meta_path.insert(0, _BlockTorch())

from pathlib import Path

from localvqgan.pipeline.backends import make_generator, resolve_engine
from localvqgan.server.app import create_app
from localvqgan.server.main import default_manager

engine, reason = resolve_engine("auto", "imagenet_16384", "ViT-B-32")
assert engine == "mlx", (engine, reason)
gen = make_generator("mlx")
assert gen.device.type == "mlx"
app = create_app(default_manager(Path("outputs")))
routes = {r.path for r in app.routes}
assert "/api/jobs" in routes and "/api/system" in routes
try:
    resolve_engine("torch", "imagenet_16384", "ViT-B-32")
except RuntimeError as e:
    assert "localvqgan[torch]" in str(e), e
else:
    raise AssertionError("explicit torch engine should fail without torch")
print("SLIM_OK")
"""


@pytest.mark.skipif(not backends.mlx_available(), reason="needs mlx (Apple Silicon)")
def test_server_and_mlx_engine_work_with_torch_blocked():
    proc = subprocess.run([sys.executable, "-c", BLOCK_TORCH_AND_RUN],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "SLIM_OK" in proc.stdout


GENERATE_UNDER_BLOCK = """
import sys

class _BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked in this slim-install test")
        return None

sys.meta_path.insert(0, _BlockTorch())

import numpy as np
from localvqgan.pipeline.backends import make_generator
from localvqgan.pipeline.settings import GenerationSettings

g = make_generator("mlx")
g.load("imagenet_16384", "ViT-B-32")
s = GenerationSettings(prompts="a red square", width=256, height=256,
                       iterations=2, cutouts=4, seed=42, display_freq=1)
frames = list(g.generate(s))
assert len(frames) == 2 and frames[-1].image.size == (256, 256)
assert all(np.isfinite(f.loss) for f in frames)
print("SLIM_GEN_OK")
"""


@pytest.mark.slow
@pytest.mark.skipif(not backends.mlx_available(), reason="needs mlx (Apple Silicon)")
def test_real_generation_with_torch_blocked():
    # real cached weights, whole load+generate path, torch import forbidden
    from localvqgan.pipeline import checkpoints

    if not checkpoints.has_mlx_weights("imagenet_16384"):
        pytest.skip("no converted imagenet_16384 MLX weights cached")
    proc = subprocess.run([sys.executable, "-c", GENERATE_UNDER_BLOCK],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "SLIM_GEN_OK" in proc.stdout


def test_resolve_engine_errors_without_any_engine(monkeypatch):
    monkeypatch.setattr(backends, "mlx_available", lambda: False)
    monkeypatch.setattr(backends, "torch_available", lambda: False)
    with pytest.raises(RuntimeError, match="no engine installed"):
        backends.resolve_engine("auto", "imagenet_16384", "ViT-B-32")


def test_resolve_engine_auto_falls_back_to_torch(monkeypatch):
    monkeypatch.setattr(backends, "mlx_available", lambda: False)
    monkeypatch.setattr(backends, "torch_available", lambda: True)
    assert backends.resolve_engine("auto", "imagenet_16384", "ViT-B-32") == (
        "torch", "mlx not installed")

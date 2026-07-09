"""The numpy torch-pickle reader must match torch.load exactly — it feeds the
slim install's weight conversion, where fidelity is the product."""
from pathlib import Path

import numpy as np
import pytest
import torch

from localvqgan.pipeline.torch_pickle import load_torch_pickle, load_torch_state_dict

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_vqgan.yaml"


def test_roundtrip_matches_torch_load(tmp_path):
    torch.manual_seed(7)
    base = torch.randn(4, 6)
    sd = {
        "a.weight": torch.randn(3, 5, 2, 2),
        "a.bias": torch.randn(3),
        "shared.view": base.t(),  # non-contiguous, shares storage with base
        "shared.base": base,
        "scalar": torch.tensor(2.5),
        "ints": torch.arange(7, dtype=torch.int64),
        "half": torch.randn(4).half(),
        "flag": torch.tensor(True),
    }
    p = tmp_path / "sd.pt"
    torch.save(sd, p)
    ours = load_torch_pickle(p)
    theirs = torch.load(p, map_location="cpu", weights_only=True)
    assert set(ours) == set(theirs)
    for k in theirs:
        expected = theirs[k].numpy()
        assert ours[k].dtype == expected.dtype, k
        assert np.array_equal(ours[k], expected), k


def test_nested_lightning_style_ckpt(tmp_path):
    sd = {"state_dict": {"w": torch.ones(2, 2)}, "epoch": 3,
          "extra": {"note": "meta"}}
    p = tmp_path / "last.ckpt"
    torch.save(sd, p)
    got = load_torch_state_dict(p)
    assert list(got) == ["w"]
    assert np.array_equal(got["w"], np.ones((2, 2), dtype=np.float32))


@pytest.mark.slow
def test_clip_bin_parity_with_torch_load():
    from huggingface_hub import snapshot_download

    try:
        snapshot_dir = Path(snapshot_download(
            repo_id="openai/clip-vit-base-patch32",
            allow_patterns=["*.bin"], local_files_only=True))
    except Exception:
        pytest.skip("CLIP pytorch_model.bin not in local HF cache")
    bin_path = snapshot_dir / "pytorch_model.bin"
    ours = load_torch_state_dict(bin_path)
    theirs = torch.load(bin_path, map_location="cpu", weights_only=True)
    tensor_keys = {k for k, v in theirs.items() if isinstance(v, torch.Tensor)}
    assert set(ours) == tensor_keys
    for k in sorted(tensor_keys):
        assert np.array_equal(ours[k], theirs[k].numpy()), k


@pytest.mark.slow
def test_vqgan_ckpt_torchless_conversion_parity():
    from localvqgan.pipeline import checkpoints
    from localvqgan.pipeline.backends.mlx_backend.convert import (
        ckpt_to_mlx_weights, torch_vqgan_to_mlx_weights)

    if not checkpoints.is_downloaded("imagenet_16384"):
        pytest.skip("imagenet_16384 ckpt not downloaded")
    cfg, ckpt = checkpoints.checkpoint_paths("imagenet_16384")
    torchless = ckpt_to_mlx_weights(cfg, ckpt)
    via_torch = torch_vqgan_to_mlx_weights(cfg, ckpt)
    assert set(torchless) == set(via_torch)
    for k in sorted(via_torch):
        assert torchless[k].dtype == via_torch[k].dtype, k
        assert np.array_equal(torchless[k], via_torch[k]), k

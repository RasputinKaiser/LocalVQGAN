from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from omegaconf import OmegaConf

from localvqgan.pipeline.backends.mlx_backend.vqgan import MlxVQGAN


def _load_config(config_path):
    config = OmegaConf.load(config_path)
    params = config.model.params
    return OmegaConf.to_container(params.ddconfig), params.n_embed, params.embed_dim


def _validate_weights(config_path, weights, consumed):
    ddconfig, n_embed, embed_dim = _load_config(config_path)
    model = MlxVQGAN(ddconfig, n_embed, embed_dim)
    mlx_params = dict(tree_flatten(model.parameters()))
    torch_only = sorted(consumed - set(mlx_params))
    mlx_only = sorted(set(mlx_params) - set(weights))
    shape_mismatches = []
    for name, value in weights.items():
        if name in mlx_params and tuple(value.shape) != tuple(mlx_params[name].shape):
            shape_mismatches.append((name, tuple(value.shape), tuple(mlx_params[name].shape)))
    if torch_only or mlx_only or shape_mismatches:
        raise ValueError(
            "VQGAN MLX conversion mismatch:\n"
            f"torch-only names: {torch_only}\n"
            f"mlx-only names: {mlx_only}\n"
            f"shape mismatches: {shape_mismatches}"
        )


def _state_arrays_to_mlx_weights(config_path, state_arrays) -> dict[str, np.ndarray]:
    if any(k.startswith(("quantize.embed.", "quantize.proj."))
           for k in state_arrays):
        raise NotImplementedError(
            "Gumbel-quantizer checkpoints are not supported by the MLX engine; "
            "use the torch engine for this checkpoint")
    weights = {}
    consumed = set()
    for name, array in state_arrays.items():
        if name.startswith("loss."):
            continue  # perceptual/discriminator weights, unused at inference
        if array.ndim == 4:
            array = array.transpose(0, 2, 3, 1)
        weights[name] = array
        consumed.add(name)
    _validate_weights(config_path, weights, consumed)
    return weights


def torch_vqgan_to_mlx_weights(config_path, ckpt_path, torch_wrapper=None) -> dict[str, np.ndarray]:
    import torch

    if torch_wrapper is None:
        from localvqgan.pipeline.vqgan import load_vqgan

        torch_wrapper = load_vqgan(config_path, ckpt_path, torch.device("cpu"))
    state_arrays = {name: tensor.detach().cpu().numpy()
                    for name, tensor in torch_wrapper.model.state_dict().items()}
    return _state_arrays_to_mlx_weights(config_path, state_arrays)


def ckpt_to_mlx_weights(config_path, ckpt_path) -> dict[str, np.ndarray]:
    """Torch-free conversion straight from the official ckpt pickle. Parity
    with the torch path is pinned by tests (identical arrays)."""
    from localvqgan.pipeline.torch_pickle import load_torch_state_dict

    return _state_arrays_to_mlx_weights(config_path, load_torch_state_dict(ckpt_path))


def cached_vqgan_weights(name: str) -> Path:
    from localvqgan.pipeline.backends import torch_available
    from localvqgan.pipeline.checkpoints import checkpoint_paths

    path = Path.home() / ".cache" / "localvqgan" / name / "mlx" / "vqgan.safetensors"
    if path.exists():
        return path
    config_path, ckpt_path = checkpoint_paths(name)
    if torch_available():
        weights = torch_vqgan_to_mlx_weights(config_path, ckpt_path)
    else:
        # slim install: read the official ckpt pickle with numpy directly
        weights = ckpt_to_mlx_weights(config_path, ckpt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    mx.save_safetensors(str(tmp), {k: mx.array(v).astype(mx.float16) for k, v in weights.items()})
    tmp.replace(path)
    return path


def cached_clip_weights(model_name: str) -> Path:
    import shutil

    from huggingface_hub import snapshot_download

    repos = {
        "ViT-B-32": "openai/clip-vit-base-patch32",
        "ViT-B-16": "openai/clip-vit-base-patch16",
    }
    if model_name not in repos:
        raise ValueError(f"Unknown CLIP model: {model_name}")

    path = Path.home() / ".cache" / "localvqgan" / "clip" / model_name / "mlx" / "clip.safetensors"
    if path.exists():
        return path

    # The OpenAI repos only publish pytorch_model.bin (no safetensors); the
    # numpy pickle reader keeps this torch-free on the slim install.
    from localvqgan.pipeline.torch_pickle import load_torch_state_dict

    snapshot_dir = Path(snapshot_download(
        repo_id=repos[model_name],
        allow_patterns=["*.bin", "*.json", "*.txt"],
    ))
    state_dict = load_torch_state_dict(snapshot_dir / "pytorch_model.bin")
    weights = {}
    for key, array in state_dict.items():
        if "position_ids" in key or key == "logit_scale":
            continue
        if "patch_embedding.weight" in key:
            array = array.transpose(0, 2, 3, 1)
        weights[key] = array

    path.parent.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "vocab.json", "merges.txt"):
        shutil.copyfile(snapshot_dir / name, path.parent / name)

    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    mx.save_safetensors(str(tmp), {k: mx.array(v).astype(mx.float32) for k, v in weights.items()})
    tmp.replace(path)
    return path

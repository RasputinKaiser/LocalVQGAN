from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from localvqgan.pipeline.vendor.taming_model import Decoder, Encoder
from localvqgan.pipeline.vendor.taming_quantize import GumbelQuantize, VectorQuantizer2


class VQModel(nn.Module):
    def __init__(self, ddconfig, n_embed, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer2(n_embed, embed_dim, beta=0.25,
                                         remap=None, sane_index_shape=False)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)


class GumbelVQModel(nn.Module):
    def __init__(self, ddconfig, n_embed, embed_dim, kl_weight=1e-8):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = GumbelQuantize(ddconfig["z_channels"], embed_dim,
                                       n_embed, kl_weight=kl_weight, temp_init=1.0)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)


class VQGANWrapper:
    def __init__(self, model: nn.Module, is_gumbel: bool, f: int, device: torch.device):
        self.model = model.eval().requires_grad_(False).to(device)
        self.is_gumbel = is_gumbel
        self.f = f
        self.device = device

    @property
    def codebook(self) -> torch.Tensor:
        q = self.model.quantize
        return q.embed.weight if self.is_gumbel else q.embedding.weight

    @property
    def n_toks(self) -> int:
        return self.codebook.shape[0]

    @property
    def e_dim(self) -> int:
        return self.codebook.shape[1]

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        h = self.model.encoder(img.to(self.device))
        return self.model.quant_conv(h)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.model.decoder(self.model.post_quant_conv(z_q))


def _lightning_stubs() -> list[type]:
    stubs = []
    for module, name in (
        ("pytorch_lightning.callbacks.model_checkpoint", "ModelCheckpoint"),
        ("pytorch_lightning.callbacks.early_stopping", "EarlyStopping"),
    ):
        cls = type(name, (), {})
        cls.__module__ = module
        stubs.append(cls)
    return stubs


def load_vqgan(config_path: Path, ckpt_path: Path | None, device: torch.device) -> VQGANWrapper:
    config = OmegaConf.load(config_path)
    is_gumbel = "Gumbel" in config.model.target
    params = config.model.params
    cls = GumbelVQModel if is_gumbel else VQModel
    kwargs = dict(ddconfig=OmegaConf.to_container(params.ddconfig),
                  n_embed=params.n_embed, embed_dim=params.embed_dim)
    if is_gumbel and "kl_weight" in params:
        kwargs["kl_weight"] = params.kl_weight
    model = cls(**kwargs)
    if ckpt_path is not None:
        # Lightning ckpts pickle references to callback classes; allowlist
        # inert stubs so the safe (weights_only) unpickler still applies.
        with torch.serialization.safe_globals(_lightning_stubs()):
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        # checkpoints include loss-network weights we don't define; ignore them
        model.load_state_dict(sd, strict=False)
    f = 2 ** (len(params.ddconfig.ch_mult) - 1)
    return VQGANWrapper(model, is_gumbel, f, device)

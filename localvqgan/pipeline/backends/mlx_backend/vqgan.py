from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from omegaconf import OmegaConf


def _swish(x):
    return x * mx.sigmoid(x)


def _norm(channels: int):
    return nn.GroupNorm(32, channels, eps=1e-6, pytorch_compatible=True)


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int | None = None, dropout: float):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = _norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
        self.norm2 = _norm(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, self.out_channels, 1)

    def __call__(self, x):
        h = self.norm1(x)
        h = _swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = _swish(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = _norm(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def __call__(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        b, height, width, channels = q.shape
        q = q.reshape(b, height * width, channels)
        k = k.reshape(b, height * width, channels).transpose(0, 2, 1)
        w = (q @ k) * (int(channels) ** -0.5)
        w = mx.softmax(w, axis=2)
        v = v.reshape(b, height * width, channels)
        h = (w @ v).reshape(b, height, width, channels)
        h = self.proj_out(h)
        return x + h


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def __call__(self, x):
        x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2)

    def __call__(self, x):
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, self.ch, 3, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = []
        for i_level in range(self.num_resolutions):
            block = []
            attn = []
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.norm_out = _norm(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, 3, padding=1)

    def __call__(self, x):
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = _swish(h)
        return self.conv_out(h)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        give_pre_end: bool = False,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, curr_res, curr_res, z_channels)
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        self.up = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            attn = []
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = _norm(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, padding=1)

    def __call__(self, z):
        self.last_z_shape = z.shape
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = _swish(h)
        return self.conv_out(h)


class Quantize(nn.Module):
    def __init__(self, n_embed: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_embed, embed_dim)


class MlxVQGAN(nn.Module):
    def __init__(self, ddconfig: dict, n_embed: int, embed_dim: int):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = Quantize(n_embed, embed_dim)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.f = 2 ** (len(ddconfig["ch_mult"]) - 1)
        self.n_toks = n_embed
        self.e_dim = embed_dim
        self._dtype = mx.float32

    @property
    def codebook(self):
        return self.quantize.embedding.weight.astype(mx.float32)

    def decode(self, z_q):
        z_q = z_q.astype(self._dtype)
        return self.decoder(self.post_quant_conv(z_q))

    def encode(self, img):
        img = img.astype(self._dtype)
        h = self.encoder(img)
        return self.quant_conv(h).astype(mx.float32)

    def set_dtype(self, dtype):
        if self._dtype == dtype:
            return
        self.apply(lambda p: p.astype(dtype))
        self._dtype = dtype


def load_mlx_vqgan_from_arrays(config_path: Path, weights: dict, dtype) -> MlxVQGAN:
    config = OmegaConf.load(config_path)
    params = config.model.params
    ddconfig = OmegaConf.to_container(params.ddconfig)
    model = MlxVQGAN(ddconfig, params.n_embed, params.embed_dim)
    model.update(tree_unflatten([(k, mx.array(v).astype(dtype)) for k, v in weights.items()]))
    mx.eval(model.parameters())
    model.eval().freeze()
    model._dtype = dtype
    return model

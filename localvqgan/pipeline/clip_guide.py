import open_clip
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed: torch.Tensor, weight: float = 1.0, stop: float = float("-inf")):
        super().__init__()
        self.register_buffer("embed", embed)
        self.register_buffer("weight", torch.as_tensor(weight))
        self.register_buffer("stop", torch.as_tensor(stop))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


class FastPatchEmbed(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        assert conv.stride == conv.kernel_size
        assert conv.padding == (0, 0)
        assert conv.dilation == (1, 1)
        assert conv.groups == 1
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.conv.kernel_size[0]
        B, C, H, W = x.shape
        x = x.to(dtype=self.conv.weight.dtype)
        patches = (
            x.reshape(B, C, H // k, k, W // k, k)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, (H // k) * (W // k), C * k * k)
        )
        out = patches @ self.conv.weight.reshape(self.conv.out_channels, -1).T
        if self.conv.bias is not None:
            out = out + self.conv.bias
        return out.permute(0, 2, 1).reshape(B, self.conv.out_channels, H // k, W // k)


_CLIP_NORM = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                  std=(0.26862954, 0.26130258, 0.27577711))


class ClipGuide:
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        # OpenAI weights use QuickGELU; the plain names are GELU variants
        arch = model_name if model_name.endswith("-quickgelu") else f"{model_name}-quickgelu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained="openai")
        self.model = self.model.eval().requires_grad_(False).to(device)
        if device.type == "mps":
            # Conv2d(kernel=stride=patch) backward on MPS is pathologically slow (~43 s/call, ~99% of iteration time); reshape+matmul is bit-exact and fast.
            self.model.visual.conv1 = FastPatchEmbed(self.model.visual.conv1)
        self._dtype = torch.float32
        self.tokenizer = open_clip.get_tokenizer(model_name)
        size = self.model.visual.image_size
        self.cut_size = size[0] if isinstance(size, (tuple, list)) else size

    def set_dtype(self, dtype: torch.dtype) -> None:
        if self._dtype == dtype:
            return
        self.model.to(dtype=dtype)
        self._dtype = dtype

    def embed_text(self, s: str) -> torch.Tensor:
        toks = self.tokenizer([s]).to(self.device)
        return self.model.encode_text(toks).float()

    def embed_image(self, pil_img) -> torch.Tensor:
        t = self.preprocess(pil_img).unsqueeze(0).to(self.device, dtype=self._dtype)
        return self.model.encode_image(t).float()

    def encode_cutouts(self, batch: torch.Tensor) -> torch.Tensor:
        batch = _CLIP_NORM(batch).to(dtype=self._dtype)
        return self.model.encode_image(batch).float()

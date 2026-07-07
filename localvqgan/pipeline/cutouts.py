import kornia.augmentation as K
import torch
from torch import nn
from torch.nn import functional as F


class MakeCutouts(nn.Module):
    def __init__(self, cut_size: int, cutn: int, cut_pow: float = 1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        side_y, side_x = input.shape[2:4]
        max_size = min(side_x, side_y)
        min_size = min(side_x, side_y, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            ox = int(torch.randint(0, side_x - size + 1, ()))
            oy = int(torch.randint(0, side_y - size + 1, ()))
            cut = input[:, :, oy:oy + size, ox:ox + size]
            cutouts.append(F.adaptive_avg_pool2d(cut, self.cut_size))
        batch = self.augs(torch.cat(cutouts))
        if self.noise_fac:
            facs = batch.new_empty([batch.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch

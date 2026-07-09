from dataclasses import asdict, dataclass, field


@dataclass
class GenerationSettings:
    prompts: str = ""
    width: int = 256
    height: int = 256
    iterations: int = 300
    cutouts: int = 32
    cut_pow: float = 1.0
    step_size: float = 0.1
    seed: int = -1
    init_image: str | None = None
    init_weight: float = 0.0
    image_prompts: list[str] = field(default_factory=list)
    checkpoint: str = "imagenet_16384"
    clip_model: str = "ViT-B-32"
    engine: str = "auto"  # auto | mlx | torch
    display_freq: int = 5
    precision: str = "auto"  # auto | fp16 | fp32
    # Coarse-to-fine: spend the first fraction of iterations at half resolution,
    # then upsample the latent and finish at full res. ~1.3x faster at 256² and
    # more at 512² for the same iteration/cutout/step budget. Changes the exact
    # pixels (breaks historic seed reproducibility), so it's opt-in, never the
    # default. Both engines honor it (MLX measured 1.30x at 256²; torch
    # measured smaller — its fine stage falls back to an fp32 VQGAN on MPS).
    fast_mode: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

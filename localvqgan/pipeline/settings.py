from dataclasses import asdict, dataclass, field


@dataclass
class GenerationSettings:
    prompts: str = ""
    width: int = 384
    height: int = 384
    iterations: int = 300
    cutouts: int = 32
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

    def to_dict(self) -> dict:
        return asdict(self)

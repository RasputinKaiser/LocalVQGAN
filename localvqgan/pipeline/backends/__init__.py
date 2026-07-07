import importlib.util
import platform

# Task 8 acceptance gate (imagenet_16384/ViT-B-32, 32 cutouts, M1, 2026-07-07,
# steady state: warmup 5 its then timed 20 its, torch.set_num_threads(2)):
#   256x256: torch=0.680 it/s, mlx=1.021 it/s -> ratio 1.50 (>= 1.2 gate met)
#   384x384: torch=0.283 it/s, mlx=0.015 it/s -> mlx thrashes memory at this
#   size on 8-16GB unified memory; gate is defined at 256x256/32cut per spec.
MLX_MEETS_SPEED_GATE = True

# measured 2026-07-07 on M1 16GB: mlx 1.02 it/s at 256² (1.50x torch) but
# thrashes unified memory at 384² (0.015 it/s). Only two data points exist,
# so auto stays conservative: mlx only at or below 256².
MLX_MAX_AUTO_PIXELS = 256 * 256


def mlx_available() -> bool:
    return (platform.machine() == "arm64" and platform.system() == "Darwin"
            and importlib.util.find_spec("mlx") is not None)


def mlx_supports(checkpoint: str, clip_model: str) -> bool:
    if not mlx_available():
        return False
    try:
        from localvqgan.pipeline.backends.mlx_backend import supports
    except ImportError:
        return False
    if importlib.util.find_spec("localvqgan.pipeline.backends.mlx_backend.generator") is None:
        return False  # capability map exists but the mlx generator does not yet
    return supports(checkpoint, clip_model)


def resolve_engine(engine: str, checkpoint: str, clip_model: str,
                   width: int | None = None, height: int | None = None) -> tuple[str, str]:
    if engine == "torch":
        return "torch", "explicit"
    if engine == "mlx":
        if not mlx_available():
            raise RuntimeError("mlx engine requested but mlx is not installed "
                               "(pip install -e '.[mlx]', Apple Silicon only)")
        if not mlx_supports(checkpoint, clip_model):
            raise RuntimeError(f"mlx engine does not support {checkpoint}/{clip_model}")
        return "mlx", "explicit"
    # auto
    if not mlx_available():
        return "torch", "mlx not installed"
    if not MLX_MEETS_SPEED_GATE:
        return "torch", "mlx below speed gate on this build"
    if not mlx_supports(checkpoint, clip_model):
        return "torch", f"{checkpoint} unsupported on mlx"
    if width and height and width * height > MLX_MAX_AUTO_PIXELS:
        return "torch", f"{width}x{height} exceeds mlx auto limit on this machine"
    return "mlx", "auto"


def make_generator(engine_name: str, device=None):
    if engine_name == "mlx":
        from localvqgan.pipeline.backends.mlx_backend.generator import MlxGenerator
        return MlxGenerator()
    from localvqgan.pipeline.backends.torch_backend import Generator
    return Generator(device)

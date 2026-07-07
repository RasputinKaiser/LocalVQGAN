import importlib.util
import platform

# flipped to the measured verdict in Task 8 per the spec's acceptance gate
MLX_MEETS_SPEED_GATE = True


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
    return supports(checkpoint, clip_model)


def resolve_engine(engine: str, checkpoint: str, clip_model: str) -> tuple[str, str]:
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
    return "mlx", "auto"


def make_generator(engine_name: str, device=None):
    if engine_name == "mlx":
        from localvqgan.pipeline.backends.mlx_backend.generator import MlxGenerator
        return MlxGenerator()
    from localvqgan.pipeline.backends.torch_backend import Generator
    return Generator(device)

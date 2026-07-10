import importlib.util
import platform

# Task 8 acceptance gate (imagenet_16384/ViT-B-32, 32 cutouts, M1, 2026-07-07,
# steady state: warmup 5 its then timed 20 its, torch.set_num_threads(2)):
#   256x256: the original mlx=1.021 it/s measurement was invalidated by an
#   fp16 NaN bug in the MLX VQGAN backward (clamp_with_grad dtype promotion
#   plus fp16 decoder-activation overflow at 256x256). Fixed by running VQGAN
#   in fp32 while keeping CLIP fp16, then re-measured mlx at 0.873 / 0.929 /
#   0.985 it/s across 3 runs (avg 0.929) vs unchanged torch=0.680 it/s ->
#   ratio ~1.37 (range 1.28-1.45), still >= 1.2 gate.
#   The old size cap was removed after the large-canvas MLX path gained
#   chunked cutouts, bf16 VQGAN decode, and cache limiting: 512x512 now measures
#   mlx ~0.22 it/s (0.215-0.247) while torch measures 0.010-0.011 it/s or DNF
#   from working-set pressure on 16GB unified memory.
MLX_MEETS_SPEED_GATE = True


def mlx_available() -> bool:
    return (platform.machine() == "arm64" and platform.system() == "Darwin"
            and importlib.util.find_spec("mlx") is not None)


def torch_available() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


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
        if not torch_available():
            raise RuntimeError("torch engine requested but torch is not installed "
                               "(pip install 'localvqgan[torch]')")
        return "torch", "explicit"
    if engine == "mlx":
        if not mlx_available():
            raise RuntimeError("mlx engine requested but mlx is not installed "
                               "(pip install 'localvqgan[mlx]', Apple Silicon only)")
        if not mlx_supports(checkpoint, clip_model):
            raise RuntimeError(f"mlx engine does not support {checkpoint}/{clip_model}")
        return "mlx", "explicit"
    # auto
    if not mlx_available():
        if not torch_available():
            raise RuntimeError(
                "no engine installed — pip install 'localvqgan[torch]' "
                "(any platform) or 'localvqgan[mlx]' (Apple Silicon)")
        return "torch", "mlx not installed"
    if MLX_MEETS_SPEED_GATE and mlx_supports(checkpoint, clip_model):
        return "mlx", "auto"
    reason = ("mlx below speed gate on this build" if not MLX_MEETS_SPEED_GATE
              else f"{checkpoint} unsupported on mlx")
    if not torch_available():
        raise RuntimeError(
            f"{reason}, and torch is not installed as a fallback "
            "(pip install 'localvqgan[torch]')")
    return "torch", reason


def make_generator(engine_name: str, device=None):
    if engine_name == "mlx":
        from localvqgan.pipeline.backends.mlx_backend.generator import MlxGenerator
        return MlxGenerator()
    from localvqgan.pipeline.backends.torch_backend import Generator
    return Generator(device)

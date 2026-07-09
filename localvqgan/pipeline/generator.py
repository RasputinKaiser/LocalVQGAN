from localvqgan.pipeline.backends import make_generator, resolve_engine  # noqa: F401
from localvqgan.pipeline.frames import FrameUpdate, GenerationOOM  # noqa: F401


def __getattr__(name: str):
    # Torch's Generator stays import-lazy so the slim (torch-free) MLX install
    # can import this module; accessing Generator without torch raises cleanly.
    if name == "Generator":
        from localvqgan.pipeline.backends.torch_backend import Generator
        return Generator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

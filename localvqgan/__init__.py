import os
from importlib.metadata import PackageNotFoundError, version

# grid_sampler_2d_backward (kornia affine/perspective augs) has no MPS kernel;
# let just that op fall back to CPU. Must be set before torch initializes.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    __version__ = version("localvqgan")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0.dev0"

"""Read PyTorch zip-format checkpoints (state dicts) with numpy only.

The slim Apple-Silicon install has no torch, but the historic weights only
exist as torch pickles (VQGAN ``model.ckpt``, CLIP ``pytorch_model.bin``).
Those are frozen 2020/2021 artifacts in the post-1.6 zip format: an archive
holding ``<name>/data.pkl`` plus one raw little-endian buffer per storage
under ``<name>/data/``. The pickle references tensors through persistent IDs
and ``torch._utils._rebuild_tensor(_v2)``, both reproducible with numpy.

Safety: ``find_class`` never resolves real classes — everything outside the
tensor-rebuild machinery (Lightning callback refs, optimizer state classes)
becomes an inert stub, so this is strictly narrower than ``torch.load``.
Parity with ``torch.load`` on the real checkpoints is pinned by tests.
"""
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "DoubleStorage": np.float64,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _Stub:
    """Inert stand-in for any class the pickle references but we don't need
    (Lightning callbacks, optimizer classes). Absorbs construction/state."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _make_stub(module: str, name: str) -> type:
    cls = type(name, (_Stub,), {})
    cls.__module__ = module
    return cls


class _NpStorage:
    def __init__(self, array: np.ndarray):
        self.array = array


def _rebuild_tensor(storage: _NpStorage, storage_offset, size, stride, *unused):
    arr = storage.array
    if not size:
        return np.array(arr[storage_offset])  # 0-d tensor (e.g. logit_scale)
    strides = tuple(s * arr.itemsize for s in stride)
    view = np.lib.stride_tricks.as_strided(
        arr[storage_offset:], shape=tuple(size), strides=strides)
    return np.ascontiguousarray(view)


class _TorchPickleReader(pickle.Unpickler):
    def __init__(self, file, zf: zipfile.ZipFile, prefix: str):
        super().__init__(file)
        self._zf = zf
        self._prefix = prefix

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor_v2",
                                                 "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            if name not in _STORAGE_DTYPES:
                raise pickle.UnpicklingError(
                    f"unsupported torch storage type {name} (no numpy dtype)")
            return name  # tag consumed by persistent_load
        return _make_stub(module, name)

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unexpected persistent id: {pid!r}")
        _, storage_type, key, _location, _numel = pid
        dtype = _STORAGE_DTYPES[storage_type]
        data = self._zf.read(f"{self._prefix}/data/{key}")
        return _NpStorage(np.frombuffer(data, dtype=dtype))


def load_torch_pickle(path: Path):
    """Load a zip-format torch pickle; tensors come back as numpy arrays."""
    if not zipfile.is_zipfile(path):
        raise ValueError(
            f"{path} is not a zip-format torch checkpoint (legacy pre-1.6 "
            "format?) — install 'localvqgan[torch]' to convert it")
    with zipfile.ZipFile(path) as zf:
        pkl_name = next((n for n in zf.namelist() if n.endswith("/data.pkl")),
                        None)
        if pkl_name is None:
            raise ValueError(f"{path} has no data.pkl — not a torch checkpoint")
        prefix = pkl_name[: -len("/data.pkl")]
        with zf.open(pkl_name) as fh:
            return _TorchPickleReader(fh, zf, prefix).load()


def load_torch_state_dict(path: Path) -> dict[str, np.ndarray]:
    obj = load_torch_pickle(path)
    sd = obj.get("state_dict", obj)  # Lightning ckpts nest under "state_dict"
    return {k: v for k, v in sd.items() if isinstance(v, np.ndarray)}

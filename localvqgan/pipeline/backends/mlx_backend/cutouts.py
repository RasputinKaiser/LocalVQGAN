from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class CutoutSpec:
    sizes: list[int]
    offsets: list[tuple[int, int]]
    flip: mx.array
    factor: mx.array
    sat: mx.array
    fac: mx.array
    noise: mx.array


def _resize_bilinear(img: mx.array, size: int) -> mx.array:
    n, h, w, c = img.shape
    del n, c

    def weights(src: int, dst: int) -> mx.array:
        s = (mx.arange(dst) + 0.5) * (src / dst) - 0.5
        # Clamp the sample coordinates themselves: unclamped boundary coords go
        # negative when upsampling, yielding negative weights and out-of-range output.
        s = mx.clip(s, 0, src - 1)
        i0 = mx.clip(mx.floor(s), 0, src - 1).astype(mx.int32)
        i1 = mx.minimum(i0 + 1, src - 1)
        frac = s - i0.astype(s.dtype)
        cols = mx.arange(src)[None, :]
        return (
            (cols == i0[:, None]).astype(s.dtype) * (1 - frac)[:, None]
            + (cols == i1[:, None]).astype(s.dtype) * frac[:, None]
        )

    # Matmul-form resize has a deterministic vjp on Metal; duplicate-index gathers
    # backprop through scatter-add atomics and can wobble by ~1 ULP run to run.
    out = mx.einsum("iy,nyxc->nixc", weights(h, size), img)
    return mx.einsum("jx,nixc->nijc", weights(w, size), out)


def _sharpness_kernel(channels: int) -> mx.array:
    k = mx.array([[1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 1.0]]) / 13.0
    rows = []
    for out_channel in range(channels):
        cols = []
        for in_channel in range(channels):
            cols.append(k if in_channel == out_channel else mx.zeros_like(k))
        rows.append(mx.stack(cols, axis=-1))
    return mx.stack(rows, axis=0)


def _sharpness(batch: mx.array, factor: mx.array) -> mx.array:
    # kornia sharpness blurs with a valid (unpadded) conv and leaves the outer
    # 1px ring untouched; zero-padding would darken the border ring
    kernel = _sharpness_kernel(batch.shape[-1])
    blurred = mx.conv2d(batch, kernel, padding=0)
    interior = batch[:, 1:-1, 1:-1, :]
    blended = mx.clip(
        interior + (interior - blurred) * (factor - 1.0)[:, None, None, None], 0, 1)
    out = mx.concatenate([
        batch[:, :1, :, :],
        mx.concatenate([batch[:, 1:-1, :1, :], blended, batch[:, 1:-1, -1:, :]], axis=2),
        batch[:, -1:, :, :],
    ], axis=1)
    return out


def make_cutout_spec(
    img: mx.array, cutn: int, cut_size: int, cut_pow: float = 1.0
) -> CutoutSpec:
    _, h, w, _ = img.shape
    max_size = min(h, w)
    min_size = min(h, w, cut_size)
    sizes = (mx.random.uniform(shape=(cutn,)) ** cut_pow) * (max_size - min_size) + min_size

    py_sizes = []
    offsets = []
    for i in range(cutn):
        size = int(sizes[i])
        py_sizes.append(size)
        oy = int(mx.random.randint(0, h - size + 1))
        ox = int(mx.random.randint(0, w - size + 1))
        offsets.append((oy, ox))

    flip = mx.random.uniform(shape=(cutn, 1, 1, 1)) < 0.5

    do_sharp = mx.random.uniform(shape=(cutn,)) < 0.4
    factor = mx.where(
        do_sharp,
        mx.random.uniform(low=0.7, high=1.3, shape=(cutn,)),
        mx.ones((cutn,)),
    )

    do_jit = (mx.random.uniform(shape=(cutn, 1, 1, 1)) < 0.7).astype(img.dtype)
    sat = 1.0 + (mx.random.uniform(shape=(cutn, 1, 1, 1)) * 2 - 1) * 0.01 * do_jit

    fac = mx.random.uniform(shape=(cutn, 1, 1, 1)) * 0.1
    noise = mx.random.normal((cutn, cut_size, cut_size, img.shape[-1]))
    return CutoutSpec(py_sizes, offsets, flip, factor, sat, fac, noise)


def apply_cutout_spec(
    img: mx.array, spec: CutoutSpec, cut_size: int, start: int = 0, end: int | None = None
) -> mx.array:
    end = len(spec.sizes) if end is None else end
    outs = []
    for i in range(start, end):
        size = spec.sizes[i]
        oy, ox = spec.offsets[i]
        cut = img[:, oy:oy + size, ox:ox + size, :]
        outs.append(_resize_bilinear(cut, cut_size))
    batch = mx.concatenate(outs, axis=0)

    flip = spec.flip[start:end]
    batch = mx.where(flip, batch[:, :, ::-1, :], batch)

    batch = _sharpness(batch, spec.factor[start:end])

    mean = batch.mean(axis=-1, keepdims=True)
    batch = mx.clip(mean + (batch - mean) * spec.sat[start:end], 0, 1)

    batch = batch + spec.fac[start:end] * spec.noise[start:end]
    return batch


def make_cutouts(img: mx.array, cutn: int, cut_size: int, cut_pow: float = 1.0) -> mx.array:
    spec = make_cutout_spec(img, cutn, cut_size, cut_pow)
    return apply_cutout_spec(img, spec, cut_size)

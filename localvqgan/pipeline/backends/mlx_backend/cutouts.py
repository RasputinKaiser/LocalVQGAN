import mlx.core as mx


def _resize_bilinear(img: mx.array, size: int) -> mx.array:
    n, h, w, c = img.shape
    del n, c
    ys = (mx.arange(size) + 0.5) * (h / size) - 0.5
    xs = (mx.arange(size) + 0.5) * (w / size) - 0.5
    # clamp the sample coordinates themselves: unclamped boundary coords go
    # negative when upsampling, yielding negative weights and out-of-range output
    ys = mx.clip(ys, 0, h - 1)
    xs = mx.clip(xs, 0, w - 1)
    y0 = mx.clip(mx.floor(ys), 0, h - 1).astype(mx.int32)
    x0 = mx.clip(mx.floor(xs), 0, w - 1).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, h - 1)
    x1 = mx.minimum(x0 + 1, w - 1)
    wy = (ys - y0.astype(ys.dtype))[None, :, None, None]
    wx = (xs - x0.astype(xs.dtype))[None, None, :, None]
    top = img[:, y0][:, :, x0] * (1 - wx) + img[:, y0][:, :, x1] * wx
    bot = img[:, y1][:, :, x0] * (1 - wx) + img[:, y1][:, :, x1] * wx
    return top * (1 - wy) + bot * wy


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


def make_cutouts(img: mx.array, cutn: int, cut_size: int, cut_pow: float = 1.0) -> mx.array:
    _, h, w, _ = img.shape
    max_size = min(h, w)
    min_size = min(h, w, cut_size)
    outs = []
    sizes = (mx.random.uniform(shape=(cutn,)) ** cut_pow) * (max_size - min_size) + min_size
    for i in range(cutn):
        size = int(sizes[i])
        oy = int(mx.random.randint(0, h - size + 1))
        ox = int(mx.random.randint(0, w - size + 1))
        cut = img[:, oy:oy + size, ox:ox + size, :]
        outs.append(_resize_bilinear(cut, cut_size))
    batch = mx.concatenate(outs, axis=0)

    flip = mx.random.uniform(shape=(cutn, 1, 1, 1)) < 0.5
    batch = mx.where(flip, batch[:, :, ::-1, :], batch)

    do_sharp = mx.random.uniform(shape=(cutn,)) < 0.4
    factor = mx.where(
        do_sharp,
        mx.random.uniform(low=0.7, high=1.3, shape=(cutn,)),
        mx.ones((cutn,)),
    )
    batch = _sharpness(batch, factor)

    do_jit = (mx.random.uniform(shape=(cutn, 1, 1, 1)) < 0.7).astype(batch.dtype)
    sat = 1.0 + (mx.random.uniform(shape=(cutn, 1, 1, 1)) * 2 - 1) * 0.01 * do_jit
    mean = batch.mean(axis=-1, keepdims=True)
    batch = mx.clip(mean + (batch - mean) * sat, 0, 1)

    fac = mx.random.uniform(shape=(cutn, 1, 1, 1)) * 0.1
    batch = batch + fac * mx.random.normal(batch.shape)
    return batch

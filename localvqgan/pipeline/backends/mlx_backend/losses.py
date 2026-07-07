import math

import mlx.core as mx


def replace_grad(fwd: mx.array, bwd: mx.array) -> mx.array:
    return bwd + mx.stop_gradient(fwd - bwd)


@mx.custom_function
def clamp_with_grad(x, lo, hi):
    return mx.clip(x, lo, hi)


@clamp_with_grad.vjp
def _clamp_vjp(primals, cotangent, output):
    x, lo, hi = primals
    clamped = mx.clip(x, lo, hi)
    keep = (cotangent * (x - clamped)) >= 0
    return cotangent * keep, None, None


def vector_quantize(x: mx.array, codebook: mx.array) -> mx.array:
    d = (
        (x**2).sum(axis=-1, keepdims=True)
        + (codebook**2).sum(axis=1)
        - 2 * x @ codebook.T
    )
    indices = d.argmin(axis=-1)
    x_q = codebook[indices]
    return replace_grad(x_q, x)


def _normalize(v: mx.array) -> mx.array:
    return v / mx.maximum(mx.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def prompt_loss(embeds: mx.array, target: mx.array, weight: float, stop: float) -> mx.array:
    a = _normalize(embeds)[:, None, :]
    b = _normalize(target)[None, :, :]
    dists = mx.arcsin(mx.linalg.norm(a - b, axis=2) / 2) ** 2 * 2
    dists = dists * math.copysign(1.0, weight)
    stopped = mx.maximum(dists, mx.array(stop))
    return abs(weight) * replace_grad(dists, stopped).mean()

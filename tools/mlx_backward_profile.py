#!/usr/bin/env python3
"""Profile MLX VQGAN+CLIP backward attribution without changing generation code."""

from __future__ import annotations

import argparse
import os
import statistics
import time

import mlx.core as mx

from localvqgan.pipeline.backends.mlx_backend.cutouts import make_cutouts
from localvqgan.pipeline.backends.mlx_backend.generator import MlxGenerator
from localvqgan.pipeline.backends.mlx_backend.losses import prompt_loss
from localvqgan.pipeline.prompts import parse_prompts
from localvqgan.pipeline.settings import GenerationSettings


def _time_ms(fn, runs: int) -> list[float]:
    values = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        values.append((time.perf_counter() - t0) * 1000)
    return values


def _summarize(values: list[float]) -> tuple[float, float, float]:
    return statistics.mean(values), min(values), max(values)


def _print_row(label: str, values: list[float]) -> None:
    mean, lo, hi = _summarize(values)
    print(f"{label:28s} mean={mean:8.1f} ms  min={lo:8.1f}  max={hi:8.1f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="imagenet_16384")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--prompt", default="a lighthouse on a cliff at dusk")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--cutouts", type=int, default=32)
    parser.add_argument("--cut-pow", type=float, default=1.0)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    if args.width * args.height > 256 * 256:
        parser.error("this profiler is for the 256px attribution path; use generation benchmarks for large canvases")

    os.environ.setdefault("LOCALVQGAN_MLX_COMPILE", "0")
    settings = GenerationSettings(
        prompts=args.prompt,
        width=args.width,
        height=args.height,
        iterations=1,
        cutouts=args.cutouts,
        cut_pow=args.cut_pow,
        step_size=args.step_size,
        seed=args.seed,
        checkpoint=args.checkpoint,
        clip_model=args.clip_model,
        engine="mlx",
        display_freq=1,
    )

    generator = MlxGenerator()
    generator.load(args.checkpoint, args.clip_model)
    generator.vqgan.set_dtype(mx.float32)
    generator.clip.set_dtype(mx.float16)

    mx.random.seed(args.seed)
    z = mx.array(generator._init_z(settings), dtype=mx.float32)
    targets = [
        (generator.clip.embed_text(prompt.text), prompt.weight, prompt.stop)
        for prompt in parse_prompts(settings.prompts)
    ]
    mx.eval(z, generator.vqgan.parameters(), generator.clip.model.parameters(), *[t for t, _, _ in targets])

    def synth():
        out = generator._synth(z)
        mx.eval(out)
        return out

    out = synth()

    def clip_cutout_value_and_grad():
        mx.random.seed(args.seed)

        def branch_loss(out_):
            cutouts = make_cutouts(out_, settings.cutouts, generator.clip.cut_size, settings.cut_pow)
            embeds = generator.clip.encode_cutouts(cutouts)
            losses = [prompt_loss(embeds, target, weight, stop) for target, weight, stop in targets]
            return sum(losses, mx.array(0.0))

        loss, grad_out = mx.value_and_grad(branch_loss)(out)
        mx.eval(loss, grad_out)
        return loss, grad_out

    _, grad_out = clip_cutout_value_and_grad()

    def synth_pullback():
        grad_z = generator._synth_pullback(z, grad_out)
        mx.eval(grad_z)
        return grad_z

    def two_stage_backward():
        _, local_grad_out = clip_cutout_value_and_grad()
        grad_z = generator._synth_pullback(z, local_grad_out)
        mx.eval(grad_z)

    def full_value_and_grad():
        mx.random.seed(args.seed)

        def loss_fn(z_):
            local_out = generator._synth(z_)
            cutouts = make_cutouts(
                local_out,
                settings.cutouts,
                generator.clip.cut_size,
                settings.cut_pow,
            )
            embeds = generator.clip.encode_cutouts(cutouts)
            losses = [prompt_loss(embeds, target, weight, stop) for target, weight, stop in targets]
            return sum(losses, mx.array(0.0))

        loss, grad_z = mx.value_and_grad(loss_fn)(z)
        mx.eval(loss, grad_z)

    for _ in range(args.warmup):
        full_value_and_grad()
        out = synth()
        clip_cutout_value_and_grad()
        synth_pullback()
        two_stage_backward()

    print(
        f"MLX backward attribution: {args.width}x{args.height}, "
        f"{args.cutouts} cutouts, warmup={args.warmup}, runs={args.runs}",
        flush=True,
    )
    _print_row("synth forward", _time_ms(synth, args.runs))
    _print_row("clip+cutouts backward", _time_ms(clip_cutout_value_and_grad, args.runs))
    _print_row("synth pullback", _time_ms(synth_pullback, args.runs))
    _print_row("two-stage backward", _time_ms(two_stage_backward, args.runs))
    _print_row("full value_and_grad", _time_ms(full_value_and_grad, args.runs))
    mx.clear_cache()


if __name__ == "__main__":
    main()

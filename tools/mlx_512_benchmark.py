#!/usr/bin/env python3
"""Run MLX benchmark legs with parity and swap guards."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
import re
import subprocess
import time
from collections.abc import Iterator

import numpy as np

from localvqgan.pipeline.backends.mlx_backend.generator import MlxGenerator
from localvqgan.pipeline.settings import GenerationSettings


ENV_KEYS = (
    "LOCALVQGAN_MLX_COMPILE",
    "LOCALVQGAN_MLX_ASYNC_CHUNKS",
    "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE",
    "LOCALVQGAN_MLX_CACHE_LIMIT_GB",
    "LOCALVQGAN_MLX_COMPILED_OPTIMIZER",
    "LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE",
    "LOCALVQGAN_MLX_SINGLE_LOSS_SUM_FASTPATH",
    "LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE",
    "LOCALVQGAN_MLX_RESIZE_IDENTITY_SKIP",
)


@dataclasses.dataclass(frozen=True)
class Leg:
    name: str
    env: dict[str, str | None]


@dataclasses.dataclass
class LegResult:
    name: str
    seconds: float
    iterations: int
    loss: float | None
    image: np.ndarray
    swap_before_mb: float
    swap_after_mb: float

    @property
    def it_per_sec(self) -> float:
        return self.iterations / self.seconds


def swap_used_mb() -> float:
    out = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    match = re.search(r"used = ([0-9.]+)M", out)
    if not match:
        raise RuntimeError(f"could not parse swap usage: {out.strip()}")
    return float(match.group(1))


@contextlib.contextmanager
def patched_env(values: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in ENV_KEYS}
    try:
        for key in ENV_KEYS:
            if key in values:
                value = values[key]
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_settings(args: argparse.Namespace, iterations: int) -> GenerationSettings:
    display_freq = args.display_freq if args.display_freq > 0 else iterations
    return GenerationSettings(
        prompts=args.prompt,
        width=args.width,
        height=args.height,
        iterations=iterations,
        cutouts=args.cutouts,
        cut_pow=args.cut_pow,
        step_size=args.step_size,
        seed=args.seed,
        checkpoint=args.checkpoint,
        clip_model=args.clip_model,
        engine="mlx",
        display_freq=display_freq,
    )


def run_generation(args: argparse.Namespace, iterations: int) -> tuple[float, np.ndarray]:
    settings = make_settings(args, iterations)
    generator = MlxGenerator()
    generator.load(args.checkpoint, args.clip_model)
    frames = list(generator.generate(settings))
    frame = frames[-1]
    if frame.image is None or frame.loss is None:
        raise RuntimeError("benchmark generation did not produce a final image/loss")
    return float(frame.loss), np.asarray(frame.image)


def run_leg(args: argparse.Namespace, leg: Leg) -> LegResult:
    with patched_env(leg.env):
        if args.warmup:
            run_generation(args, args.warmup)
        swap_before = swap_used_mb()
        if swap_before > args.max_swap_mb and not args.allow_dirty_swap:
            raise RuntimeError(
                f"{leg.name} warmup raised swap to {swap_before:.0f} MB, "
                f"above max {args.max_swap_mb:.0f} MB; treat output as non-canonical"
            )
        start = time.perf_counter()
        loss, image = run_generation(args, args.timed)
        seconds = time.perf_counter() - start
        swap_after = swap_used_mb()
    return LegResult(
        name=leg.name,
        seconds=seconds,
        iterations=args.timed,
        loss=loss,
        image=image,
        swap_before_mb=swap_before,
        swap_after_mb=swap_after,
    )


def print_result(result: LegResult) -> None:
    print(
        f"{result.name:24s} {result.it_per_sec:7.3f} it/s "
        f"({result.seconds:7.2f}s / {result.iterations} it) "
        f"loss={result.loss:.8f} "
        f"swap={result.swap_before_mb:.0f}->{result.swap_after_mb:.0f} MB",
        flush=True,
    )


def compare(reference: LegResult, candidate: LegResult) -> tuple[float, int, bool]:
    loss_diff = abs(float(reference.loss) - float(candidate.loss))
    image_diff = int(
        np.max(
            np.abs(
                reference.image.astype(np.int16)
                - candidate.image.astype(np.int16)
            )
        )
    )
    return loss_diff, image_diff, np.array_equal(reference.image, candidate.image)


def legs(args: argparse.Namespace) -> list[Leg]:
    if args.width * args.height <= 256 * 256:
        if args.candidate == "resize-identity-skip":
            no_skip = Leg(
                "no-resize-identity-skip",
                {
                    "LOCALVQGAN_MLX_COMPILE": None,
                    "LOCALVQGAN_MLX_ASYNC_CHUNKS": None,
                    "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(args.chunk_size),
                    "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(args.cache_gb),
                    "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": "0",
                    "LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE": "1",
                    "LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE": "1",
                    "LOCALVQGAN_MLX_RESIZE_IDENTITY_SKIP": "0",
                },
            )
            skip = Leg(
                "resize-identity-skip",
                {
                    "LOCALVQGAN_MLX_COMPILE": None,
                    "LOCALVQGAN_MLX_ASYNC_CHUNKS": None,
                    "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(args.chunk_size),
                    "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(args.cache_gb),
                    "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": "0",
                    "LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE": "1",
                    "LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE": "1",
                    "LOCALVQGAN_MLX_RESIZE_IDENTITY_SKIP": "1",
                },
            )
            return [no_skip, skip]

        if args.candidate == "single-loss-sum":
            list_sum = Leg(
                "list-sum-loss",
                {
                    "LOCALVQGAN_MLX_COMPILE": None,
                    "LOCALVQGAN_MLX_ASYNC_CHUNKS": None,
                    "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(args.chunk_size),
                    "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(args.cache_gb),
                    "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": "0",
                    "LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE": "1",
                    "LOCALVQGAN_MLX_SINGLE_LOSS_SUM_FASTPATH": "0",
                    "LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE": "1",
                    "LOCALVQGAN_MLX_RESIZE_IDENTITY_SKIP": "1",
                },
            )
            fastpath = Leg(
                "single-loss-sum",
                {
                    "LOCALVQGAN_MLX_COMPILE": None,
                    "LOCALVQGAN_MLX_ASYNC_CHUNKS": None,
                    "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(args.chunk_size),
                    "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(args.cache_gb),
                    "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": "0",
                    "LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE": "1",
                    "LOCALVQGAN_MLX_SINGLE_LOSS_SUM_FASTPATH": "1",
                    "LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE": "1",
                    "LOCALVQGAN_MLX_RESIZE_IDENTITY_SKIP": "1",
                },
            )
            return [list_sum, fastpath]

        if args.candidate == "sharpness-cache":
            no_cache = Leg(
                "no-sharpness-cache",
                {
                    "LOCALVQGAN_MLX_COMPILE": None,
                    "LOCALVQGAN_MLX_ASYNC_CHUNKS": None,
                    "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(args.chunk_size),
                    "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(args.cache_gb),
                    "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": "0",
                    "LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE": "1",
                    "LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE": "0",
                },
            )
            cache = Leg(
                "sharpness-cache",
                {
                    "LOCALVQGAN_MLX_COMPILE": None,
                    "LOCALVQGAN_MLX_ASYNC_CHUNKS": None,
                    "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(args.chunk_size),
                    "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(args.cache_gb),
                    "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": "0",
                    "LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE": "1",
                    "LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE": "1",
                },
            )
            return [no_cache, cache]

        base = Leg(
            "previous-default",
            {
                "LOCALVQGAN_MLX_COMPILE": None,
                "LOCALVQGAN_MLX_ASYNC_CHUNKS": None,
                "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(args.chunk_size),
                "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(args.cache_gb),
                "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": "0",
                "LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE": "0",
            },
        )
        if args.candidate == "compiled-optimizer":
            candidate = Leg(
                "compiled-optimizer",
                {
                    "LOCALVQGAN_MLX_COMPILE": None,
                    "LOCALVQGAN_MLX_ASYNC_CHUNKS": None,
                    "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(args.chunk_size),
                    "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(args.cache_gb),
                    "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": "1",
                    "LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE": "0",
                },
            )
        else:
            candidate = Leg(
                "small-preview-reuse",
                {
                    "LOCALVQGAN_MLX_COMPILE": None,
                    "LOCALVQGAN_MLX_ASYNC_CHUNKS": None,
                    "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(args.chunk_size),
                    "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(args.cache_gb),
                    "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": "0",
                    "LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE": "1",
                },
            )
        return [base, candidate]

    base = Leg(
        "baseline-sync-eager",
        {
            "LOCALVQGAN_MLX_COMPILE": "0",
            "LOCALVQGAN_MLX_ASYNC_CHUNKS": "0",
            "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(args.chunk_size),
            "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(args.cache_gb),
            "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": "0",
            "LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE": "0",
        },
    )
    current = Leg(
        "current-defaults",
        {
            "LOCALVQGAN_MLX_COMPILE": None,
            "LOCALVQGAN_MLX_ASYNC_CHUNKS": None,
            "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(args.chunk_size),
            "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(args.cache_gb),
            "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": None,
            "LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE": None,
        },
    )
    if not args.sweep:
        return [base, current]

    sweep_legs = [base]
    for chunk_size in args.sweep_chunk_sizes:
        for cache_gb in args.sweep_cache_gb:
            sweep_legs.append(
                Leg(
                    f"chunk{chunk_size}-cache{cache_gb:g}",
                    {
                        "LOCALVQGAN_MLX_COMPILE": None,
                        "LOCALVQGAN_MLX_ASYNC_CHUNKS": None,
                        "LOCALVQGAN_MLX_CUTOUT_CHUNK_SIZE": str(chunk_size),
                        "LOCALVQGAN_MLX_CACHE_LIMIT_GB": str(cache_gb),
                        "LOCALVQGAN_MLX_COMPILED_OPTIMIZER": None,
                        "LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE": None,
                    },
                )
            )
    return sweep_legs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="imagenet_16384")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--prompt", default="a lighthouse on a cliff at dusk")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--cutouts", type=int, default=32)
    parser.add_argument("--cut-pow", type=float, default=1.0)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timed", type=int, default=20)
    parser.add_argument("--display-freq", type=int, default=0)
    parser.add_argument(
        "--candidate",
        choices=[
            "compiled-optimizer",
            "small-preview-reuse",
            "sharpness-cache",
            "resize-identity-skip",
            "single-loss-sum",
        ],
        default="small-preview-reuse",
    )
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--cache-gb", type=float, default=4)
    parser.add_argument("--max-swap-mb", type=float, default=10 * 1024)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-chunk-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--sweep-cache-gb", type=float, nargs="+", default=[2, 4, 6])
    parser.add_argument("--allow-dirty-swap", action="store_true")
    return parser


def main_args_for_test(argv: list[str]) -> None:
    run(build_parser().parse_args(argv))


def main() -> None:
    run(build_parser().parse_args())


def run(args: argparse.Namespace) -> None:
    start_swap = swap_used_mb()
    if start_swap > args.max_swap_mb and not args.allow_dirty_swap:
        raise SystemExit(
            f"swap is {start_swap:.0f} MB, above max {args.max_swap_mb:.0f} MB; "
            "rerun later or pass --allow-dirty-swap for diagnostic-only output"
        )

    print(
        f"MLX benchmark: {args.width}x{args.height}, "
        f"{args.cutouts} cutouts, warmup={args.warmup}, timed={args.timed}, "
        f"display_freq={args.display_freq if args.display_freq > 0 else args.timed}, "
        f"start_swap={start_swap:.0f} MB",
        flush=True,
    )

    results: list[LegResult] = []
    for leg in legs(args):
        result = run_leg(args, leg)
        print_result(result)
        results.append(result)

    reference = results[0]
    for result in results[1:]:
        loss_diff, image_diff, image_equal = compare(reference, result)
        print(
            f"parity vs {reference.name}: {result.name} "
            f"loss_diff={loss_diff:.8g} image_equal={image_equal} "
            f"image_max_abs={image_diff}",
            flush=True,
        )
        if loss_diff > 1e-5 or not image_equal:
            raise SystemExit(1)

    worst_swap_after = max(result.swap_after_mb for result in results)
    if worst_swap_after > args.max_swap_mb and not args.allow_dirty_swap:
        raise SystemExit(
            f"swap ended at {worst_swap_after:.0f} MB, above max {args.max_swap_mb:.0f} MB; "
            "treat output as non-canonical"
        )


if __name__ == "__main__":
    main()

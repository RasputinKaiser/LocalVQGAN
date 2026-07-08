import os

import numpy as np
import pytest

from tools import mlx_512_benchmark as bench


def test_patched_env_restores_values(monkeypatch):
    monkeypatch.setenv("LOCALVQGAN_MLX_COMPILE", "before")
    monkeypatch.delenv("LOCALVQGAN_MLX_ASYNC_CHUNKS", raising=False)

    with bench.patched_env({
        "LOCALVQGAN_MLX_COMPILE": "0",
        "LOCALVQGAN_MLX_ASYNC_CHUNKS": "0",
    }):
        assert os.environ["LOCALVQGAN_MLX_COMPILE"] == "0"
        assert os.environ["LOCALVQGAN_MLX_ASYNC_CHUNKS"] == "0"

    assert os.environ["LOCALVQGAN_MLX_COMPILE"] == "before"
    assert "LOCALVQGAN_MLX_ASYNC_CHUNKS" not in os.environ


def test_compare_reports_loss_and_image_parity():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    ref = bench.LegResult("ref", 1.0, 1, 0.5, image, 0.0, 0.0)
    same = bench.LegResult("same", 1.0, 1, 0.5, image.copy(), 0.0, 0.0)
    changed_image = image.copy()
    changed_image[0, 0, 0] = 2
    changed = bench.LegResult("changed", 1.0, 1, 0.75, changed_image, 0.0, 0.0)

    assert bench.compare(ref, same) == (0.0, 0, True)
    assert bench.compare(ref, changed) == (0.25, 2, False)


def test_dirty_swap_refuses_run(monkeypatch):
    monkeypatch.setattr(bench, "swap_used_mb", lambda: 20_000.0)

    with pytest.raises(SystemExit) as exc:
        bench.main_args_for_test(["--max-swap-mb", "1"])

    assert "above max" in str(exc.value)


def test_warmup_swap_guard_refuses_timed_leg(monkeypatch):
    calls = []

    def fake_run_generation(args, iterations):
        calls.append(iterations)
        return 0.5, np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(bench, "run_generation", fake_run_generation)
    monkeypatch.setattr(bench, "swap_used_mb", lambda: 20_000.0)

    args = bench.build_parser().parse_args([
        "--warmup", "3",
        "--timed", "10",
        "--max-swap-mb", "10240",
    ])
    leg = bench.Leg("test-leg", {})

    with pytest.raises(RuntimeError, match="warmup raised swap"):
        bench.run_leg(args, leg)

    assert calls == [3]


def test_256_legs_compare_preview_reuse_against_previous_default():
    args = bench.build_parser().parse_args([
        "--width", "256",
        "--height", "256",
        "--candidate", "small-preview-reuse",
    ])

    legs = bench.legs(args)

    assert [leg.name for leg in legs] == ["previous-default", "small-preview-reuse"]
    assert legs[0].env["LOCALVQGAN_MLX_COMPILE"] is None
    assert legs[0].env["LOCALVQGAN_MLX_COMPILED_OPTIMIZER"] == "0"
    assert legs[0].env["LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE"] == "0"
    assert legs[1].env["LOCALVQGAN_MLX_COMPILED_OPTIMIZER"] == "0"
    assert legs[1].env["LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE"] == "1"


def test_256_legs_can_compare_optimizer_fold_against_previous_default():
    args = bench.build_parser().parse_args([
        "--width", "256",
        "--height", "256",
        "--candidate", "compiled-optimizer",
    ])

    legs = bench.legs(args)

    assert [leg.name for leg in legs] == ["previous-default", "compiled-optimizer"]
    assert legs[0].env["LOCALVQGAN_MLX_COMPILED_OPTIMIZER"] == "0"
    assert legs[1].env["LOCALVQGAN_MLX_COMPILED_OPTIMIZER"] == "1"
    assert legs[1].env["LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE"] == "0"


def test_256_legs_can_compare_sharpness_cache_against_no_cache():
    args = bench.build_parser().parse_args([
        "--width", "256",
        "--height", "256",
        "--candidate", "sharpness-cache",
    ])

    legs = bench.legs(args)

    assert [leg.name for leg in legs] == ["no-sharpness-cache", "sharpness-cache"]
    assert legs[0].env["LOCALVQGAN_MLX_COMPILED_OPTIMIZER"] == "0"
    assert legs[1].env["LOCALVQGAN_MLX_COMPILED_OPTIMIZER"] == "0"
    assert legs[0].env["LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE"] == "1"
    assert legs[1].env["LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE"] == "1"
    assert legs[0].env["LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE"] == "0"
    assert legs[1].env["LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE"] == "1"


def test_256_legs_can_compare_resize_identity_skip_against_no_skip():
    args = bench.build_parser().parse_args([
        "--width", "256",
        "--height", "256",
        "--candidate", "resize-identity-skip",
    ])

    legs = bench.legs(args)

    assert [leg.name for leg in legs] == [
        "no-resize-identity-skip",
        "resize-identity-skip",
    ]
    assert legs[0].env["LOCALVQGAN_MLX_COMPILED_OPTIMIZER"] == "0"
    assert legs[1].env["LOCALVQGAN_MLX_COMPILED_OPTIMIZER"] == "0"
    assert legs[0].env["LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE"] == "1"
    assert legs[1].env["LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE"] == "1"
    assert legs[0].env["LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE"] == "1"
    assert legs[1].env["LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE"] == "1"
    assert legs[0].env["LOCALVQGAN_MLX_RESIZE_IDENTITY_SKIP"] == "0"
    assert legs[1].env["LOCALVQGAN_MLX_RESIZE_IDENTITY_SKIP"] == "1"


def test_256_legs_can_compare_single_loss_sum_against_list_sum():
    args = bench.build_parser().parse_args([
        "--width", "256",
        "--height", "256",
        "--candidate", "single-loss-sum",
    ])

    legs = bench.legs(args)

    assert [leg.name for leg in legs] == ["list-sum-loss", "single-loss-sum"]
    assert legs[0].env["LOCALVQGAN_MLX_COMPILED_OPTIMIZER"] == "0"
    assert legs[1].env["LOCALVQGAN_MLX_COMPILED_OPTIMIZER"] == "0"
    assert legs[0].env["LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE"] == "1"
    assert legs[1].env["LOCALVQGAN_MLX_SMALL_PREVIEW_REUSE"] == "1"
    assert legs[0].env["LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE"] == "1"
    assert legs[1].env["LOCALVQGAN_MLX_SHARPNESS_KERNEL_CACHE"] == "1"
    assert legs[0].env["LOCALVQGAN_MLX_RESIZE_IDENTITY_SKIP"] == "1"
    assert legs[1].env["LOCALVQGAN_MLX_RESIZE_IDENTITY_SKIP"] == "1"
    assert legs[0].env["LOCALVQGAN_MLX_SINGLE_LOSS_SUM_FASTPATH"] == "0"
    assert legs[1].env["LOCALVQGAN_MLX_SINGLE_LOSS_SUM_FASTPATH"] == "1"

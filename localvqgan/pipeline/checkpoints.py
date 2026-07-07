from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    config_url: str
    ckpt_url: str
    size_mb: int
    # True when every known public mirror is dead; manual install into the
    # cache dir still works and is_downloaded() will pick it up.
    mirror_offline: bool = False


# URLs extracted from justinjohn0306/VQGAN-CLIP "VQGAN+CLIP(Updated).ipynb".
# size_mb is informational (approximate download size of the ckpt).
CHECKPOINTS: dict[str, CheckpointSpec] = {
    "imagenet_1024": CheckpointSpec(
        "imagenet_1024",
        "https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1",
        "https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
        913),
    "imagenet_16384": CheckpointSpec(
        "imagenet_16384",
        "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1",
        "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
        934),
    "gumbel_8192": CheckpointSpec(
        "gumbel_8192",
        "https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1",
        "https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
        359),
    "coco": CheckpointSpec(
        "coco",
        "https://dl.nmkd.de/ai/clip/coco/coco.yaml",
        "https://dl.nmkd.de/ai/clip/coco/coco.ckpt",
        8045),
    "faceshq": CheckpointSpec(
        "faceshq",
        "https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT",
        "https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt",
        3800, mirror_offline=True),
    "wikiart_1024": CheckpointSpec(
        "wikiart_1024",
        "https://github.com/pixray/pixray/releases/download/v1.7.1/vqgan_wikiart_1024.yaml",
        "https://github.com/pixray/pixray/releases/download/v1.7.1/vqgan_wikiart_1024.ckpt",
        913),
    "wikiart_16384": CheckpointSpec(
        "wikiart_16384",
        "https://github.com/pixray/pixray/releases/download/v1.7.1/vqgan_wikiart_16384.yaml",
        "https://github.com/pixray/pixray/releases/download/v1.7.1/vqgan_wikiart_16384.ckpt",
        958),
    "sflckr": CheckpointSpec(
        "sflckr",
        "https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1",
        "https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1",
        4066),
    "ade20k": CheckpointSpec(
        "ade20k",
        "https://static.miraheze.org/intercriaturaswiki/b/bf/Ade20k.txt",
        "https://app.koofr.net/content/links/0f65c2cd-7102-4550-a2bd-07fd383aac9e/files/get/last.ckpt?path=%2F2020-11-20T21-45-44_ade20k_transformer%2Fcheckpoints%2Flast.ckpt",
        3700, mirror_offline=True),
    "ffhq": CheckpointSpec(
        "ffhq",
        "https://app.koofr.net/content/links/0fc005bf-3dca-4079-9d40-cdf38d42cd7a/files/get/2021-04-23T18-19-01-project.yaml?path=%2F2021-04-23T18-19-01_ffhq_transformer%2Fconfigs%2F2021-04-23T18-19-01-project.yaml&force",
        "https://app.koofr.net/content/links/0fc005bf-3dca-4079-9d40-cdf38d42cd7a/files/get/last.ckpt?path=%2F2021-04-23T18-19-01_ffhq_transformer%2Fcheckpoints%2Flast.ckpt&force",
        3500, mirror_offline=True),
    "celebahq": CheckpointSpec(
        "celebahq",
        "https://app.koofr.net/content/links/6dddf083-40c8-470a-9360-a9dab2a94e96/files/get/2021-04-23T18-11-19-project.yaml?path=%2F2021-04-23T18-11-19_celebahq_transformer%2Fconfigs%2F2021-04-23T18-11-19-project.yaml&force",
        "https://app.koofr.net/content/links/6dddf083-40c8-470a-9360-a9dab2a94e96/files/get/last.ckpt?path=%2F2021-04-23T18-11-19_celebahq_transformer%2Fcheckpoints%2Flast.ckpt&force",
        3400, mirror_offline=True),
}


def cache_dir() -> Path:
    return Path.home() / ".cache" / "localvqgan"


def checkpoint_paths(name: str) -> tuple[Path, Path]:
    d = cache_dir() / name
    return d / "config.yaml", d / "model.ckpt"


def is_downloaded(name: str) -> bool:
    cfg, ckpt = checkpoint_paths(name)
    return cfg.exists() and ckpt.exists() and ckpt.stat().st_size > 0


def _stream_to_file(url: str, dest: Path, cb: Callable[[int, int], None]) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, headers=headers, timeout=30) as r:
        if r.status_code == 416:  # already complete
            part.rename(dest)
            return
        r.raise_for_status()
        if r.status_code != 206:
            done = 0  # server ignored Range; restart
        total = done + int(r.headers.get("content-length", 0))
        mode = "ab" if done and r.status_code == 206 else "wb"
        with open(part, mode) as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                cb(done, total)
    part.rename(dest)


def download(name: str, progress_cb: Callable[[str, int, int], None]) -> None:
    spec = CHECKPOINTS[name]
    cfg, ckpt = checkpoint_paths(name)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.exists():
        _stream_to_file(spec.config_url, cfg, lambda d, t: None)
    if not (ckpt.exists() and ckpt.stat().st_size > 0):
        _stream_to_file(spec.ckpt_url, ckpt, lambda d, t: progress_cb(name, d, t))

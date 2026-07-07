import localvqgan.pipeline.checkpoints as cp


def test_registry_complete():
    for name in ["imagenet_1024", "imagenet_16384", "wikiart_1024", "wikiart_16384",
                 "coco", "faceshq", "sflckr", "ade20k", "ffhq", "celebahq", "gumbel_8192"]:
        assert name in cp.CHECKPOINTS


def test_paths_and_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "cache_dir", lambda: tmp_path)
    cfg, ckpt = cp.checkpoint_paths("imagenet_16384")
    assert not cp.is_downloaded("imagenet_16384")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("x")
    ckpt.write_bytes(b"x")
    assert cp.is_downloaded("imagenet_16384")


def test_download_resumes(tmp_path, monkeypatch):
    calls = []

    def fake_stream(url, dest, cb):
        calls.append((url, dest.name))
        dest.write_bytes(b"data")

    monkeypatch.setattr(cp, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(cp, "_stream_to_file", fake_stream)
    cp.download("imagenet_16384", progress_cb=lambda *a: None)
    assert len(calls) == 2  # config + ckpt

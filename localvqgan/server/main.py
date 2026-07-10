import argparse
import socket
import threading
import webbrowser
from pathlib import Path

import uvicorn

from localvqgan.pipeline.backends import make_generator, resolve_engine
from localvqgan.pipeline.settings import GenerationSettings
from localvqgan.server.app import create_app
from localvqgan.server.jobs import JobManager


def default_manager(outputs_root: Path) -> JobManager:
    # Warm the engine auto would pick for the default settings, so a slim
    # (torch-free) MLX install never imports torch and a torch-only install
    # never probes mlx.
    defaults = GenerationSettings()
    engine, _ = resolve_engine("auto", defaults.checkpoint, defaults.clip_model)
    return JobManager(lambda: make_generator(engine), outputs_root,
                      default_engine=engine)


def _free_port(preferred: int = 8420) -> int:
    # A second launch (or another app on 8420) shouldn't crash with a bind
    # error — walk forward to the next free port and say so.
    for port in range(preferred, preferred + 20):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    return preferred


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="localvqgan",
        description="The classic 2021 VQGAN+CLIP aesthetic, locally with a web GUI")
    parser.add_argument("--port", type=int, default=8420,
                        help="preferred port (falls forward if taken; default 8420)")
    parser.add_argument("--outputs", type=Path, default=Path.cwd() / "outputs",
                        help="where run folders are written (default ./outputs)")
    parser.add_argument("--no-browser", action="store_true",
                        help="don't open a browser tab on start")
    args = parser.parse_args(argv)

    manager = default_manager(args.outputs)
    app = create_app(manager)
    port = _free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    note = "" if port == args.port else f" ({args.port} was taken)"
    print(f"LocalVQGAN running at {url}{note}", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port)

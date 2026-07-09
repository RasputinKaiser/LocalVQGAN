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


def run() -> None:
    manager = default_manager(Path.cwd() / "outputs")
    app = create_app(manager)
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    note = "" if port == 8420 else " (8420 was taken)"
    print(f"LocalVQGAN running at {url}{note}", flush=True)
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port)

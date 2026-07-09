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


def run() -> None:
    manager = default_manager(Path.cwd() / "outputs")
    app = create_app(manager)
    print("LocalVQGAN running at http://127.0.0.1:8420", flush=True)
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8420")).start()
    uvicorn.run(app, host="127.0.0.1", port=8420)

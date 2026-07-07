import threading
import webbrowser
from pathlib import Path

import uvicorn

from localvqgan.pipeline.generator import Generator
from localvqgan.server.app import create_app
from localvqgan.server.jobs import JobManager


def run() -> None:
    manager = JobManager(Generator, Path.cwd() / "outputs")
    app = create_app(manager)
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8420")).start()
    uvicorn.run(app, host="127.0.0.1", port=8420)

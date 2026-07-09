import asyncio
import base64
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from localvqgan.pipeline import checkpoints
from localvqgan.pipeline.outputs import RunWriter, list_runs
from localvqgan.server.jobs import Busy, JobManager

WEB_DIR = Path(__file__).parent.parent / "web"


def _max_side(total_ram_gb: float) -> int:
    if total_ram_gb >= 32:
        return 896
    if total_ram_gb >= 16:
        return 640
    return 448


def create_app(manager: JobManager) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.attach_loop(asyncio.get_running_loop())
        yield

    app = FastAPI(title="LocalVQGAN", lifespan=lifespan)

    @app.get("/api/status")
    def status():
        return manager.status()

    @app.post("/api/jobs")
    def start_job(body: dict):
        settings = body.get("settings", {})
        try:
            for key, target in (("init_image_data", "init_image"),
                                ("image_prompt_data", "image_prompts")):
                data_url = body.get(key)
                if not data_url:
                    continue
                _, b64data = data_url.split(",", 1)
                updir = manager.outputs_root / "_uploads"
                updir.mkdir(parents=True, exist_ok=True)
                n = len(list(updir.glob("*"))) + 1
                p = updir / f"upload-{n:04d}.png"
                p.write_bytes(base64.b64decode(b64data))
                if target == "image_prompts":
                    settings.setdefault("image_prompts", []).append(str(p))
                else:
                    settings[target] = str(p)
            if body.get("type") == "animation":
                run_id = manager.start_animation(settings,
                                                 body.get("keyframes", []))
            else:
                run_id = manager.start_still(settings)
        except Busy:
            raise HTTPException(409, "A job is already running")
        except (TypeError, ValueError) as e:
            raise HTTPException(422, f"Bad request: {e}")
        return {"run_id": run_id}

    @app.post("/api/jobs/cancel")
    def cancel():
        manager.cancel()
        return {"ok": True}

    @app.get("/api/checkpoints")
    def list_checkpoints():
        # converted MLX weights count as downloaded: a slim install may never
        # have (or need) the original torch ckpt
        return [{"name": s.name, "size_mb": s.size_mb,
                 "downloaded": (checkpoints.is_downloaded(s.name)
                                or checkpoints.has_mlx_weights(s.name)),
                 "mirror_offline": s.mirror_offline}
                for s in checkpoints.CHECKPOINTS.values()]

    @app.post("/api/checkpoints/{name}/download")
    def download_checkpoint(name: str):
        if name not in checkpoints.CHECKPOINTS:
            raise HTTPException(404)
        if checkpoints.CHECKPOINTS[name].mirror_offline:
            raise HTTPException(410, "All known mirrors for this checkpoint are "
                                     "offline; install it manually into "
                                     f"{checkpoints.cache_dir() / name}")

        def cb(n, done, total):
            manager._publish({"type": "download", "name": n,
                              "done": done, "total": total})

        threading.Thread(target=checkpoints.download, args=(name, cb),
                         daemon=True).start()
        return {"ok": True}

    @app.get("/api/gallery")
    def gallery():
        return list_runs(manager.outputs_root)

    @app.get("/api/gallery/{run_id}/final.png")
    def final_png(run_id: str):
        p = manager.outputs_root / run_id / "final.png"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/api/gallery/{run_id}/settings.json")
    def sidecar(run_id: str):
        p = manager.outputs_root / run_id / "settings.json"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/api/gallery/{run_id}/timelapse.mp4")
    def timelapse(run_id: str, fps: int = 30):
        d = manager.outputs_root / run_id
        if not d.exists():
            raise HTTPException(404)
        out = d / "timelapse.mp4"
        if not out.exists():
            w = RunWriter.__new__(RunWriter)
            w.dir = d
            w.make_timelapse(fps=fps)
        return FileResponse(out, media_type="video/mp4")

    @app.get("/api/system")
    def system():
        import localvqgan
        from localvqgan.pipeline import backends
        ram_gb = psutil.virtual_memory().total / 2**30
        engines = ((["torch"] if backends.torch_available() else [])
                   + (["mlx"] if backends.mlx_available() else []))
        return {"device": manager.generator.device.type,
                "version": localvqgan.__version__,
                "total_ram_gb": round(ram_gb, 1),
                "max_recommended_side": _max_side(ram_gb),
                "engines": engines}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        # lifespan may not have run (e.g. bare TestClient); ensure publishes flow
        manager.attach_loop(asyncio.get_running_loop())
        snapshot = manager.status()
        if manager.latest_preview:
            snapshot["image_b64"] = base64.b64encode(manager.latest_preview).decode()
        await sock.send_json(snapshot)
        q = manager.subscribe()
        try:
            while True:
                msg = dict(await q.get())
                if "image_jpeg" in msg:
                    msg["image_b64"] = base64.b64encode(msg.pop("image_jpeg")).decode()
                await sock.send_json(msg)
        except WebSocketDisconnect:
            pass
        finally:
            manager.unsubscribe(q)

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def _preview_app() -> FastAPI:
    # uvicorn --factory entry for dev preview; mirrors main.run() wiring
    from localvqgan.server.main import default_manager
    return create_app(default_manager(Path.cwd() / "outputs"))

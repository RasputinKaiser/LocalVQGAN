import time

from fastapi.testclient import TestClient
from PIL import Image

from localvqgan.pipeline.generator import FrameUpdate
from localvqgan.server.app import create_app
from localvqgan.server.jobs import JobManager


class FakeGenerator:
    device = type("D", (), {"type": "cpu"})()

    def load(self, checkpoint, clip_model):
        pass

    def generate(self, settings, cancel=None):
        for i in range(1, 6):
            if cancel is not None and cancel.is_set():
                return
            time.sleep(0.01)
            yield FrameUpdate(i, 5, Image.new("RGB", (32, 32)), 0.5)


class FakeGeneratorNaN(FakeGenerator):
    def generate(self, settings, cancel=None):
        losses = [float("nan"), 0.5, float("nan"), 0.25, 0.1]
        for i, loss in enumerate(losses, start=1):
            if cancel is not None and cancel.is_set():
                return
            # Generous vs. shared/throttled CI runners: the polling test below
            # needs a window wide enough to reliably observe "running" even
            # under scheduling jitter, not just on a quiet local machine.
            time.sleep(0.1)
            yield FrameUpdate(i, 5, Image.new("RGB", (32, 32)), loss)


def make_client(tmp_path):
    mgr = JobManager(lambda: FakeGenerator(), tmp_path)
    # keep the mlx path faked too: with the real mlx generator installed,
    # engine=auto would otherwise construct it and load real checkpoints
    mgr._mlx_generator = FakeGenerator()
    return TestClient(create_app(mgr)), mgr


def _wait_idle(client, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get("/api/status").json()
        if st["state"] in ("done", "error", "idle"):
            return st
        time.sleep(0.05)
    raise TimeoutError


def test_job_lifecycle(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/api/jobs", json={"type": "still",
                                       "settings": {"prompts": "x", "iterations": 5}})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    st = _wait_idle(client)
    assert st["state"] == "done"
    assert client.get(f"/api/gallery/{run_id}/final.png").status_code == 200
    gallery = client.get("/api/gallery").json()
    assert gallery[0]["run_id"] == run_id


def test_animation_sidecar_records_engine(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/api/jobs", json={"type": "animation",
        "settings": {"prompts": "x", "engine": "torch", "width": 32, "height": 32},
        "keyframes": [{"prompts": "x", "frames": 2, "iterations_per_frame": 2}]})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    _wait_idle(client)
    s = client.get(f"/api/gallery/{run_id}/settings.json").json()
    assert s["engine_used"] == "torch"


def test_busy_returns_409(tmp_path):
    client, _ = make_client(tmp_path)
    client.post("/api/jobs", json={"type": "still", "settings": {"prompts": "x"}})
    r2 = client.post("/api/jobs", json={"type": "still", "settings": {"prompts": "y"}})
    assert r2.status_code == 409
    _wait_idle(client)


def test_cancel(tmp_path):
    client, _ = make_client(tmp_path)
    client.post("/api/jobs", json={"type": "still",
                                   "settings": {"prompts": "x", "iterations": 5}})
    assert client.post("/api/jobs/cancel").status_code == 200
    st = _wait_idle(client)
    assert st["state"] in ("done", "idle")


def test_nan_loss_does_not_crash_status(tmp_path):
    client, mgr = make_client(tmp_path)
    mgr._mlx_generator = FakeGeneratorNaN()
    r = client.post("/api/jobs", json={"type": "still",
                                       "settings": {"prompts": "x", "iterations": 5,
                                                    "engine": "mlx"}})
    assert r.status_code == 200
    saw_running = False
    for _ in range(100):
        status = client.get("/api/status")
        assert status.status_code == 200
        state = status.json()["state"]
        if state == "running":
            saw_running = True
        if state in ("done", "error"):
            break
        time.sleep(0.02)
    assert saw_running
    st = _wait_idle(client)
    assert st["state"] == "done"
    assert client.get("/api/status").status_code == 200


def test_system_info(tmp_path):
    client, _ = make_client(tmp_path)
    info = client.get("/api/system").json()
    assert "device" in info and "max_recommended_side" in info


def test_unknown_settings_are_ignored(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/api/jobs", json={"type": "still",
                                       "settings": {"prompts": "x", "bogus_field": 1}})
    assert r.status_code == 200
    _wait_idle(client)


def test_cut_pow_round_trips_to_sidecar(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/api/jobs", json={"type": "still",
        "settings": {"prompts": "x", "iterations": 5, "cut_pow": 1.7,
                     "engine": "torch"}})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    _wait_idle(client)
    s = client.get(f"/api/gallery/{run_id}/settings.json").json()
    assert s["cut_pow"] == 1.7


def test_cut_pow_defaults_to_one_in_sidecar(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/api/jobs", json={"type": "still",
        "settings": {"prompts": "x", "iterations": 5, "engine": "torch"}})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    _wait_idle(client)
    s = client.get(f"/api/gallery/{run_id}/settings.json").json()
    assert s["cut_pow"] == 1.0


def test_ws_snapshot_on_connect(tmp_path):
    client, _ = make_client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["state"] == "idle"


def test_ws_streams_frames_and_reattach(tmp_path):
    client, mgr = make_client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # snapshot
        client.post("/api/jobs", json={"type": "still",
                                       "settings": {"prompts": "x", "iterations": 5,
                                                    "display_freq": 1}})
        got_image = False
        for _ in range(20):
            msg = ws.receive_json()
            if msg.get("image_b64"):
                got_image = True
            if msg.get("state") == "done":
                break
        assert got_image
    # reattach after job: snapshot carries last preview
    with client.websocket_connect("/ws") as ws2:
        snap = ws2.receive_json()
        assert snap["state"] == "done" and snap.get("image_b64")


def _data_url():
    import base64 as b64
    from io import BytesIO
    buf = BytesIO()
    Image.new("RGB", (16, 16), (0, 255, 0)).save(buf, format="PNG")
    return "data:image/png;base64," + b64.b64encode(buf.getvalue()).decode()


def test_init_image_upload_decoded(tmp_path):
    client, mgr = make_client(tmp_path)
    r = client.post("/api/jobs", json={"type": "still",
        "settings": {"prompts": "x", "iterations": 5},
        "init_image_data": _data_url()})
    assert r.status_code == 200
    _wait_idle(client)
    uploads = list((tmp_path / "_uploads").glob("*.png"))
    assert len(uploads) == 1


def test_system_reports_engines(tmp_path):
    client, _ = make_client(tmp_path)
    info = client.get("/api/system").json()
    assert "engines" in info and "torch" in info["engines"]


def test_engine_recorded_in_sidecar(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/api/jobs", json={"type": "still",
        "settings": {"prompts": "x", "iterations": 5, "engine": "torch"}})
    run_id = r.json()["run_id"]
    _wait_idle(client)
    s = client.get(f"/api/gallery/{run_id}/settings.json").json()
    assert s["engine_used"] == "torch"

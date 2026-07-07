from PIL import Image

from localvqgan.pipeline.outputs import RunWriter, list_runs
from localvqgan.pipeline.settings import GenerationSettings


def _img():
    return Image.new("RGB", (64, 64), (200, 30, 30))


def test_run_lifecycle(tmp_path):
    w = RunWriter(tmp_path, GenerationSettings(prompts="x"))
    for i in range(1, 4):
        w.save_frame(i, _img())
    w.save_final(_img())
    w.write_sidecar({"seed_used": 42})
    mp4 = w.make_timelapse(fps=10)
    assert mp4.exists() and mp4.stat().st_size > 0
    runs = list_runs(tmp_path)
    assert runs[0]["run_id"] == w.run_id
    assert runs[0]["settings"]["prompts"] == "x"


def test_run_ids_increment(tmp_path):
    a = RunWriter(tmp_path, GenerationSettings())
    b = RunWriter(tmp_path, GenerationSettings())
    assert a.run_id != b.run_id

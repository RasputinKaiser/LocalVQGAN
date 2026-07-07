import numpy as np
import pytest

pytest.importorskip("mlx")


@pytest.mark.slow
def test_embeddings_match_open_clip():
    import mlx.core as mx
    import torch
    from localvqgan.pipeline.backends.mlx_backend.clip import MlxClip
    from localvqgan.pipeline.clip_guide import ClipGuide

    ref = ClipGuide("ViT-B-32", torch.device("cpu"))
    ours = MlxClip.load("ViT-B-32")

    t_ref = ref.embed_text("a lighthouse on a cliff at dusk").detach().numpy()[0]
    t_out = np.array(ours.embed_text("a lighthouse on a cliff at dusk"))[0]
    cos_t = float(np.dot(t_ref, t_out) / (np.linalg.norm(t_ref) * np.linalg.norm(t_out)))

    img = torch.rand(4, 3, 224, 224)
    i_ref = ref.encode_cutouts(img).detach().numpy()
    i_out = np.array(ours.encode_cutouts(mx.array(img.numpy().transpose(0, 2, 3, 1))))
    cos_i = min(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                for a, b in zip(i_ref, i_out))

    assert cos_t > 0.999 and cos_i > 0.999

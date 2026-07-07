import numpy as np
import pytest

pytest.importorskip("mlx")


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["ViT-B-32", "ViT-B-16"])
def test_embeddings_match_open_clip(model_name):
    import mlx.core as mx
    import torch
    from localvqgan.pipeline.backends.mlx_backend.clip import MlxClip
    from localvqgan.pipeline.clip_guide import ClipGuide

    ref = ClipGuide(model_name, torch.device("cpu"))
    ours = MlxClip.load(model_name)

    t_ref = ref.embed_text("a lighthouse on a cliff at dusk").detach().numpy()[0]
    t_out = np.array(ours.embed_text("a lighthouse on a cliff at dusk"))[0]
    cos_t = float(np.dot(t_ref, t_out) / (np.linalg.norm(t_ref) * np.linalg.norm(t_out)))

    img = torch.rand(4, 3, 224, 224)
    i_ref = ref.encode_cutouts(img).detach().numpy()
    i_out = np.array(ours.encode_cutouts(mx.array(img.numpy().transpose(0, 2, 3, 1))))
    cos_i = min(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                for a, b in zip(i_ref, i_out))

    assert cos_t > 0.999 and cos_i > 0.999, (
        f"{model_name} cos_t={cos_t:.6f} cos_i={cos_i:.6f}"
    )


def test_tokenizer_truncates_to_context_length():
    from localvqgan.pipeline.backends.mlx_backend.clip import CLIPTokenizer

    vocab = {"<|startoftext|>": 0, "a</w>": 1, "<|endoftext|>": 2}
    tokenizer = CLIPTokenizer({}, vocab, context_length=77)

    ids = tokenizer.tokenize("a " * 200)

    assert len(ids) <= 77
    assert int(ids[-1]) == tokenizer.eos_token


@pytest.mark.slow
def test_mlx_clip_long_prompt_truncates_before_embedding():
    from localvqgan.pipeline.backends.mlx_backend.clip import MlxClip

    clip = MlxClip.load("ViT-B-32")
    embedding = clip.embed_text("a " * 200)

    assert embedding.shape == (1, 512)

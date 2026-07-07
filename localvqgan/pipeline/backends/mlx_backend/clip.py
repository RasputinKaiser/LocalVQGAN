# Adapted from ml-explore/mlx-examples (clip/model.py, clip/tokenizer.py).
# MIT License, Copyright (c) 2023-2024 Apple Inc.

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import regex
from mlx.utils import tree_map

from localvqgan.pipeline.backends.mlx_backend.convert import cached_clip_weights


@dataclass
class CLIPTextConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    max_position_embeddings: int
    vocab_size: int
    layer_norm_eps: float


@dataclass
class CLIPVisionConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int
    layer_norm_eps: float


@dataclass
class CLIPConfig:
    text_config: CLIPTextConfig
    vision_config: CLIPVisionConfig
    projection_dim: int


def quick_gelu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(1.702 * x)


class Attention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: int | None = None,
        key_input_dims: int | None = None,
        value_input_dims: int | None = None,
        value_dims: int | None = None,
        value_output_dims: int | None = None,
        bias: bool = False,
    ):
        super().__init__()
        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        self.q_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.k_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.v_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, queries, keys, values, mask=None):
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        num_heads = self.num_heads
        b, l, _ = queries.shape
        _, s, _ = keys.shape
        queries = queries.reshape(b, l, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(b, s, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(b, s, num_heads, -1).transpose(0, 2, 1, 3)

        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(b, l, -1)
        return self.out_proj(values_hat)


class MLP(nn.Module):
    def __init__(self, config: CLIPTextConfig | CLIPVisionConfig):
        super().__init__()
        self.activation_fn = quick_gelu
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.activation_fn(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config: CLIPTextConfig | CLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(config.hidden_size, config.num_attention_heads, bias=True)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        y = self.layer_norm1(x)
        y = self.self_attn(y, y, y, mask)
        x = x + y
        y = self.layer_norm2(x)
        y = self.mlp(y)
        return x + y


class TextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        embeddings = self.token_embedding(x)
        embeddings += self.position_embedding.weight[: x.shape[1]]
        return embeddings


class Encoder(nn.Module):
    def __init__(self, config: CLIPTextConfig | CLIPVisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]


class ClipTextModel(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embeddings = TextEmbeddings(config)
        self.encoder = Encoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        b, n = x.shape
        eot_tokens = mx.argmax(x, axis=-1)
        x = self.embeddings(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(n, x.dtype)
        for layer in self.encoder.layers:
            x = layer(x, mask)
        x = self.final_layer_norm(x)
        return x[mx.arange(b), eot_tokens]


class VisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = mx.zeros((config.hidden_size,))
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size = x.shape[0]
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        embed_dim = patch_embeddings.shape[-1]
        cls_embeddings = mx.broadcast_to(
            self.class_embedding, (batch_size, 1, embed_dim)
        )
        embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
        embeddings += self.position_embedding.weight
        return embeddings


class ClipVisionModel(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.embeddings(x)
        x = self.pre_layrnorm(x)
        for layer in self.encoder.layers:
            x = layer(x)
        return self.post_layernorm(x[:, 0, :])


class CLIPModel(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.text_model = ClipTextModel(config.text_config)
        self.vision_model = ClipVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(
            config.vision_config.hidden_size, config.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            config.text_config.hidden_size, config.projection_dim, bias=False
        )

    def get_text_features(self, x: mx.array) -> mx.array:
        return self.text_projection(self.text_model(x))

    def get_image_features(self, x: mx.array) -> mx.array:
        return self.visual_projection(self.vision_model(x))


class CLIPTokenizer:
    def __init__(self, bpe_ranks, vocab, context_length: int | None = None):
        self.bpe_ranks = bpe_ranks
        self.vocab = vocab
        self.context_length = context_length
        self.pat = regex.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex.IGNORECASE,
        )
        self._cache = {self.bos: self.bos, self.eos: self.eos}

    @property
    def bos(self):
        return "<|startoftext|>"

    @property
    def bos_token(self):
        return self.vocab[self.bos]

    @property
    def eos(self):
        return "<|endoftext|>"

    @property
    def eos_token(self):
        return self.vocab[self.eos]

    def bpe(self, text):
        if text in self._cache:
            return self._cache[text]

        unigrams = list(text[:-1]) + [text[-1] + "</w>"]
        unique_bigrams = set(zip(unigrams, unigrams[1:]))
        if not unique_bigrams:
            return unigrams

        while unique_bigrams:
            bigram = min(
                unique_bigrams, key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )
            if bigram not in self.bpe_ranks:
                break

            new_unigrams = []
            skip = False
            for a, b in zip(unigrams, unigrams[1:]):
                if skip:
                    skip = False
                    continue
                if (a, b) == bigram:
                    new_unigrams.append(a + b)
                    skip = True
                else:
                    new_unigrams.append(a)

            if not skip:
                new_unigrams.append(b)

            unigrams = new_unigrams
            unique_bigrams = set(zip(unigrams, unigrams[1:]))

        self._cache[text] = unigrams
        return unigrams

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenize(*args, **kwargs)

    def tokenize(self, text, prepend_bos=True, append_eos=True) -> mx.array:
        if isinstance(text, list):
            return mx.array([self.tokenize(t, prepend_bos, append_eos) for t in text])

        clean_text = regex.sub(r"\s+", " ", text.lower())
        tokens = regex.findall(self.pat, clean_text)
        bpe_tokens = [token for t in tokens for token in self.bpe(t)]

        ids = []
        if prepend_bos:
            ids.append(self.bos_token)
        ids.extend(self.vocab[token] for token in bpe_tokens)
        if append_eos:
            ids.append(self.eos_token)
        if self.context_length is not None and len(ids) > self.context_length:
            ids = ids[: self.context_length]
            ids[-1] = self.eos_token
        return mx.array(ids)

    @staticmethod
    def from_pretrained(path: str | Path, context_length: int | None = None):
        path = Path(path)
        with open(path / "vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(path / "merges.txt", encoding="utf-8") as f:
            bpe_merges = f.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        bpe_ranks = dict(map(reversed, enumerate(bpe_merges)))
        return CLIPTokenizer(bpe_ranks, vocab, context_length=context_length)


def _config_from_pretrained(path: Path) -> CLIPConfig:
    with open(path / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    text_config = config["text_config"]
    vision_config = config["vision_config"]
    return CLIPConfig(
        text_config=CLIPTextConfig(
            num_hidden_layers=text_config["num_hidden_layers"],
            hidden_size=text_config["hidden_size"],
            intermediate_size=text_config["intermediate_size"],
            num_attention_heads=text_config["num_attention_heads"],
            max_position_embeddings=text_config["max_position_embeddings"],
            vocab_size=text_config["vocab_size"],
            layer_norm_eps=text_config["layer_norm_eps"],
        ),
        vision_config=CLIPVisionConfig(
            num_hidden_layers=vision_config["num_hidden_layers"],
            hidden_size=vision_config["hidden_size"],
            intermediate_size=vision_config["intermediate_size"],
            num_attention_heads=vision_config["num_attention_heads"],
            num_channels=vision_config.get("num_channels", 3),
            image_size=vision_config["image_size"],
            patch_size=vision_config["patch_size"],
            layer_norm_eps=vision_config["layer_norm_eps"],
        ),
        projection_dim=config["projection_dim"],
    )


class MlxClip:
    def __init__(
        self,
        model_name: str,
        model: CLIPModel,
        tokenizer: CLIPTokenizer,
        cut_size: int,
    ):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.cut_size = cut_size
        self._dtype = mx.float32

    @classmethod
    def load(cls, model_name: str) -> "MlxClip":
        if model_name not in {"ViT-B-32", "ViT-B-16"}:
            raise ValueError(f"Unknown CLIP model: {model_name}")
        weights_path = cached_clip_weights(model_name)
        model_dir = weights_path.parent
        config = _config_from_pretrained(model_dir)
        model = CLIPModel(config)
        model.load_weights(str(weights_path))
        model.eval().freeze()
        tokenizer = CLIPTokenizer.from_pretrained(
            model_dir, context_length=config.text_config.max_position_embeddings
        )
        return cls(model_name, model, tokenizer, config.vision_config.image_size)

    def embed_text(self, s: str) -> mx.array:
        tokens = self.tokenizer(s)[None, :]
        return self.model.get_text_features(tokens).astype(mx.float32)

    def encode_cutouts(self, batch: mx.array) -> mx.array:
        mean = mx.array((0.48145466, 0.4578275, 0.40821073), dtype=batch.dtype)
        std = mx.array((0.26862954, 0.26130258, 0.27577711), dtype=batch.dtype)
        batch = ((batch - mean) / std).astype(self._dtype)
        return self.model.get_image_features(batch).astype(mx.float32)

    def set_dtype(self, dtype) -> None:
        if self._dtype == dtype:
            return
        self.model.update(tree_map(lambda p: p.astype(dtype), self.model.parameters()))
        self._dtype = dtype

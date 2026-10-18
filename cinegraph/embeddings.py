"""Embedding backends.

`SentenceTransformerEmbedder` is the real one. `HashingEmbedder` is a zero-dependency
fallback (bag of stemmed words hashed into a fixed vector) so the project runs and the
tests pass without downloading a model.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_STOP = set(
    "a an the and or of to in on at for with by from is are was were be been it its "
    "his her their them they he she who whom that this these those as into than then "
    "about what which movie movies film films one two show me find any some".split()
)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for w in words:
        if w in _STOP or len(w) < 3:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


class HashingEmbedder:
    """Deterministic hashed bag-of-words + bigrams, L2-normalised."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, tok: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = _tokens(t)
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                j, sign = self._bucket(f)
                out[i, j] += sign
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )


def make_embedder(kind: str = "auto", model_name: str | None = None) -> Embedder:
    if kind in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            if kind == "sentence-transformers":
                raise
    return HashingEmbedder()

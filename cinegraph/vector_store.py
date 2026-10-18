"""Vector stores: Qdrant (real) and a numpy in-memory store (fallback/tests).

Both store the graph `uid` in the payload so hits map straight back onto graph nodes.
"""
from __future__ import annotations

import uuid
from typing import Any, Protocol

import numpy as np

from .models import VectorHit

_NS = uuid.UUID("6f1c2a52-6a3e-4c55-9d6f-0c10e0000000")


class VectorStore(Protocol):
    def recreate(self, dim: int) -> None: ...
    def upsert(self, uids: list[str], vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None: ...
    def search(self, vector: np.ndarray, k: int = 10) -> list[VectorHit]: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._uids: list[str] = []
        self._payloads: list[dict[str, Any]] = []
        self._mat: np.ndarray | None = None

    def recreate(self, dim: int) -> None:
        self._uids, self._payloads, self._mat = [], [], np.zeros((0, dim), dtype=np.float32)

    def upsert(self, uids, vectors, payloads) -> None:
        assert self._mat is not None, "call recreate() first"
        for uid, vec, payload in zip(uids, vectors, payloads):
            if uid in self._uids:
                i = self._uids.index(uid)
                self._mat[i] = vec
                self._payloads[i] = payload
            else:
                self._uids.append(uid)
                self._payloads.append(payload)
                self._mat = np.vstack([self._mat, vec[None, :]])

    def search(self, vector, k: int = 10) -> list[VectorHit]:
        if self._mat is None or len(self._uids) == 0:
            return []
        scores = self._mat @ vector
        order = np.argsort(-scores)[:k]
        return [VectorHit(self._uids[i], float(scores[i]), self._payloads[i]) for i in order]


class QdrantVectorStore:
    def __init__(self, url: str = "http://localhost:6333", collection: str = "cinegraph_movies",
                 api_key: str | None = None):
        from qdrant_client import QdrantClient

        self.collection = collection
        self.client = QdrantClient(location=":memory:") if url == ":memory:" else QdrantClient(url=url, api_key=api_key)

    @staticmethod
    def _point_id(uid: str) -> str:
        # Qdrant needs int/UUID ids; derive a stable UUID from the graph uid.
        return str(uuid.uuid5(_NS, uid))

    def recreate(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

    def upsert(self, uids, vectors, payloads) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=self._point_id(u), vector=v.tolist(), payload={**p, "uid": u})
            for u, v, p in zip(uids, vectors, payloads)
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, vector, k: int = 10) -> list[VectorHit]:
        res = self.client.query_points(self.collection, query=vector.tolist(), limit=k, with_payload=True)
        return [VectorHit(p.payload["uid"], float(p.score), dict(p.payload)) for p in res.points]

"""Wiring: build stores from env config, ingest data, expose a ready retriever."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .embeddings import Embedder, make_embedder
from .graph_store import GraphStore, Neo4jGraphStore, NetworkXGraphStore
from .models import Movie
from .retriever import HybridRetriever, RetrieverConfig
from .vector_store import InMemoryVectorStore, QdrantVectorStore, VectorStore

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "movies.json"


@dataclass
class Settings:
    backend: str = os.environ.get("CINEGRAPH_BACKEND", "memory")  # memory | server
    neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "cinegraph-pass")
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.environ.get("QDRANT_API_KEY")
    embedder: str = os.environ.get("CINEGRAPH_EMBEDDER", "auto")  # auto | hash | sentence-transformers


def build_stores(s: Settings) -> tuple[GraphStore, VectorStore]:
    if s.backend == "memory":
        return NetworkXGraphStore(), InMemoryVectorStore()
    if s.backend == "server":
        return (Neo4jGraphStore(s.neo4j_uri, s.neo4j_user, s.neo4j_password),
                QdrantVectorStore(s.qdrant_url, api_key=s.qdrant_api_key))
    raise ValueError(f"unknown backend {s.backend!r} (use 'memory' or 'server')")


def load_movies(path: Path | str = DEFAULT_DATA) -> list[Movie]:
    return [Movie.from_dict(d) for d in json.loads(Path(path).read_text())]


def ingest(movies: list[Movie], graph: GraphStore, vectors: VectorStore, embedder: Embedder) -> None:
    """Write each movie to BOTH stores under the same uid."""
    graph.clear()
    for m in movies:
        graph.upsert_movie(m)
    vecs = embedder.embed([m.embedding_text() for m in movies])
    vectors.recreate(embedder.dim)
    vectors.upsert([m.id for m in movies], vecs,
                   [{"title": m.title, "year": m.year} for m in movies])


def build_retriever(settings: Settings | None = None, data: Path | str = DEFAULT_DATA,
                    do_ingest: bool | None = None, config: RetrieverConfig | None = None
                    ) -> HybridRetriever:
    s = settings or Settings()
    graph, vectors = build_stores(s)
    embedder = make_embedder(s.embedder)
    # In-memory stores start empty every process, so always ingest for them.
    if do_ingest or (do_ingest is None and s.backend == "memory"):
        ingest(load_movies(data), graph, vectors, embedder)
    return HybridRetriever(graph, vectors, embedder, config)

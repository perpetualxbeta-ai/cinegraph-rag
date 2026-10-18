import pytest

from cinegraph.embeddings import HashingEmbedder
from cinegraph.generator import build_context
from cinegraph.graph_store import NetworkXGraphStore
from cinegraph.pipeline import ingest, load_movies
from cinegraph.retriever import HybridRetriever, RetrieverConfig
from cinegraph.vector_store import InMemoryVectorStore, QdrantVectorStore


def _vector_stores():
    yield pytest.param(InMemoryVectorStore, id="numpy")
    try:
        import qdrant_client  # noqa: F401
        yield pytest.param(lambda: QdrantVectorStore(":memory:"), id="qdrant-local")
    except ImportError:
        pass


@pytest.fixture(params=list(_vector_stores()))
def retriever(request):
    graph, vectors, emb = NetworkXGraphStore(), request.param(), HashingEmbedder()
    ingest(load_movies(), graph, vectors, emb)
    return HybridRetriever(graph, vectors, emb)


def titles(result, n):
    return [m.title for m in result.movies[:n]]


def test_shared_uid_links_both_stores(retriever):
    hit = retriever.vectors.search(retriever.embedder.embed(["lighthouse shipwreck letters"])[0], k=1)[0]
    assert hit.uid == "m01"
    assert retriever.graph.movie_facts([hit.uid])["m01"]["director"] == "Ines Moreau"


def test_entity_intersection_ranks_first(retriever):
    r = retriever.retrieve("sci-fi movies with Rina Sato", k=5)
    assert {e.name for e in r.linked} == {"Rina Sato", "Sci-Fi"}
    assert set(titles(r, 3)) == {"Orbit of Ash", "Neon Tide", "Starlight Diner"}


def test_director_and_actor_constraint(retriever):
    r = retriever.retrieve("films directed by Kapoor starring Marcus Obi", k=5)
    assert set(titles(r, 2)) == {"Orbit of Ash", "Quiet Frequency"}


def test_genre_plus_actor_beats_noisy_semantic_hits(retriever):
    r = retriever.retrieve("thriller with Joon Park", k=5)
    assert titles(r, 1) == ["Glass Harbor"]  # the only Thriller starring Joon Park


def test_more_like_a_named_movie_uses_graph_neighbours(retriever):
    r = retriever.retrieve("something like Glass Harbor", k=5)
    # Lantern Keeper shares director, actor and genre with Glass Harbor
    assert "The Lantern Keeper" in titles(r, 3)
    lk = next(m for m in r.movies if m.title == "The Lantern Keeper")
    assert any("Ines Moreau" in p.describe() for p in lk.paths)


def test_pure_semantic_query_without_entities(retriever):
    r = retriever.retrieve("ferry passengers who cannot remember how they died", k=3)
    assert r.linked == []
    assert "Ferryman" in titles(r, 3)


@pytest.mark.parametrize("fusion", ["auto", "rrf", "weighted"])
def test_fusion_modes_run(fusion):
    graph, vectors, emb = NetworkXGraphStore(), InMemoryVectorStore(), HashingEmbedder()
    ingest(load_movies(), graph, vectors, emb)
    r = HybridRetriever(graph, vectors, emb, RetrieverConfig(fusion=fusion)).retrieve(
        "thriller with Joon Park", k=4)
    assert r.movies and all(m.facts for m in r.movies)


def test_context_contains_graph_and_vector_info(retriever):
    ctx = build_context(retriever.retrieve("sci-fi movies with Rina Sato", k=2))
    assert "Graph link: Rina Sato -[ACTED_IN]-" in ctx
    assert "Plot:" in ctx and "semantic #" in ctx

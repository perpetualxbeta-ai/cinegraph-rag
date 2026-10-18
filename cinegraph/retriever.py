"""HybridRetriever: vector search + graph traversal, fused into one ranking.

Pipeline for a query
    1. Semantic leg   - embed the query, search the vector store  -> ranked movies
    2. Entity linking - names in the query ("Rina Sato", "sci-fi") -> graph seeds
    3. Graph leg      - expand from (a) linked entities and (b) the top semantic hits
                        through the graph; score each reachable movie by its paths
    4. Fusion         - Reciprocal Rank Fusion (default) or weighted score blend
    5. Enrich         - pull structured facts for the winners from the graph

Why both? Vectors catch *meaning* ("a story about grief and AI") but are blind to
hard structure ("directed by Kapoor AND starring Obi"). The graph nails structure but
knows nothing about plot. Each leg covers the other's blind spot, and the shared `uid`
on vector payloads + graph nodes makes the hand-off trivial.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .embeddings import Embedder
from .graph_store import GraphStore
from .linker import EntityLinker
from .models import Entity, GraphPath, RetrievedMovie
from .vector_store import VectorStore


@dataclass
class RetrieverConfig:
    vector_k: int = 20            # how many semantic candidates to pull
    seed_k: int = 3               # top semantic hits used as graph seeds ("more like these")
    max_hops: int = 2
    hop_decay: float = 0.5        # each extra hop multiplies the path score by this
    # Edge weights for expanding *from semantic seeds*. Genres are hubs, so a shared
    # genre is weaker evidence of relatedness than a shared director or actor.
    rel_weights: dict[str, float] = field(default_factory=lambda: {
        "DIRECTED_BY": 1.0, "ACTED_IN": 0.8, "IN_GENRE": 0.3})
    # "auto": weighted blend when the query names entities (graph score magnitudes
    # matter: matching 2 named entities should beat matching 1), RRF otherwise.
    fusion: str = "auto"          # "auto" | "rrf" | "weighted"
    rrf_k: int = 60
    vector_weight: float = 1.0
    graph_weight: float = 1.0
    graph_boost_when_linked: float = 1.5  # trust the graph more when the query names entities
    min_vector_score: float = 0.0


@dataclass
class RetrievalResult:
    query: str
    linked: list[Entity]
    movies: list[RetrievedMovie]


class HybridRetriever:
    def __init__(self, graph: GraphStore, vectors: VectorStore, embedder: Embedder,
                 config: RetrieverConfig | None = None):
        self.graph = graph
        self.vectors = vectors
        self.embedder = embedder
        self.cfg = config or RetrieverConfig()
        self._linker: EntityLinker | None = None

    @property
    def linker(self) -> EntityLinker:
        if self._linker is None:
            self._linker = EntityLinker(self.graph.entities())
        return self._linker

    def refresh_entities(self) -> None:
        self._linker = None

    # ------------------------------------------------------------------ legs
    def _semantic(self, query: str):
        qv = self.embedder.embed([query])[0]
        hits = self.vectors.search(qv, k=self.cfg.vector_k)
        return [h for h in hits if h.score > self.cfg.min_vector_score]

    def _graph(self, linked: list[Entity], semantic_seeds: list[tuple[str, float]]):
        cfg = self.cfg
        scores: dict[str, float] = defaultdict(float)
        paths_by_movie: dict[str, list[GraphPath]] = defaultdict(list)

        # (a) Explicitly named entities: every path counts at full rel weight.
        #     A movie connected to several named entities accumulates score, so
        #     "sci-fi with Rina Sato" ranks the intersection first.
        linked_uids = [e.uid for e in linked]
        label_of = {e.uid: e.label for e in linked}
        for e in linked:
            if e.label == "Movie":
                scores[e.uid] += 1.0  # naming a movie is direct evidence for it
        if linked_uids:
            # Named people/genres: 1 hop reaches their movies. Named movies: 2 hops
            # reach movies that share a person/genre with them.
            for p in self.graph.expand(linked_uids, cfg.max_hops):
                start_is_movie = label_of.get(p.start_uid) == "Movie"
                if not start_is_movie and len(p) > 1:
                    continue  # person->movie->person->movie drifts too far
                w = cfg.hop_decay ** (len(p) - 1)
                if start_is_movie:
                    w *= min(cfg.rel_weights.get(h.rel, 1.0) for h in p.hops)
                scores[p.end_uid] += w
                paths_by_movie[p.end_uid].append(p)

        # (b) Semantic seeds: "more like the best plot matches", weighted by similarity.
        seed_weight = dict(semantic_seeds)
        if seed_weight:
            for p in self.graph.expand(list(seed_weight), cfg.max_hops):
                if p.end_uid == p.start_uid:
                    continue
                w = seed_weight[p.start_uid] * cfg.hop_decay ** (len(p) - 1)
                w *= min(cfg.rel_weights.get(h.rel, 1.0) for h in p.hops)
                scores[p.end_uid] += 0.5 * w  # weaker than explicit entities
                paths_by_movie[p.end_uid].append(p)

        return dict(scores), paths_by_movie

    # ------------------------------------------------------------------ fusion
    def _fuse(self, vec_rank: dict[str, int], vec_score: dict[str, float],
              g_rank: dict[str, int], g_score: dict[str, float], has_links: bool):
        cfg = self.cfg
        gw = cfg.graph_weight * (cfg.graph_boost_when_linked if has_links else 1.0)
        out: dict[str, float] = {}
        candidates = set(vec_rank) | set(g_rank)
        mode = cfg.fusion
        if mode == "auto":
            mode = "weighted" if has_links else "rrf"
        if mode == "rrf":
            for u in candidates:
                s = 0.0
                if u in vec_rank:
                    s += cfg.vector_weight / (cfg.rrf_k + vec_rank[u])
                if u in g_rank:
                    s += gw / (cfg.rrf_k + g_rank[u])
                out[u] = s
        elif mode == "weighted":
            def norm(d):
                if not d:
                    return {}
                lo, hi = min(d.values()), max(d.values())
                return {k: (v - lo) / (hi - lo) if hi > lo else 1.0 for k, v in d.items()}
            vn, gn = norm(vec_score), norm(g_score)
            for u in candidates:
                out[u] = cfg.vector_weight * vn.get(u, 0.0) + gw * gn.get(u, 0.0)
        else:
            raise ValueError(f"unknown fusion {cfg.fusion!r}")
        return out

    # ------------------------------------------------------------------ public
    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        hits = self._semantic(query)
        linked = self.linker.link(query)

        semantic_seeds = [(h.uid, h.score) for h in hits[: self.cfg.seed_k]]
        g_scores, paths = self._graph(linked, semantic_seeds)

        vec_score = {h.uid: h.score for h in hits}
        vec_rank = {h.uid: i + 1 for i, h in enumerate(hits)}
        g_sorted = sorted(g_scores, key=lambda u: -g_scores[u])
        g_rank = {u: i + 1 for i, u in enumerate(g_sorted)}

        fused = self._fuse(vec_rank, vec_score, g_rank, g_scores, bool(linked))
        top = sorted(fused, key=lambda u: -fused[u])[:k]
        facts = self.graph.movie_facts(top)

        linked_uids = {e.uid for e in linked}
        movies = []
        for u in top:
            f = facts.get(u, {})
            # Prefer paths from entities the user named; fall back to "similar to" paths.
            uniq = {p.describe(): p for p in paths.get(u, [])}.values()
            explicit = [p for p in uniq if p.start_uid in linked_uids]
            ps = sorted(explicit or uniq, key=len)
            movies.append(RetrievedMovie(
                uid=u, title=f.get("title", u), score=fused[u],
                vector_score=vec_score.get(u), vector_rank=vec_rank.get(u),
                graph_score=g_scores.get(u, 0.0), graph_rank=g_rank.get(u),
                paths=list(ps)[:3], facts=f,
            ))
        return RetrievalResult(query, linked, movies)

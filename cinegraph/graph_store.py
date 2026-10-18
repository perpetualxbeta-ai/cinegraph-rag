"""Graph stores: Neo4j (real) and NetworkX (fallback/tests).

Schema
    (:Movie  {uid, name, title, year, plot})
    (:Person {uid, name})
    (:Genre  {uid, name})
    (Movie)-[:DIRECTED_BY]->(Person)
    (Person)-[:ACTED_IN]->(Movie)
    (Movie)-[:IN_GENRE]->(Genre)

Paths are treated as undirected for retrieval: "Rina Sato -[ACTED_IN]- Neon Tide".
"""
from __future__ import annotations

from typing import Any, Protocol

import networkx as nx

from .models import Entity, GraphPath, Hop, Movie, genre_uid, person_uid


class GraphStore(Protocol):
    def clear(self) -> None: ...
    def upsert_movie(self, m: Movie) -> None: ...
    def entities(self) -> list[Entity]: ...
    def expand(self, seeds: list[str], max_hops: int = 2) -> list[GraphPath]: ...
    def movie_facts(self, uids: list[str]) -> dict[str, dict[str, Any]]: ...


# --------------------------------------------------------------------------- NetworkX
class NetworkXGraphStore:
    def __init__(self) -> None:
        self.g = nx.MultiGraph()

    def clear(self) -> None:
        self.g.clear()

    def _node(self, uid: str, label: str, name: str, **props) -> None:
        self.g.add_node(uid, label=label, name=name, **props)

    def upsert_movie(self, m: Movie) -> None:
        if m.id in self.g:
            self.g.remove_node(m.id)
        self._node(m.id, "Movie", m.title, title=m.title, year=m.year, plot=m.plot)
        d = person_uid(m.director)
        self._node(d, "Person", m.director)
        self.g.add_edge(m.id, d, rel="DIRECTED_BY")
        for actor in m.cast:
            a = person_uid(actor)
            self._node(a, "Person", actor)
            self.g.add_edge(a, m.id, rel="ACTED_IN")
        for genre in m.genres:
            gu = genre_uid(genre)
            self._node(gu, "Genre", genre)
            self.g.add_edge(m.id, gu, rel="IN_GENRE")

    def entities(self) -> list[Entity]:
        return [Entity(u, d["label"], d["name"]) for u, d in self.g.nodes(data=True)]

    def expand(self, seeds: list[str], max_hops: int = 2) -> list[GraphPath]:
        """All simple paths from each seed that END at a Movie node, up to max_hops."""
        paths: list[GraphPath] = []
        for s in seeds:
            if s not in self.g:
                continue
            start_name = self.g.nodes[s]["name"]
            # DFS over simple paths, tracking the relation on each edge.
            stack: list[tuple[str, list[Hop], set[str]]] = [(s, [], {s})]
            while stack:
                node, hops, seen = stack.pop()
                if hops and self.g.nodes[node]["label"] == "Movie":
                    paths.append(GraphPath(s, start_name, list(hops)))
                if len(hops) >= max_hops:
                    continue
                for nbr, edges in self.g[node].items():
                    if nbr in seen:
                        continue
                    for e in edges.values():
                        hop = Hop(e["rel"], nbr, self.g.nodes[nbr]["name"])
                        stack.append((nbr, hops + [hop], seen | {nbr}))
        return paths

    def movie_facts(self, uids: list[str]) -> dict[str, dict[str, Any]]:
        out = {}
        for u in uids:
            if u not in self.g:
                continue
            n = self.g.nodes[u]
            facts = {"title": n["title"], "year": n["year"], "plot": n["plot"],
                     "director": None, "cast": [], "genres": []}
            for nbr, edges in self.g[u].items():
                for e in edges.values():
                    name = self.g.nodes[nbr]["name"]
                    if e["rel"] == "DIRECTED_BY":
                        facts["director"] = name
                    elif e["rel"] == "ACTED_IN":
                        facts["cast"].append(name)
                    elif e["rel"] == "IN_GENRE":
                        facts["genres"].append(name)
            facts["cast"].sort()
            facts["genres"].sort()
            out[u] = facts
        return out


# --------------------------------------------------------------------------- Neo4j
class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str, database: str | None = None):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._run("CREATE CONSTRAINT node_uid IF NOT EXISTS FOR (n:Node) REQUIRE n.uid IS UNIQUE")

    def close(self) -> None:
        self.driver.close()

    def _run(self, cypher: str, **params) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(cypher, params, database_=self.database)
        return [r.data() for r in records]

    def clear(self) -> None:
        self._run("MATCH (n:Node) DETACH DELETE n")

    def upsert_movie(self, m: Movie) -> None:
        # Every node also carries the :Node label so one uniqueness constraint covers all.
        self._run(
            """
            MERGE (mv:Node {uid: $id}) SET mv:Movie, mv.name = $title, mv.title = $title,
                                          mv.year = $year, mv.plot = $plot
            WITH mv
            OPTIONAL MATCH (mv)-[old]-() DELETE old
            WITH DISTINCT mv
            MERGE (d:Node {uid: $director_uid}) SET d:Person, d.name = $director
            MERGE (mv)-[:DIRECTED_BY]->(d)
            FOREACH (actor IN $cast |
              MERGE (a:Node {uid: actor.uid}) SET a:Person, a.name = actor.name
              MERGE (a)-[:ACTED_IN]->(mv))
            FOREACH (genre IN $genres |
              MERGE (g:Node {uid: genre.uid}) SET g:Genre, g.name = genre.name
              MERGE (mv)-[:IN_GENRE]->(g))
            """,
            id=m.id, title=m.title, year=m.year, plot=m.plot,
            director=m.director, director_uid=person_uid(m.director),
            cast=[{"uid": person_uid(a), "name": a} for a in m.cast],
            genres=[{"uid": genre_uid(g), "name": g} for g in m.genres],
        )

    def entities(self) -> list[Entity]:
        rows = self._run(
            "MATCH (n:Node) RETURN n.uid AS uid, n.name AS name, "
            "[l IN labels(n) WHERE l <> 'Node'][0] AS label"
        )
        return [Entity(r["uid"], r["label"], r["name"]) for r in rows]

    def expand(self, seeds: list[str], max_hops: int = 2) -> list[GraphPath]:
        hops = max(1, int(max_hops))  # Cypher can't parameterise path length; int() keeps it safe
        rows = self._run(
            # Keep only simple paths (no node visited twice), matching NetworkX behaviour.
            f"""
            MATCH p = (s:Node)-[*1..{hops}]-(m:Movie)
            WHERE s.uid IN $seeds
              AND ALL(i IN range(0, size(nodes(p)) - 2)
                      WHERE NOT nodes(p)[i] IN nodes(p)[i + 1..])
            RETURN s.uid AS start_uid, s.name AS start_name,
                   [r IN relationships(p) | type(r)] AS rels,
                   [n IN tail(nodes(p)) | {{uid: n.uid, name: n.name}}] AS nodes
            """,
            seeds=seeds,
        )
        return [
            GraphPath(r["start_uid"], r["start_name"],
                      [Hop(rel, n["uid"], n["name"]) for rel, n in zip(r["rels"], r["nodes"])])
            for r in rows
        ]

    def movie_facts(self, uids: list[str]) -> dict[str, dict[str, Any]]:
        rows = self._run(
            """
            MATCH (m:Movie) WHERE m.uid IN $uids
            OPTIONAL MATCH (m)-[:DIRECTED_BY]->(d:Person)
            OPTIONAL MATCH (a:Person)-[:ACTED_IN]->(m)
            OPTIONAL MATCH (m)-[:IN_GENRE]->(g:Genre)
            RETURN m.uid AS uid, m.title AS title, m.year AS year, m.plot AS plot,
                   d.name AS director, collect(DISTINCT a.name) AS cast,
                   collect(DISTINCT g.name) AS genres
            """,
            uids=uids,
        )
        return {
            r["uid"]: {"title": r["title"], "year": r["year"], "plot": r["plot"],
                       "director": r["director"], "cast": sorted(r["cast"]),
                       "genres": sorted(r["genres"])}
            for r in rows
        }

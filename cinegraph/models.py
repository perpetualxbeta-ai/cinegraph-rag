"""Shared data types and the ID scheme that ties the graph and vector stores together.

The core idea of the hybrid design: every node in the graph has a `uid`, and every
vector point stores that same `uid` in its payload. That shared key is what lets a
vector hit become a graph seed, and a graph neighbour become a scored result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def person_uid(name: str) -> str:
    return f"person:{name}"


def genre_uid(name: str) -> str:
    return f"genre:{name}"


@dataclass
class Movie:
    id: str
    title: str
    year: int
    director: str
    cast: list[str]
    genres: list[str]
    plot: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Movie":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    def embedding_text(self) -> str:
        """The text we embed: title + genres + plot. Cast/director live in the graph."""
        return f"{self.title} ({self.year}). Genres: {', '.join(self.genres)}. {self.plot}"


@dataclass
class Entity:
    uid: str
    label: str  # "Movie" | "Person" | "Genre"
    name: str


@dataclass
class Hop:
    """One edge of a graph path, as read from the start node outward."""
    rel: str
    node_uid: str
    node_name: str


@dataclass
class GraphPath:
    start_uid: str
    start_name: str
    hops: list[Hop]

    @property
    def end_uid(self) -> str:
        return self.hops[-1].node_uid

    def __len__(self) -> int:
        return len(self.hops)

    def describe(self) -> str:
        parts = [self.start_name]
        for h in self.hops:
            parts.append(f"-[{h.rel}]-")
            parts.append(h.node_name)
        return " ".join(parts)


@dataclass
class VectorHit:
    uid: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedMovie:
    uid: str
    title: str
    score: float
    vector_score: float | None = None
    vector_rank: int | None = None
    graph_score: float = 0.0
    graph_rank: int | None = None
    paths: list[GraphPath] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def explain(self) -> str:
        bits = []
        if self.vector_rank is not None:
            bits.append(f"semantic #{self.vector_rank} ({self.vector_score:.3f})")
        if self.graph_rank is not None:
            bits.append(f"graph #{self.graph_rank} ({self.graph_score:.3f})")
        return ", ".join(bits) or "no signal"

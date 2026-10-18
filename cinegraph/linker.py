"""Entity linking: find graph nodes that the query mentions by name.

Deliberately simple (string matching against node names + a few genre synonyms).
Swap in an LLM- or NER-based linker if your entity set gets large or messy.
"""
from __future__ import annotations

import re

from .models import Entity

GENRE_SYNONYMS = {
    "Sci-Fi": ["sci-fi", "sci fi", "scifi", "science fiction", "space"],
    "Romance": ["romance", "romantic", "love story"],
    "Comedy": ["comedy", "comedies", "funny"],
    "Thriller": ["thriller", "thrillers", "suspense"],
    "Mystery": ["mystery", "mysteries", "whodunit"],
    "Fantasy": ["fantasy"],
    "Action": ["action"],
    "Drama": ["drama", "dramas"],
    "Adventure": ["adventure", "adventures"],
    "Historical": ["historical", "period piece"],
}


def _has(phrase: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])", text) is not None


class EntityLinker:
    def __init__(self, entities: list[Entity]):
        self.entities = entities

    def link(self, query: str) -> list[Entity]:
        q = query.lower()
        found: dict[str, Entity] = {}
        for e in self.entities:
            if e.label == "Person":
                surname = e.name.split()[-1]
                if _has(e.name, q) or (len(surname) > 3 and _has(surname, q)):
                    found[e.uid] = e
            elif e.label == "Genre":
                aliases = GENRE_SYNONYMS.get(e.name, []) + [e.name]
                if any(_has(a, q) for a in aliases):
                    found[e.uid] = e
            elif e.label == "Movie":
                if _has(e.name, q):
                    found[e.uid] = e
        return list(found.values())

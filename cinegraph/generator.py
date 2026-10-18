"""Turn retrieval results into an LLM prompt and (optionally) an answer."""
from __future__ import annotations

import os

from .retriever import RetrievalResult

SYSTEM = (
    "You answer questions about a small movie catalogue. Use ONLY the context provided. "
    "Each movie lists its structured facts (from a knowledge graph) and its plot (from a "
    "vector index), plus the graph paths that connected it to the question. Cite movies "
    "by title. If the context doesn't contain the answer, say so."
)


def build_context(result: RetrievalResult) -> str:
    lines = []
    if result.linked:
        lines.append("Entities recognised in the question: "
                     + ", ".join(f"{e.name} ({e.label})" for e in result.linked))
        lines.append("")
    for i, m in enumerate(result.movies, 1):
        f = m.facts
        lines.append(f"[{i}] {m.title} ({f.get('year')})")
        lines.append(f"    Director: {f.get('director')}")
        lines.append(f"    Cast: {', '.join(f.get('cast', []))}")
        lines.append(f"    Genres: {', '.join(f.get('genres', []))}")
        lines.append(f"    Plot: {f.get('plot')}")
        for p in m.paths:
            lines.append(f"    Graph link: {p.describe()}")
        lines.append(f"    Retrieved via: {m.explain()}")
        lines.append("")
    return "\n".join(lines).rstrip()


def answer(question: str, result: RetrievalResult, model: str | None = None) -> str | None:
    """Ask Claude to answer from the retrieved context. Returns None if no API key is set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model or os.environ.get("CINEGRAPH_LLM_MODEL", "claude-sonnet-5-5"),
        max_tokens=800,
        system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"Context:\n{build_context(result)}\n\nQuestion: {question}"}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")

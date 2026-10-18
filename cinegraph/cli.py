"""Command line:  python -m cinegraph {ingest,ask} ..."""
from __future__ import annotations

import argparse
import sys

from .generator import answer, build_context
from .pipeline import DEFAULT_DATA, Settings, build_retriever
from .retriever import RetrieverConfig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cinegraph", description="Graph + vector hybrid RAG over movies")
    ap.add_argument("--backend", choices=["memory", "server"], help="override CINEGRAPH_BACKEND")
    ap.add_argument("--embedder", choices=["auto", "hash", "sentence-transformers"])
    sub = ap.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="load movies into Neo4j + Qdrant")
    ing.add_argument("--data", default=str(DEFAULT_DATA))

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=5)
    ask.add_argument("--fusion", choices=["auto", "rrf", "weighted"], default="auto")
    ask.add_argument("--context-only", action="store_true", help="skip the LLM, print retrieval")

    args = ap.parse_args(argv)
    s = Settings()
    if args.backend:
        s.backend = args.backend
    if args.embedder:
        s.embedder = args.embedder

    if args.cmd == "ingest":
        if s.backend == "memory":
            print("memory backend ingests on every run; nothing to persist. Use --backend server.")
            return 0
        build_retriever(s, data=args.data, do_ingest=True)
        print("Ingested into Neo4j and Qdrant.")
        return 0

    retriever = build_retriever(s, config=RetrieverConfig(fusion=args.fusion))
    result = retriever.retrieve(args.question, k=args.k)
    print(build_context(result))
    if not args.context_only:
        reply = answer(args.question, result)
        print("\n" + "=" * 60)
        print(reply if reply is not None else "(set ANTHROPIC_API_KEY to get an LLM answer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

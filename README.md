# CineGraph RAG

Hybrid retrieval-augmented generation over a movie catalogue, combining a **graph database (Neo4j)** with a **vector database (Qdrant)**.

- The **vector store** understands *meaning*: "a story about a grieving programmer and an AI".
- The **graph** understands *structure*: "sci-fi films starring Rina Sato", "directed by Kapoor *and* starring Obi", "more like Glass Harbor".
- A **hybrid retriever** runs both, fuses the rankings, and hands an LLM context that includes plots, structured facts, and the graph paths that explain *why* each movie was retrieved.

## How the two stores are joined

Every movie is written to both stores under the same `uid` (e.g. `m08`). Graph nodes carry it as a property; Qdrant points carry it in their payload. That shared key is the whole trick: a vector hit can become a graph seed, and a graph neighbour can be scored and enriched without any lookup table.

```
                 query
                   │
       ┌───────────┴────────────┐
       ▼                        ▼
 embed + Qdrant search     entity linking
 (top semantic movies)     ("Rina Sato", "sci-fi", "Glass Harbor")
       │                        │
       │   top hits as seeds    │ named entities as seeds
       └──────────┬─────────────┘
                  ▼
        Neo4j path expansion (≤2 hops)
        score movies by the paths reaching them
                  │
                  ▼
     fusion: weighted blend (entities named) / RRF (pure semantic)
                  │
                  ▼
     enrich winners with graph facts → LLM context → answer
```

### Scoring

- **Named entities** (people, genres, titles found in the question): each path to a movie adds `hop_decay^(hops-1)`. A movie connected to *two* named entities scores ~2, one connected to one scores ~1, so intersections win.
- **Semantic seeds** (top-3 vector hits): expand "more like this" through the graph, weighted by similarity and by edge type (shared director 1.0 > actor 0.8 > genre 0.3, since genres are hubs).
- **Fusion** (`--fusion auto`, the default): when the query names entities, a min-max-normalised weighted blend keeps the graph's score gaps meaningful; for pure-meaning queries, Reciprocal Rank Fusion. Both are available explicitly via `--fusion rrf|weighted`. All knobs live in `RetrieverConfig`.

## Quick start (no services)

```bash
pip install -e ".[dev]"
python -m cinegraph ask "sci-fi movies with Rina Sato" --context-only
pytest
```

The `memory` backend uses NetworkX + numpy and re-ingests on each run.

## With Neo4j + Qdrant

```bash
docker compose up -d
pip install -e ".[embeddings]"           # real sentence-transformers embeddings
export CINEGRAPH_BACKEND=server
python -m cinegraph ingest
python -m cinegraph ask "films directed by Kapoor starring Marcus Obi"
```

Browse the graph at http://localhost:7474 (`neo4j` / `cinegraph-pass`), e.g.
`MATCH p=(:Person {name:'Rina Sato'})-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(:Genre) RETURN p`.
Qdrant's dashboard is at http://localhost:6333/dashboard.

Set `ANTHROPIC_API_KEY` to get a generated answer after the retrieved context.

## Example

```
$ python -m cinegraph ask "something like Glass Harbor" -k 3 --context-only
Entities recognised in the question: Glass Harbor (Movie)

[2] The Lantern Keeper (2014)
    Graph link: Glass Harbor -[IN_GENRE]- Mystery -[IN_GENRE]- The Lantern Keeper
    Graph link: Glass Harbor -[ACTED_IN]- Rina Sato -[ACTED_IN]- The Lantern Keeper
    Graph link: Glass Harbor -[DIRECTED_BY]- Ines Moreau -[DIRECTED_BY]- The Lantern Keeper
    Retrieved via: graph #1 (1.168)
```

## Layout

| Path | What it does |
|---|---|
| `cinegraph/graph_store.py` | `Neo4jGraphStore` (Cypher) and `NetworkXGraphStore`, same interface |
| `cinegraph/vector_store.py` | `QdrantVectorStore` and `InMemoryVectorStore`, same interface |
| `cinegraph/embeddings.py` | sentence-transformers, with a hashing fallback for offline use |
| `cinegraph/linker.py` | string-match entity linker (people, genres, titles) |
| `cinegraph/retriever.py` | `HybridRetriever`: both legs + fusion + explanations |
| `cinegraph/generator.py` | builds the LLM context and calls Claude |
| `cinegraph/pipeline.py` | config, ingestion into both stores, wiring |
| `data/movies.json` | 16 fictional movies to play with |

## Using your own data

Replace `data/movies.json` (same fields: `id, title, year, director, cast, genres, plot`) or pass `--data path.json` to `ingest`. For other domains, keep the pattern — a shared id on both stores, a linker for your entity types, and edge weights that reflect how informative each relation is.

## Known limits

- The hashing embedder is lexical only; install `sentence-transformers` for real semantic matching.
- The linker is plain string matching. For larger catalogues, swap in NER or an LLM-based linker.
- The Neo4j backend is written against the same tested interface but was not run against a live server in CI here; `docker compose up` and try it.

# Architecture

CineGraph RAG answers questions about a movie catalogue by combining a **graph store** (structure: who directed/acted in what, which genres) with a **vector store** (meaning: what the plot is about). Both stores hold every movie under the same `uid`, which is what lets results flow between them.

## System overview

```mermaid
flowchart TB
    user(["User"])
    data[("data/movies.json")]

    subgraph app["cinegraph (Python package)"]
        cli["cli.py<br/>ingest | ask"]
        pipeline["pipeline.py<br/>Settings · build_stores · ingest · build_retriever"]
        embed["embeddings.py<br/>SentenceTransformer / Hashing fallback"]
        linker["linker.py<br/>EntityLinker (string match + genre synonyms)"]
        retriever["retriever.py<br/>HybridRetriever + RetrieverConfig"]
        generator["generator.py<br/>build_context · answer"]
        models["models.py<br/>Movie · Entity · GraphPath · VectorHit · RetrievedMovie"]

        subgraph gs["graph_store.py — GraphStore protocol"]
            nx["NetworkXGraphStore<br/>(backend=memory)"]
            neo["Neo4jGraphStore<br/>(backend=server)"]
        end
        subgraph vs["vector_store.py — VectorStore protocol"]
            mem["InMemoryVectorStore<br/>numpy (backend=memory)"]
            qd["QdrantVectorStore<br/>(backend=server)"]
        end
    end

    subgraph ext["External services (docker-compose / API)"]
        neo4j[("Neo4j 5<br/>bolt :7687 · UI :7474")]
        qdrant[("Qdrant<br/>:6333 · collection cinegraph_movies")]
        claude["Anthropic API<br/>Claude (optional, needs ANTHROPIC_API_KEY)"]
    end

    user --> cli
    cli --> pipeline
    cli --> retriever
    cli --> generator
    data --> pipeline
    pipeline --> embed
    pipeline --> gs
    pipeline --> vs
    pipeline -->|constructs| retriever
    retriever --> embed
    retriever --> linker
    retriever --> gs
    retriever --> vs
    linker -. "entities() from graph" .-> gs
    neo --> neo4j
    qd --> qdrant
    generator --> claude
```

`CINEGRAPH_BACKEND` picks the store pair: `memory` (NetworkX + numpy, re-ingested on every run, no services) or `server` (Neo4j + Qdrant from `docker-compose.yml`). `CINEGRAPH_EMBEDDER` picks `sentence-transformers` (all-MiniLM-L6-v2) or the lexical hashing fallback; `auto` uses the former if it is installed.

## Ingestion

```mermaid
flowchart LR
    json[("movies.json<br/>id, title, year, director,<br/>cast, genres, plot")] --> load["load_movies()"]
    load --> movies["list[Movie]"]

    movies -->|"upsert_movie(m)<br/>nodes keyed by uid"| gstore[("Graph store<br/>Neo4j / NetworkX")]
    movies -->|"embedding_text()<br/>title + year + genres + plot"| emb["Embedder.embed()"]
    emb -->|"vectors + payload {uid, title, year}"| vec[("Vector store<br/>Qdrant / numpy")]

    gstore -.-|"same uid, e.g. m08"| vec
```

Cast and director are deliberately **not** embedded: they live in the graph, so the vector leg stays about plot and theme.

## Graph schema

```mermaid
flowchart LR
    P["(:Person {uid: 'person:&lt;name&gt;', name})"]
    M["(:Movie {uid: 'm08', name, title, year, plot})"]
    G["(:Genre {uid: 'genre:&lt;name&gt;', name})"]
    M -- DIRECTED_BY --> P
    P -- ACTED_IN --> M
    M -- IN_GENRE --> G
```

In Neo4j every node also has a `:Node` label, which carries a single uniqueness constraint on `uid`. Paths are traversed as undirected during retrieval.

## Query path (`cinegraph ask`)

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant CLI as cli.py
    participant R as HybridRetriever
    participant E as Embedder
    participant V as Vector store
    participant L as EntityLinker
    participant G as Graph store
    participant Gen as generator.py
    participant C as Claude

    U->>CLI: ask "sci-fi movies with Rina Sato"
    CLI->>R: retrieve(question, k)
    R->>E: embed([query])
    R->>V: search(qv, vector_k=20)
    V-->>R: VectorHits (uid, cosine)
    R->>L: link(query)
    L-->>R: named entities (Person / Genre / Movie)
    R->>G: expand(named entity uids, ≤2 hops)
    R->>G: expand(top seed_k=3 vector hits, ≤2 hops)
    G-->>R: GraphPaths ending at Movies
    Note over R: score paths: hop_decay^(hops-1) × edge weight<br/>(DIRECTED_BY 1.0 · ACTED_IN 0.8 · IN_GENRE 0.3)
    Note over R: fuse: weighted blend if entities named,<br/>else Reciprocal Rank Fusion
    R->>G: movie_facts(top k uids)
    G-->>R: director, cast, genres, plot
    R-->>CLI: RetrievalResult (movies + paths + explanations)
    CLI->>Gen: build_context(result)
    opt not --context-only and ANTHROPIC_API_KEY set
        CLI->>Gen: answer(question, result)
        Gen->>C: messages.create(system, context + question)
        C-->>Gen: answer text
    end
    CLI-->>U: context (+ answer)
```

All tunables (`vector_k`, `seed_k`, `max_hops`, `hop_decay`, `rel_weights`, `fusion`, `rrf_k`, weights) live in `RetrieverConfig` in `cinegraph/retriever.py`.

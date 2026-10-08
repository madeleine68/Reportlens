

Claude · MD
CLAUDE.md — ReportLens
Working name. Rename freely.

What this project is
ReportLens is a question-answering assistant over Canadian public companies' annual reports. It is a portfolio project for an AI engineer job search. It demonstrates 10 RAG techniques, and an evaluation suite measures what each technique adds.

Clean, explainable code matters as much as features. The owner must be able to explain every part of this codebase in an interview.

The 10 techniques (and where they live)
#	Technique	Module	Phase
4	Contextual chunking	ingest/chunking.py	2
1	Hybrid search (keyword + vector, RRF)	retrieval/hybrid.py	2
3	Reranking (cross-encoder)	retrieval/rerank.py	2
2	Metadata filtering	retrieval/filters.py	2
9	Multimodal parsing (tables, charts)	ingest/parse.py, ingest/figures.py	3
5	SQL + vector	sources/sql_tool.py	3
6	Knowledge graph + vector	sources/graph_tool.py	3
8	Self-RAG (retrieve? supported?)	agents/self_rag.py	4
7	Corrective RAG (grade, rewrite, fallback)	agents/corrective.py	4
10	Agentic RAG (router across sources)	agents/graph.py	4
Corpus
5 companies × 4 years = 20 annual report PDFs (e.g. RBC, TD, Scotiabank, Shopify, BCE; fiscal 2021–2024)
Raw PDFs go in data/raw/{company}/{year}.pdf and are git-ignored
Parsed output goes in data/parsed/ (also git-ignored)
Tech stack
Language: Python 3.12, managed with uv
API: FastAPI (streaming responses)
Database: Postgres 16 + pgvector, run locally via Docker Compose
Keyword search: Postgres full-text search (BM25-like). Note in the README that it is not true BM25.
Parsing: Docling
Embeddings: sentence-transformers (start with BAAI/bge-small-en-v1.5)
Reranker: cross-encoder (start with cross-encoder/ms-marco-MiniLM-L-6-v2)
LLM: accessed only through core/llm.py. Model names come from .env, never hardcoded.
Validation: Pydantic v2
Orchestration (Phase 4): LangGraph
Graph DB (Phase 3): Neo4j
Web fallback (Phase 4): Tavily
Quality: ruff (lint and format), pytest, type hints everywhere
Repo structure
reportlens/
├── CLAUDE.md
├── DECISIONS.md          # decision log, updated as we go
├── README.md
├── docker-compose.yml    # postgres+pgvector (neo4j added in Phase 3)
├── .env.example
├── pyproject.toml
├── src/reportlens/
│   ├── core/             # config.py (pydantic-settings), llm.py, db.py, logging.py
│   ├── ingest/           # parse.py, chunking.py, embed.py, load.py, figures.py
│   ├── retrieval/        # vector.py, keyword.py, hybrid.py, rerank.py, filters.py
│   ├── sources/          # sql_tool.py, graph_tool.py, web_tool.py
│   ├── agents/           # graph.py, router.py, self_rag.py, corrective.py
│   ├── generation/       # answer.py (answers with page citations)
│   └── api/              # main.py (FastAPI)
├── evals/
│   ├── questions.jsonl   # OWNED BY THE HUMAN, see rules below
│   ├── run_evals.py
│   ├── metrics.py
│   └── results.md        # one row per experiment, append-only
├── scripts/              # one-off CLI scripts (ingest_all.py, etc.)
└── tests/
Commands
bash
uv sync                                  # install deps
docker compose up -d                     # start postgres (+ neo4j later)
uv run python scripts/ingest_all.py      # parse, chunk, embed, load
uv run python evals/run_evals.py --name "baseline"   # run evals, append to results.md
uv run uvicorn reportlens.api.main:app --reload       # start API
uv run pytest                            # tests
uv run ruff check . && uv run ruff format .
Keep this section up to date when commands change.

Phase plan
Current phase: 1 (update this line as phases complete)

Phase 0: Setup
[x] uv project, pyproject.toml, ruff, pytest configured
[x] docker-compose.yml with Postgres + pgvector (running; host port from POSTGRES_PORT, 5433 locally)
[x] core/config.py loading .env; .env.example committed
[x] core/llm.py thin wrapper (one function for chat, one for structured output)
[x] DECISIONS.md and evals/results.md created with headers
Phase 1: Baseline RAG + evals
 Parse PDFs with Docling, keeping page numbers
 Naive chunking (~500 tokens, small overlap)
 Embed and store in pgvector with metadata (company, year, page, section)
 Retrieve top-5 by vector similarity, answer with page citations
 Eval runner: recall@5, MRR, answer correctness (LLM judge), faithfulness, latency, cost
 Record the baseline row in evals/results.md
Phase 2: Better retrieval
 Contextual chunking: section-aware splits plus a context prefix (company, year, section, one-line summary)
 Keyword retriever (Postgres FTS)
 Hybrid merge with RRF (human writes this, see below)
 Cross-encoder reranking: top-50 → top-5
 Metadata filters: LLM extracts {company, year, section} into a Pydantic model and applies them as SQL WHERE clauses
 Eval run after each technique, one row each in results.md
Phase 3: Structured data
 Tables preserved as structured data; charts summarized by a multimodal LLM
 financials table: (company, year, metric, value, unit, source_page)
 Text-to-SQL tool: read-only DB role, approved tables only, query timeout, row limit
 Neo4j graph: Company, Segment, Subsidiary, Executive nodes and their relationships
 Cypher query tool
 Add numeric and relationship questions to evals (human adds them)
Phase 4: Agentic layer
 LangGraph state graph; router node picks vector | sql | graph | web | none (human writes the router)
 Self-RAG: decide if retrieval is needed; after answering, check the answer is supported and retry once if not
 Corrective RAG: grade retrieved chunks; if weak, rewrite the query and retry, then fall back to Tavily
 Multi-step questions (e.g. comparing companies across SQL and text)
 Max-step limit and clear error handling on every node
Phase 5: Ship
 FastAPI endpoints with streaming; response includes route taken, sources and pages
 Simple UI showing answer, route, sources and retrieved chunks
 Deploy (API: Railway/Fly; DB: Neon/Supabase; graph: Neo4j AuraDB free)
 README: architecture diagram, results table, design decisions, demo video link
Rules for the coding agent
How to work
One task at a time. Do the specific task asked. Don't start the next checklist item unprompted.
For anything over ~100 lines or touching multiple modules, propose a short plan first and wait for approval.
Explain non-obvious code in a brief summary after each task: what you built, key choices, anything I should review closely.
Write tests for every non-trivial function (parsing, chunking, RRF, filters, SQL safety).
Update this file's checklist and the "Current phase" line when a task is done.
Propose a DECISIONS.md entry whenever a meaningful choice is made (library, parameter, approach), in this format: ## YYYY-MM-DD — Title / Context / Options considered / Decision / Eval evidence (if any)
Human-owned pieces (do not implement unless explicitly asked)
These are the parts I will write myself so I can explain them in interviews. You may review them, suggest fixes or write tests for them.

evals/questions.jsonl: never add, edit or delete questions
The RRF merge function in retrieval/hybrid.py
The router logic in agents/router.py
Code conventions
Type hints on all functions; Pydantic models for structured data crossing module boundaries
Small, single-purpose functions; no file over ~300 lines
No hardcoded model names, API keys or file paths: use core/config.py
All LLM calls go through core/llm.py (so cost and latency can be logged in one place)
Log every retrieval and LLM call with timing and token counts
Prefer plain Python over frameworks until a phase requires one (no LangChain in Phases 1–3)
Docstrings on public functions, one or two lines
Cost and safety
Cache embeddings and parsed output; never re-parse or re-embed unchanged files
Use the cheaper model (LLM_MODEL_FAST) for grading, routing and query rewriting; the stronger model (LLM_MODEL_MAIN) only for final answers
The SQL tool must use a read-only Postgres role. Never execute LLM-generated SQL with write permissions.
Never commit .env, PDFs or API keys
Eval set format
evals/questions.jsonl, one JSON object per line:

json
{"id": "q001", "question": "What was RBC's net income in fiscal 2023?", "type": "numeric", "expected_answer": "...", "sources": [{"company": "RBC", "year": 2023, "page": 12}], "notes": ""}
type is one of: lookup, numeric, comparison, trend, relationship, out_of_corpus.

evals/results.md is append-only. One row per experiment:

Date	Experiment	Recall@5	MRR	Correctness	Faithfulness	p50 latency	Cost/query	Notes

Claude finished the response

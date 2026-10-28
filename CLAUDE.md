# CLAUDE.md — ReportLens

> Working name. Rename freely.

## What this project is

ReportLens is a question-answering assistant over Canadian public companies' annual reports. It is a portfolio project for an AI engineer job search. It demonstrates 10 RAG techniques, and an evaluation suite measures what each technique adds.

Clean, explainable code matters as much as features. The owner must be able to explain every part of this codebase in an interview.

## The 10 techniques (and where they live)

| #  | Technique                                 | Module                                 | Phase |
|----|-------------------------------------------|----------------------------------------|-------|
| 4  | Contextual chunking                       | `ingest/chunking.py`                   | 2     |
| 1  | Hybrid search (keyword + vector, RRF)     | `retrieval/hybrid.py`                  | 2     |
| 3  | Reranking (cross-encoder)                 | `retrieval/rerank.py`                  | 2     |
| 2  | Metadata filtering                        | `retrieval/filters.py`                 | 2     |
| 9  | Multimodal parsing (tables, charts)       | `ingest/parse.py`, `ingest/figures.py` | 3     |
| 5  | SQL + vector                              | `sources/sql_tool.py`                  | 3     |
| 6  | Knowledge graph + vector                  | `sources/graph_tool.py`                | 3     |
| 8  | Self-RAG (retrieve? supported?)           | `agents/self_rag.py`                   | 4     |
| 7  | Corrective RAG (grade, rewrite, fallback) | `agents/corrective.py`                 | 4     |
| 10 | Agentic RAG (router across sources)       | `agents/graph.py`                      | 4     |

## Corpus

- 3 companies × 4 years = 12 annual report PDFs: RBC, Scotiabank and BCE, fiscal 2021–2024 (TD and Shopify PDFs are downloaded in `data/raw/` but not ingested)
- Raw PDFs go in `data/raw/{company}/{year}.pdf` and are git-ignored
- Parsed output goes in `data/parsed/` (also git-ignored)

## Tech stack

| Area                   | Choice                                                                                  |
|------------------------|-----------------------------------------------------------------------------------------|
| Language               | Python 3.12, managed with uv                                                            |
| API                    | FastAPI (streaming responses)                                                           |
| Database               | Postgres 16 + pgvector, run locally via Docker Compose                                  |
| Keyword search         | Postgres full-text search (BM25-like). Note in the README that it is not true BM25.     |
| Parsing                | Docling                                                                                 |
| Embeddings             | sentence-transformers (start with `BAAI/bge-small-en-v1.5`)                             |
| Reranker               | cross-encoder (start with `cross-encoder/ms-marco-MiniLM-L-6-v2`)                       |
| LLM                    | Accessed only through `core/llm.py`. Model names come from `.env`, never hardcoded.     |
| Validation             | Pydantic v2                                                                             |
| Orchestration (Phase 4)| LangGraph                                                                               |
| Graph DB (Phase 3)     | Neo4j                                                                                   |
| Web fallback (Phase 4) | Tavily                                                                                  |
| Quality                | ruff (lint and format), pytest, type hints everywhere                                   |

## Repo structure

```
reportlens/
├── CLAUDE.md
├── DECISIONS.md          # decision log, updated as we go
├── README.md
├── docker-compose.yml    # postgres+pgvector, neo4j (Phase 3)
├── deploy/               # Dockerfile, env.deploy.example (Phase 5: Cloud Run)
├── .env.example
├── pyproject.toml
├── src/reportlens/
│   ├── core/             # config.py (pydantic-settings), llm.py, db.py, graph_db.py, logging.py
│   ├── ingest/           # parse.py, chunking.py, embed.py, load.py, tables.py, metrics.py, financials.py, graph.py, graph_names.py, graph_load.py, figures.py
│   ├── retrieval/        # vector.py, keyword.py, hybrid.py, rerank.py, filters.py
│   ├── sources/          # sql_tool.py, sql_safety.py, graph_tool.py, cypher_safety.py, web_tool.py (Tavily, off by default)
│   ├── agents/           # router.py, graph.py (routed answers as a LangGraph graph), routed.py (answer function per source), combined.py (multi-source), corrective.py, self_rag.py (LangGraph), events.py (progress for the API)
│   ├── generation/       # answer.py (answers with page citations), prompts.py
│   ├── evals/            # questions.py, metrics.py, judge.py, runner.py, routing.py (eval logic)
│   └── api/              # main.py (FastAPI routes), service.py (answering, resources), schemas.py, access.py (demo password, rate limits), static/index.html (UI)
├── evals/
│   ├── questions.jsonl   # OWNED BY THE HUMAN, see rules below
│   ├── questions.smoke.jsonl  # smoke test for the runner only; never recorded in results.md
│   ├── run_evals.py      # CLI; logic lives in src/reportlens/evals/
│   ├── runs/             # per-question details of each run (answers, scores, judge reasons)
│   └── results.md        # one row per experiment, append-only
├── scripts/              # one-off CLI scripts (ingest_all.py, etc.)
└── tests/
```

## Commands

```bash
uv sync                                              # install deps
docker compose up -d                                 # start postgres and neo4j
uv run python scripts/ingest_all.py                  # parse, chunk, embed, load
PARSE_TABLES=true CHUNKING=tables POSTGRES_DB=reportlens_tables uv run python scripts/ingest_all.py  # Phase 3: tables kept as rows/columns (slow)
uv run python scripts/load_tables.py                 # Phase 3: table numbers -> table_cells -> financials (needs table parses)
uv run python scripts/check_financials.py            # financials vs the eval set's expected numbers
uv run python scripts/load_graph.py --show           # Phase 3: build the Neo4j graph (fast LLM, cached) and print it
uv run python scripts/describe_figures.py            # Phase 3: describe charts with the vision model (cached, resumable; ~70/day on the free tier)
RETRIEVAL_USE_FILTERS=true RETRIEVAL_MODE=hybrid RETRIEVAL_RERANK=true RETRIEVAL_FIGURE_SLOTS=1 uv run python evals/run_evals.py --name "x" --retrieval-only   # Phase 3: best stack + one chart slot
uv run python scripts/ask.py "What was RBC's net income in 2023?"   # ask one question
uv run python scripts/ask.py --sql "What was RBC's net income in 2023?"   # Phase 3: answer via the SQL tool
uv run python scripts/ask.py --graph "Who was Scotiabank's CFO in 2024?"  # Phase 3: answer via the graph tool
uv run python evals/run_evals.py --name "baseline"   # run evals, append to results.md
uv run python evals/run_evals.py --name smoke --questions evals/questions.smoke.jsonl  # runner smoke test
uv run python evals/run_evals.py --name "x" --retrieval-only    # recall/MRR only, no answer model (cheap)
uv run python evals/run_evals.py --name "sql" --source sql --types numeric,trend,comparison  # Phase 3 SQL tool
uv run python evals/run_evals.py --name "graph" --source graph --types relationship  # Phase 3 graph tool
uv run python evals/route_accuracy.py                # Phase 4: router picks vs the best source per question (fast model only)
uv run python evals/run_evals.py --name "routed" --source routed   # Phase 4: answer via the router (needs a day's quota)
uv run python scripts/ask.py --routed "Who replaced Nadine Ahn as RBC's CFO?"   # Phase 4: ask via the router
WEB_SEARCH=true uv run python scripts/ask.py --routed "What was Apple's revenue in 2023?"   # Phase 4: web route (needs TAVILY_API_KEY)
uv run python scripts/ask.py --corrective "How did RBC's and Scotiabank's dividends compare in 2024?"   # Phase 4: corrective RAG, prints its steps
uv run python evals/run_evals.py --name "corrective" --source corrective   # Phase 4: corrective RAG eval (needs a day's quota)
uv run python evals/run_evals.py --name "corrective" --source corrective --resume   # continue a run stopped by the daily limit (exit 3)
SELF_RAG=true uv run python scripts/ask.py --routed "Why did BCE's headcount change in 2024?"   # Phase 4: routed answer + Self-RAG check
nohup scripts/eval_loop.sh >> data/cache/evals_run.log 2>&1 &   # Phase 4 evals back to back, resuming after each daily limit
uv run uvicorn reportlens.api.main:app --reload      # Phase 5: start the API + UI (UI at http://localhost:8000, docs at /docs; loads models at startup)
uv run python scripts/deploy_data.py                 # Phase 5: copy local Postgres → Neon and the graph → AuraDB (targets in .env.deploy)
uv run python scripts/deploy_cloudrun.py --setup     # Phase 5, once: enable APIs, image registry, service-account roles (needs gcloud auth login)
uv run python scripts/deploy_cloudrun.py --secrets   # Phase 5: store/refresh secrets in Secret Manager from .env.deploy
uv run python scripts/deploy_cloudrun.py             # Phase 5: build with Cloud Build and deploy to Cloud Run; prints the URL
uv run pytest                                        # fast tests
uv run pytest -m slow                                # tests that run real models
uv run ruff check . && uv run ruff format .          # lint and format
```

Keep this section up to date when commands change.

## Phase plan

**Current phase: 5** (Phases 0–4 complete; Phase 5 left: demo video; update this line as phases complete)

### Phase 0: Setup

- [x] uv project, `pyproject.toml`, ruff, pytest configured
- [x] `docker-compose.yml` with Postgres + pgvector (running; host port from `POSTGRES_PORT`, 5433 locally)
- [x] `core/config.py` loading `.env`; `.env.example` committed
- [x] `core/llm.py` thin wrapper (one function for chat, one for structured output)
- [x] `DECISIONS.md` and `evals/results.md` created with headers

### Phase 1: Baseline RAG + evals

- [x] Parse PDFs with Docling, keeping page numbers (light mode; see DECISIONS.md)
- [x] Naive chunking (450 tokens, 50 overlap; see DECISIONS.md)
- [x] Embed and store in pgvector with metadata (company, year, page, section)
- [x] Retrieve top-5 by vector similarity, answer with page citations
- [x] Eval runner: recall@5, MRR, answer correctness (LLM judge), faithfulness, latency, cost
- [x] Record the baseline row in `evals/results.md`

### Phase 2: Better retrieval

- [x] Contextual chunking: section-aware splits plus a context prefix (company, year, section, one-line summary) (label from metadata, no LLM summary; separate `reportlens_contextual` DB; see DECISIONS.md)
- [x] Keyword retriever (Postgres FTS)
- [x] Hybrid merge with RRF (written by Claude at the owner's explicit request, 2026-10-09)
- [x] Cross-encoder reranking: top-50 → top-5 (top-40: the hybrid candidate pool)
- [x] Metadata filters: LLM extracts `{company, year, section}` into a Pydantic model and applies them as SQL `WHERE` clauses (company and year done; section filter not built, see DECISIONS.md 2026-10-10)
- [x] Eval run after each technique, one row each in `results.md` (best stack: naive + filters + hybrid + rerank, correctness 0.74 on n=42; see DECISIONS.md)

### Phase 3: Structured data

- [x] Tables preserved as structured data; charts summarized by a multimodal LLM (tables done and measured, `reportlens_tables` DB, not the default; charts: `ingest/figures.py` built, all 384 candidates described by 2026-10-25 and 379 loaded as chart chunks (5 judged not charts); measured 2026-10-25 (retrieval only, n=53): recall@5 0.81 vs 0.80 without charts, MRR 0.61 vs 0.63, 2 of 4 chart-only questions found; stays off (`RETRIEVAL_FIGURES=false`). One reserved chart slot instead (`RETRIEVAL_FIGURE_SLOTS=1`, charts searched separately in `retrieval/search.py`): recall@5 0.85, MRR 0.65, all 4 chart-only questions found, p50 8.4s vs 4.7s; on answers (routed corrective, n=53) 0.87 vs 0.89, within noise, because the router sends most chart questions to SQL; off by default; see DECISIONS.md 2026-10-13 and 2026-10-16)
- [x] `financials` table: `(company, year, metric, value, unit, source_page)` (plus report_year, scope, basis; 12 metrics from `table_cells` by rules + vote; 29/33 eval figures found; see DECISIONS.md 2026-10-14)
- [x] Text-to-SQL tool: read-only DB role, approved tables only, query timeout, row limit (`sources/sql_tool.py` + `sql_safety.py` with sqlglot; correctness 0.80 (with the comparison rule) vs 0.67 for text on the 33 numeric/trend/comparison questions; see DECISIONS.md 2026-10-14 and 2026-10-15)
- [x] Neo4j graph: Company, Segment, Subsidiary, Executive nodes and their relationships (segments from `financials`, subsidiaries and executives by fast-LLM extraction, every relationship with report year and page; 51 segment, 247 subsidiary and 112 executive relationships across BCE, RBC and Scotiabank; see DECISIONS.md 2026-10-16)
- [x] Cypher query tool (`sources/graph_tool.py` + `cypher_safety.py`: text check for read-only clauses and known labels, then a read transaction with a timeout and row limit; correctness 0.56 vs 0.83 for text search on the 9 relationship questions; see DECISIONS.md 2026-10-16)
- [x] Add numeric and relationship questions to evals (q044–q054 added by Claude at the owner's request on 2026-10-16: 7 relationship, 4 chart-only; answers and pages verified in the PDFs)

### Phase 4: Agentic layer

- [x] LangGraph state graph; router node picks `vector | sql | graph | web | none` (router written by Claude at the owner's request on 2026-10-16: `agents/router.py` + `agents/routed.py` with fallback to vector; route accuracy 0.83 vs 0.71 for always-vector; LangGraph graph `agents/graph.py` built 2026-10-26 (route → ask_routed → fallback → optional Self-RAG check; same behaviour, router logic unchanged); full routed eval 2026-10-21: correctness 0.88 vs 0.74 for corrective text search alone, n=53)
- [x] Self-RAG: decide if retrieval is needed; after answering, check the answer is supported and retry once if not (built 2026-10-17 in `agents/self_rag.py` as a LangGraph graph after the router; "retrieval needed?" is the router's `none` route; off by default (`SELF_RAG=false`); measured 2026-10-22: correctness 0.78 vs 0.89 without it (n=53), because the checker flags arithmetic and SQL/graph facts as unsupported and the revision drops them or refuses; see DECISIONS.md 2026-10-17)
- [x] Corrective RAG: grade retrieved chunks; if weak, rewrite the query and retry, then fall back to Tavily (built 2026-10-17 as a LangGraph state graph in `agents/corrective.py`, tested with stubs and on 3 trial questions; the router's text-search route uses it unless `ROUTED_CORRECTIVE=false`; measured on n=53: corrective alone 0.74, router alone 0.88, router + corrective 0.89; see DECISIONS.md 2026-10-17)
- [x] Multi-step questions (e.g. comparing companies across SQL and text) (simple version built 2026-10-26 in `agents/combined.py`, prompt written by Claude at the owner's request: questions routed to SQL or the graph are answered from their rows plus the top text chunks in one call, `MULTI_SOURCE=true`, off by default; trial: q023 headcount now exact; measured 2026-10-28 on n=53: 0.87 vs 0.89 without it (noise; two headcount questions fixed, chart shares q051–q053 still miss), so it stays off; the eval also exposed a crash on computed rows without a page, fixed; a planner that splits questions into sub-questions is not built)
- [x] Max-step limit and clear error handling on every node (router: falls back to text search if its call fails; routed answers: a source that raises falls back to text search; corrective RAG: `AGENT_MAX_STEPS` as LangGraph's recursion limit, refusal past it, grade/rewrite failures fail open; Self-RAG: a failed revision keeps the earlier answer; rate-limit errors are never failed open)

### Phase 5: Ship

- [x] FastAPI endpoints with streaming; response includes route taken, sources and pages (`api/`: `/health`, `/meta`, `/ask`, `/ask/stream` (server-sent progress events: route, steps, answer, done/error; not token streaming); tested with stubs and started live (health and meta); no auth yet; see DECISIONS.md 2026-10-17)
- [x] Simple UI showing answer, route, sources and retrieved chunks (`api/static/index.html` at `/`: plain HTML/CSS/JS, streams steps from `/ask/stream`; checked with a stub answerer in light and dark; 2026-10-17: "What is this?" panel and an animated pipeline strip for the streamed steps; 2026-10-28: redesigned (top bar with corpus chips and password, one question box with a segmented source picker, example cards, answer beside a sidebar of cited sources and evidence; Inter font; checked at 1280 px, 375 px and in dark mode with a fake server); screenshot for the README after the evals free up the quota; see DECISIONS.md 2026-10-17)
- [x] Deploy (API: Google Cloud Run, chosen by the owner on 2026-10-17 after Hugging Face Docker Spaces turned out to need PRO; DB: Neon; graph: Neo4j AuraDB free) (data copied to Neon (6,075 chunks, reader role checked) and AuraDB (148 nodes, 410 relationships); code ready: `deploy/Dockerfile`, `scripts/deploy_data.py`, `scripts/deploy_cloudrun.py`, demo password + rate limits in `api/access.py` with the daily count in Postgres; live at https://reportlens-400767792007.us-east1.run.app since 2026-10-17 (health: Postgres and Neo4j reachable, password required; `/ask` without it → 401); redeployed 2026-10-28 (revision reportlens-00004); first live question answered correctly (RBC 2023 net income, routed to SQL, p. 23); AuraDB Free pauses when idle, so `/health` can show neo4j false until it is resumed in the Aura console, and graph questions then fall back to text search; see DECISIONS.md 2026-10-17)
- [ ] README: architecture diagram, results table, design decisions, demo video link (2026-10-26: architecture diagram, headline results (n=53), Phase 3 chart and Phase 4 results, design-decisions table done; demo video not recorded yet)

## Rules for the coding agent

### How to work

- **One task at a time.** Do the specific task asked. Don't start the next checklist item unprompted.
- For anything over ~100 lines or touching multiple modules, **propose a short plan first** and wait for approval.
- Explain non-obvious code in a brief summary after each task: what you built, key choices, anything I should review closely.
- Write tests for every non-trivial function (parsing, chunking, RRF, filters, SQL safety).
- Update this file's checklist and the "Current phase" line when a task is done.
- Propose a `DECISIONS.md` entry whenever a meaningful choice is made (library, parameter, approach), in this format:

  ```markdown
  ## YYYY-MM-DD — Title
  **Context:** ...
  **Options considered:** ...
  **Decision:** ...
  **Eval evidence (if any):** ...
  ```

- For every implementation phase in ReportLens, include a clear Mermaid diagram in README.md illustrating the workflow, key components, and data flow from input to output. Keep the diagrams beginner-friendly, explain each step briefly, and update them as new features are implemented so the README always reflects the actual project state.

### Human-owned pieces (do not implement unless explicitly asked)

These are the parts I will write myself so I can explain them in interviews. You may review them, suggest fixes or write tests for them.

- `evals/questions.jsonl`: never add, edit or delete questions
- The RRF merge function in `retrieval/hybrid.py` (owner asked Claude to write it on 2026-10-09; owner to review and be able to explain it)
- The router logic in `agents/router.py` (owner asked Claude to write it on 2026-10-16; owner to review and be able to explain it)

### Code conventions

- Type hints on all functions; Pydantic models for structured data crossing module boundaries
- Small, single-purpose functions; no file over ~300 lines
- No hardcoded model names, API keys or file paths: use `core/config.py`
- All LLM calls go through `core/llm.py` (so cost and latency can be logged in one place)
- Log every retrieval and LLM call with timing and token counts
- Prefer plain Python over frameworks until a phase requires one (no LangChain in Phases 1–3)
- Docstrings on public functions, one or two lines

### Cost and safety

- Cache embeddings and parsed output; never re-parse or re-embed unchanged files
- Use the cheaper model (`LLM_MODEL_FAST`) for grading, routing and query rewriting; the stronger model (`LLM_MODEL_MAIN`) only for final answers
- The SQL tool must use a read-only Postgres role. **Never execute LLM-generated SQL with write permissions.**
- LLM-generated Cypher runs only in a read transaction, after `sources/cypher_safety.py` has checked it.
- Never commit `.env`, `.env.deploy`, PDFs or API keys

## Eval set format

`evals/questions.jsonl`, one JSON object per line:

```json
{"id": "q001", "question": "What was RBC's net income in fiscal 2023?", "type": "numeric", "expected_answer": "$14,866 million", "sources": [{"company": "rbc", "year": 2023, "pages": [26, 122, 141]}], "notes": ""}
```

- Each entry in `sources` is one piece of evidence the answer needs. List **every** page where that evidence appears; retrieving any one of them counts. `"page": 12` is accepted as shorthand for `"pages": [12]`.
- Pages are **PDF page numbers** (1 = first page of the file), not the numbers printed on the page. They differ in some reports: in the Scotiabank and BCE reports, PDF page = printed page + 2.
- Lines starting with `//` are comments.

`type` is one of: `lookup`, `numeric`, `comparison`, `trend`, `relationship`, `out_of_corpus`.

`evals/results.md` is append-only. One row per experiment:

| Date | Experiment | Recall@5 | MRR | Correctness | Faithfulness | p50 latency | Cost/query | Notes |
|------|------------|----------|-----|-------------|--------------|-------------|------------|-------|

# Decision log

Format: `## YYYY-MM-DD — Title` / Context / Options considered / Decision / Eval evidence (if any)

---

## 2026-10-08 — Free LLM via an OpenAI-compatible client

**Context:** This is a portfolio project, so LLM calls should cost nothing. The dev machine (Intel Mac, 8 GB RAM) is too weak to run a useful model locally.

**Options considered:**
- Paid APIs (Anthropic, OpenAI): best quality, but they cost money.
- Local Ollama: free and private, but slow and low quality on 8 GB of RAM.
- Free hosted tiers (Groq, Google Gemini): free, fast, rate-limited.

**Decision:** Use the `openai` Python SDK against any OpenAI-compatible endpoint, configured with `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_MAIN` and `LLM_MODEL_FAST`. Default to the Groq free tier: `llama-3.3-70b-versatile` as the main model and `llama-3.1-8b-instant` as the fast one. Gemini or Ollama can be swapped in by editing `.env` alone, with no code change. Structured output uses JSON mode plus Pydantic validation, with one retry, because every one of these providers supports that.

**Eval evidence:** none yet.

## 2026-10-08 — Docling in light mode on CPU for Phase 1 parsing

**Context:** Phase 1 needs every annual report parsed with page numbers kept. The dev machine is an Intel Mac with 8 GB of RAM and no GPU. PyTorch stopped publishing Intel Mac wheels after 2.2.2, and newer `docling-parse` releases have no Intel Mac wheel either.

**Options considered:**
- Full Docling pipeline (OCR + TableFormer): best tables, but slowest, and the reports are digital PDFs that don't need OCR.
- Light Docling (layout model only, `do_ocr=False`, `do_table_structure=False`): keeps headings, page numbers and table regions; tables come out as plain lines, not rows and columns.
- Plain text extraction (pypdfium2): fastest, but no headings or table regions, and these PDFs have no bookmarks to recover sections from.

**Decision:** Light Docling, pinned to `torch==2.2.2`, `numpy<2` and `docling-parse<5` for Intel Mac. Without TableFormer, Docling emits each table both as one squashed cell and as loose lines inside the table's area, so extraction folds those lines into a single table block. The raw Docling output is cached separately from our blocks (`CONVERTER_VERSION` vs `EXTRACT_VERSION`), so extraction changes don't re-run Docling. Revisit in Phase 3, when tables need real structure: turn on TableFormer, bump `CONVERTER_VERSION`, re-run.

**Eval evidence:** none yet. Measured on RBC 2023 (237 pages): 7.4 min, 2.0 GB peak RAM, 3,279 blocks (25,972 before the table fix). Cache hit: 0.1 s.

## 2026-10-09 — Naive chunks of 450 tokens with 50 overlap

**Context:** The Phase 1 baseline needs fixed-size chunks. The plan said ~500 tokens, but the embedding model (`BAAI/bge-small-en-v1.5`) reads at most 512 tokens and silently truncates the rest, and Phase 2 will prepend a context prefix to every chunk.

**Options considered:**
- 500 tokens: matches the plan, but leaves almost no room for the Phase 2 prefix, so chunk tails would be cut off without any error.
- 450 tokens: leaves ~60 tokens for the prefix.
- Smaller (256): more precise retrieval, but double the chunks and less context per answer; better tested as an experiment than picked up front.

**Decision:** 450 tokens with 50 overlap, counted with the embedding model's own tokenizer, configurable via `CHUNK_TOKENS` / `CHUNK_OVERLAP`. Chunks are fixed windows over each report's joined text, ignoring headings and tables, so Phase 2 has an honest baseline to beat. Each chunk lists every page it touches, for citations.

**Eval evidence:** none yet. On 12 reports: 5,991 chunks, ~1 s per report, longest chunk re-tokenizes to 452 tokens (limit 512). About 46% of chunks span more than one page.

## 2026-10-09 — Postgres schema and embedding storage

**Context:** Chunks need to be stored with their vectors and the metadata Phase 2 will filter on, and re-running ingestion must not re-embed unchanged reports (~1–2 min each on CPU).

**Options considered:**
- Migration tool (Alembic) vs one idempotent `schema.sql` applied on connect.
- Per-chunk embedding cache (hash of chunk text) vs per-report fingerprint.
- No vector index (exact scan) vs HNSW.

**Decision:** One `schema.sql` (`CREATE ... IF NOT EXISTS`), simple to read and explain; revisit if the schema starts changing often. A `documents` row per report stores the PDF hash plus an `ingest_config` fingerprint (converter and extract versions, chunk settings, embedding model); if it matches, the report is skipped, otherwise its chunks are replaced in one transaction. HNSW cosine index, matching what a production setup would use, even though an exact scan over ~6k rows is also fast. Company, year and section are columns on `chunks` so Phase 2 filters are plain `WHERE` clauses. Queries get bge's instruction prefix via a separate `embed_query()`; passages don't.

**Eval evidence:** none yet. 12 reports → 5,991 chunks loaded in ~19 min; unchanged re-run takes 26 s (mostly imports).

## 2026-10-09 — Citations as [n] markers parsed in code

**Context:** Answers must cite the pages they come from, and a citation must never point at something that wasn't retrieved.

**Options considered:**
- Structured JSON output (`{"answer": ..., "sources": [...]}`): explicit, but harder to read, and JSON reliability varies across the Groq/Gemini/Ollama models we can switch between.
- Plain text with `[n]` markers after each claim, parsed in code: readable, works with any chat model, and checkable.

**Decision:** Numbered sources in the prompt, `[n]` markers in the answer, parsed by `parse_citations()`. Markers outside 1..k are dropped and logged, so citations can't be invented. Full-width `【n】` markers (a gpt-oss habit, seen in the first real test) are normalized to `[n]`. A fixed refusal sentence (`prompts.NOT_FOUND`) marks "not in the reports", so the eval can score out-of-corpus questions. A citation lists every page of its chunk, since chunks can span 2–3 pages.

**Eval evidence:** none yet. Manual check: 4 questions; 2 answered with correct citations (Bell subscribers verified against BCE 2024 p. 8), 1 correct refusal (out of corpus), 1 refusal caused by a retrieval miss (RBC 2023 net income).

## 2026-10-09 — Eval runner design

**Context:** Every technique from Phase 2 on must be measured against the baseline, so the runner needs metrics that are cheap, repeatable, and easy to explain.

**Options considered:**
- Score everything with an LLM judge vs compute retrieval metrics in code.
- Judge on the main model (stronger) vs the fast model (`CLAUDE.md`: grading goes on the cheaper model).
- Ragas or a similar eval library vs a small hand-written runner.

**Decision:** Hand-written, so every number can be explained. Recall@5 and MRR are computed in code: a retrieved chunk matches an expected source when company and year match and the expected page is among the chunk's pages. Correctness and faithfulness use an LLM judge on the fast model, with JSON output validated by Pydantic and the judge giving its reason before its verdict. Refusals are scored without a judge: correct for `out_of_corpus`, 0 otherwise. Each run's per-question details go to `evals/runs/`; only runs on the owner's `questions.jsonl` add a row to `results.md`. Eval logic lives in `src/reportlens/evals/` (importable and testable); `evals/run_evals.py` is a thin CLI.

**Caveats found in the smoke run:**
- Groq's free-tier rate limits make the client wait and retry, so p50 latency (1.5 s per answer when idle, 12.6 s in the run) mostly measures waiting. Compare latency only between runs on the same tier.
- Recall needs every page that contains the answer: s03's answer came from a highlights page that wasn't listed as a source, so it scored recall 0 with a correct answer.

**Eval evidence:** Smoke set (6 questions, not the eval set): recall@5 0.20, MRR 0.20, correctness 0.50, faithfulness 1.00. The run also caught a citation format (`[3†L13-L16]`) the parser missed, now fixed.

## 2026-10-09 — Metadata filters: company and year only, with fallback

**Context:** The baseline's biggest failure: 55% of retrieved chunks came from a different fiscal year than the question named, because reports from different years read alike to vector search.

**Options considered:**
- Regex/keyword extraction of company and year: no LLM call, but misses nicknames ("Bell"), ranges ("2022 to 2024") and phrasing like "the previous year".
- LLM extraction into a Pydantic model (the plan in `CLAUDE.md`).
- Filtering on section too: the plan includes it, but chunks carry 2,074 distinct raw headings ("Table 18", "BCE NET ACTIVATIONS (LOSSES)") the LLM has no way to know, so a section filter would mostly match nothing.

**Decision:** The fast LLM extracts `companies` and `years` as lists (comparisons and ranges need several values). The prompt lists the companies and years actually in the database plus common aliases, and code drops anything else. Filters become `WHERE company = ANY(...) AND year = ANY(...)`, with pgvector 0.8's `hnsw.iterative_scan = strict_order` so a strict filter still returns k rows from the HNSW index. If the filtered search returns nothing, it reruns unfiltered. Off by default (`RETRIEVAL_USE_FILTERS`), so the baseline stays reproducible. Section filtering is deferred to contextual chunking.

**Eval evidence:** Recall@5 0.25 → 0.52, MRR 0.15 → 0.27, correctness 0.42 → 0.53, faithfulness 0.92 → 1.00. Wrong-year chunks 55% → 14%. 5 questions improved and 1 regressed (q020, Bell 2022 capex): the baseline had answered it from the 2023 report's prior-year comparison, which the year filter now excludes. Untested idea: also search report `year + 1`, which repeats the prior year's figures. Known gap: cost/query counts only the answer call, not the ~300-token filter call.

## 2026-10-09 — Keyword search with Postgres full-text search

**Context:** Many questions hinge on exact report terms ("CET1", "attributable to equity holders") that vector search blurs. Keyword search is also one of the two inputs to the hybrid RRF merge.

**Options considered:**
- Postgres full-text search (already in the stack) vs a BM25 library or extension (`rank_bm25`, ParadeDB `pg_search`).
- `plainto_tsquery` / `websearch_to_tsquery` (every term must match) vs OR-ing the terms.

**Decision:** Postgres FTS on a generated `tsvector` column with a GIN index (no re-ingest needed). The question's terms are stemmed and stopwords dropped by `plainto_tsquery`, then joined with OR so any term matches; `ts_rank_cd` ranks chunks with more terms, closer together, higher. It's BM25-like, not true BM25 (no corpus-wide IDF). Same company/year filters and fallback as vector search, via a shared `filter_conditions()`. `RETRIEVAL_MODE=vector|keyword` picks the retriever.

**Eval evidence (retrieval only):** keyword + filters recall@5 0.84, MRR 0.61, vs vector + filters 0.52 / 0.27. Keyword found the right page on every question vector did, plus 10 more; 3 neither found. The two top-5 lists overlap 16% on average. The full run (answers) hit Groq's 200k tokens/day limit after 2 questions; that partial run was discarded, and the runner no longer writes a `results.md` row when any question errors.

**Caveat:** 29 of the 31 questions were written by searching the parsed text for exact phrases, so they use the reports' own wording, which favours keyword search. Paraphrased questions are needed to judge vector search, and the hybrid merge, fairly.

## 2026-10-09 — Paraphrased questions to test wording bias

**Context:** Keyword search scored recall@5 0.84 vs vector's 0.52, but most questions reused the reports' own terms (they were written by searching the text for exact phrases), which favours keyword search.

**Decision:** Added 11 paraphrases (q033–q043) of existing questions: same facts, same verified pages, everyday wording that avoids the report's terms ("profit" for net income, "cellphone customers" for mobile phone subscribers, "core capital cushion" for CET1). Same-fact pairs isolate the effect of wording. The set grew from 31 to 42 questions, so all three retrievers were re-run (retrieval only, no answer-model tokens) for like-for-like rows marked `(n=42)`; earlier rows remain valid for n=31 only.

**Eval evidence (retrieval only, recall@5 on the 11 paired facts):** keyword + filters 0.95 with the report's wording → 0.32 with everyday wording; vector + filters 0.50 → 0.55. On all 42: keyword + filters 0.69, vector + filters 0.50, vector 0.31. Together the two top-5 lists cover 34 of 39 answerable questions, vs 29 for keyword alone, which is the case for a hybrid merge.

## 2026-10-09 — Hybrid search with Reciprocal Rank Fusion

**Context:** Vector and keyword search find different correct pages (their top-5 lists cover 34 of 39 answerable questions together, vs 29 for keyword alone), so merging them should help.

**Options considered:**
- Weighted sum of normalized scores: needs score normalization, since cosine (0–1) and `ts_rank_cd` (unbounded) aren't comparable, and the weights need tuning.
- Reciprocal Rank Fusion: uses ranks only, with one standard constant (k=60). No tuning needed.

**Decision:** RRF over the top 20 of each retriever (`HYBRID_CANDIDATES`), keeping the top 5; `RRF_K=60`. Both retrievers share the same LLM-extracted filters. Written by Claude at the owner's explicit request (`CLAUDE.md` had reserved it for the owner).

**Eval evidence (retrieval only, n=42):** recall@5 0.62, MRR 0.45, vs keyword + filters 0.69 / 0.48 and vector + filters 0.50 / 0.28. More balanced by wording (0.68 report wording / 0.50 everyday, vs keyword's 0.95 / 0.32), but lower overall: when one retriever is clearly right, chunks both rank moderately can outvote it. Sensitivity: depth 5/10/20 × k 10/60 gave 0.62–0.65, so it isn't a tuning problem, and the defaults were kept rather than tuned to 42 questions. The right page is in the combined top-20 candidates for 39/39 answerable questions, so the next step is a reranker over those candidates, not a different merge.

## 2026-10-09 — Cross-encoder reranking over 40 candidates

**Context:** Hybrid search's top-20 vector + top-20 keyword candidates contain the right page for 39/39 answerable questions, but RRF's ordering only puts it in the top 5 for 62%.

**Options considered:**
- Tune or weight RRF: sensitivity runs gave only 0.62–0.65, and tuning to 42 questions risks overfitting.
- LLM reranking (ask the fast model to pick the best chunks): strong, but it adds tokens to a daily-limited quota and is slower.
- Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, the model planned in `CLAUDE.md`): local, free, no quota.

**Decision:** Cross-encoder reranking, behind `RETRIEVAL_RERANK`. The current retriever fetches `RERANK_CANDIDATES=40` (the hybrid pool size; the plan said 50) and the cross-encoder keeps the top 5. Works with any retrieval mode. About 3 s per question on the dev CPU; no question + chunk pair exceeded the model's 512-token limit (max 465).

**Eval evidence (preliminary, retrieval only):** on the 33 answerable questions that completed before the fast model's 200k tokens/day limit (used by filter extraction and the judge) stopped the run, hybrid + filters + rerank scored recall@5 0.88, MRR 0.60, vs 0.62 / 0.45 without reranking and keyword + filters 0.77 / 0.54 on the same questions. Not yet in `results.md`: needs a full 42-question run.

## 2026-10-09 — Cache extracted filters on disk

**Context:** Every retrieval-only eval run called the fast model once per question just to extract filters, and on 2026-10-09 that, together with the judge, used up the fast model's 200k tokens/day. The LLM can also extract slightly different filters from run to run, which mixes extraction noise into retriever comparisons.

**Decision:** `FilterCache` stores filters in `data/cache/filters.json` (git-ignored), keyed by a SHA-256 of the fast model's name, the full system prompt (which includes the companies and years in the database) and the question. Changing any of those makes old entries stop matching, so nothing is stale. Only successful extractions are cached; writes are atomic (temp file + rename). Used by `search()` whenever filters are on.

**Effect:** after one run warms the cache, retrieval-only evals make no LLM calls at all, and every run sees identical filters.

## 2026-10-09 — Contextual chunking: section-aware chunks with a metadata label

**Context:** Naive 450-token windows ignore structure: 46% span pages, and chunks from different years of the same report look alike to vector search.

**Options considered:**
- The plan in `CLAUDE.md`: section-aware splits plus an LLM-written one-line summary per chunk ("contextual retrieval"). The summary would cost ~3–4M fast-model tokens for ~9k chunks, about 18 days of Groq's free tier.
- Section-aware splits with a deterministic label built from metadata we already have.
- Store both strategies in one table (a `chunking` column in every query and key) vs one database per strategy.

**Decision:** A new chunk starts at a heading once the current one has `CHUNK_MIN_TOKENS=100` (annual reports have ~750 headings each, many just table titles); blocks are never cut, and oversized blocks are split at line breaks (table rows stay whole). Each chunk gets a label (`RBC 2023 annual report | Section: X | Page 26`), with `CONTEXT_RESERVE_TOKENS=40` kept free so label + text stay under bge's 512 tokens (max seen: 457). The label is stored in `chunks.context`, embedded with the text, and read by the reranker (`RetrievedChunk.labelled_text()`). It is not in the full-text index; company and year come from filters. Each strategy gets its own database (`POSTGRES_DB=reportlens_contextual`): no schema branching, and switching costs no re-embedding. The naive `ingest_config` fingerprint is unchanged, so existing naive reports weren't reloaded.

**Eval evidence (retrieval only, n=42, no filters):** vector recall@5 0.31 → 0.71 (MRR 0.17 → 0.52); keyword 0.35 → 0.32 (label not indexed); hybrid + rerank 0.45 → 0.63 with the reranker reading text only, 0.71 once it reads the label too. Chunks spanning several pages: ~65% → ~15%. 9,386 chunks (median 255 tokens) vs 5,991. With contextual chunks, reranking adds nothing over vector alone without filters; the combination with filters is still to be measured.

## 2026-10-09 — Phase 2 retrieval comparison with filters (n=42)

**Context:** The rerank and contextual-chunking entries above left the filtered combinations unmeasured. The warm filter cache meant these runs needed no LLM calls after the first one, and every run saw the same filters.

**Eval evidence (retrieval only, n=42, company/year filters on):**

| Chunks | vector | keyword | hybrid + rerank | vector + rerank |
|---|---|---|---|---|
| naive | 0.50 / 0.28 | 0.69 / 0.48 | **0.85 / 0.60** | — |
| contextual | 0.76 / 0.61 | 0.76 / 0.57 | 0.76 / 0.56 | 0.78 / 0.57 |

(recall@5 / MRR). The preliminary 33-question hybrid + filters + rerank result (0.88) holds up on all 42 (0.85).

**Reading:** contextual chunks lift every retriever on its own, but with filters + hybrid + rerank, naive chunks score higher. Part of that is the metric. Recall is scored by page, and a naive chunk is 450 tokens that often span 2–3 pages, while a contextual chunk (median 255 tokens) usually covers one. So five naive hits cover more pages and more text. Some misses are real, though: q019 (BCE capex) and q040 (BCE workforce) find the right page with naive chunks only.

**Decision:** Use naive + hybrid + filters + rerank for the Phase 2 answer-level run, since it is the best measured retrieval stack. Keep the contextual DB. Whether larger contextual chunks (e.g. `CHUNK_MIN_TOKENS` closer to 450) close the gap is an open question for later, not tuned here on 42 questions.

## 2026-10-10 — Phase 2 result: answer-level run of the best stack

**Context:** The Phase 2 retrieval stack (naive chunks + company/year filters + hybrid RRF + cross-encoder rerank) had only retrieval-only scores. This is the one answer-level run that fits Groq's daily quota.

**Eval evidence (n=42):** recall@5 0.85, MRR 0.60, correctness **0.74**, faithfulness 1.00, p50 latency 7.7 s, ~2.9k tokens/question. On the 31 questions shared with the Phase 1 baseline, correctness went from 0.42 to **0.79**. By type: numeric 0.70 (n=20), trend 0.78 (n=9), comparison 0.25 (n=4), lookup/relationship/out-of-corpus 1.00. The remaining misses fall into two groups, each covered by a later phase:
- Adjusted vs reported figures or the wrong table row (q001, q013, q014, q015, q036): the right page is retrieved but the number loses its label. → Phase 3 tables.
- Comparisons across two reports (q002, q043): one top-5 search can't cover both. → Phase 4 agentic layer.

**Run note:** q037 failed with a Groq `400 json_validate_failed`: the provider rejects malformed JSON-mode output server-side, which `llm.structured` didn't retry (it only retried Pydantic errors). It now retries that error under the same single-retry budget. q037 was then re-run on its own with the same config and merged into the run file, rather than spending ~125k tokens on a full re-run. The Notes column says so.

**Not done in Phase 2: section filter.** The filter extracts only company and year. A section filter was deferred until contextual chunking, but the chosen stack uses naive chunks, whose section labels are unreliable, and no failure above is caused by the wrong section. It is left out rather than built without eval evidence; revisit if the Phase 4 router needs it.

**Comparability caveat:** the Phase 1 baseline and metadata-filters rows are n=31. Re-running them on n=42 would cost one day of quota each; the 31-question overlap above gives a like-for-like comparison in the meantime.

## 2026-10-11 — Table-mode parsing in isolated 20-page slices

**Context:** Phase 3 turns on Docling's table-structure model (`PARSE_TABLES=true`). On this Intel Mac (CPU, `torch==2.2.2`), whole-report conversions failed at random: two full ingest runs segfaulted (exit 139) partway through their second report, and RBC 2021 ran for over 1.5 hours with no result. Diagnosis: timed 20-page slices of RBC 2021 each finished in 1–4 minutes (about 27 min total), except one slice that segfaulted. That slice then passed 3 times in a row, and each of its 20 pages passed twice on its own. A second conversion in the same process was also much slower than the first. So the failure is random, and it gets likelier the longer one process converts.

**Options considered:**
- Retry whole reports: a 25+ minute conversion that fails at random wastes the most time per failure, and RBC never completed.
- Upgrade torch/Docling: not possible on Intel Mac (no newer wheels; see the 2026-10-08 entry).
- Convert short page slices, each in a fresh subprocess with a timeout and retries, then merge them.

**Decision:** In table mode, `convert_pdf` splits the PDF into 20-page slices (`TABLE_SLICE_PAGES`). Each slice runs in a fresh Python process (`python -m reportlens.ingest.parse`) with a 10-minute timeout (`SLICE_TIMEOUT_S`) and up to 3 attempts (`SLICE_ATTEMPTS`). The slices are merged with `DoclingDocument.concatenate`, which keeps the original page numbers (checked on BCE 2024 pages 1–6: same pages and the same 42 blocks as the two slices). The ingest process itself no longer runs Docling, so embedding and table parsing never share a process. Light mode is unchanged.

**Eval evidence:** none (robustness only). Cost: about one model load per slice (~15 s × 12 slices per report).

## 2026-10-13 — Table chunks measured; naive chunks stay the text-search default

**Context:** With all 12 reports parsed in table mode (`PARSE_TABLES=true CHUNKING=tables`, database `reportlens_tables`: 8,258 chunks, 3,943 of them tables), the Phase 2 stack (filters + hybrid + rerank) was run on table chunks and compared with the same stack on naive chunks.

**Options considered:**
- Switch the main stack to table chunks.
- Keep naive chunks for text search, and use the table parses as the source for the `financials` table and the SQL tool.

**Decision:** Keep naive chunks as the default for text search. Use the table parses (`data/parsed/*/{year}.tables.json`) to build the `financials` table next.

**Eval evidence (n=42):**

| Stack | Recall@5 | MRR | Correctness | Faithfulness | p50 latency |
|-------|----------|-----|-------------|--------------|-------------|
| Naive chunks (2026-10-10) | 0.85 | 0.60 | 0.74 | 1.00 | 7.7s |
| Table chunks | 0.81 | 0.55 | 0.70 | 0.92 | 14.9s |

- **Tables fixed the label problem they were meant to fix:** q013, q014, q015 went from wrong to right, and q006 from partly to fully right (+3.5). With naive chunks the model had picked the wrong row (an adjusted or segment figure); the table chunk gave it the right row.
- **Losses (about −5):** q008 and q040 lost the right page in retrieval and were refused; q012 was refused even though the right page was retrieved; q021 used a Q4 column instead of the full year; q004 and q024 dropped to partly right because they left out details.
- **Tables are under-retrieved:** they are 48% of chunks but only about 20% of the distinct chunks retrieved (31 of 159). Untested hypothesis: Markdown pipes and grids embed poorly. One fix to test: embed a plain-text version of each table but show the Markdown when answering.
- **q001 is still wrong, now for a new reason:** the model cited T21 *Canadian Banking financial performance* and reported that segment's $4,019 million, not the bank's $7,410 million. The table was intact, but the model ignored that it covers one segment. That's a case for the `financials` table, where segment and total are separate columns, not something the model must infer.
- **Part of the faithfulness drop is the judge's error:** in q023 the cited chunk says "our team consisted of 40,390 employees", yet the judge marked "BCE had 40,390 employees" as unsupported. q004 and q039 were not checked.

**Cost:** table-mode parsing took about 35–45 min per report on this CPU (vs ~7 min in light mode); one full answer-level run (~113k answer-model tokens, most of a day's free-tier quota).

## 2026-10-14 — Table cells extracted with rules, not an LLM

**Context:** The `financials` table and SQL tool need numbers with labels. The table-mode parses hold 2,302 tables as Markdown grids whose first line is the header row (Docling flattens multi-level headers, e.g. `For the year ended - 2023 - After-tax`).

**Options considered:**
- LLM extraction per table into `(company, year, metric, value, unit, page)`: flexible, but thousands of tables would take days of the free tier's fast-model quota, and the output would need checking anyway.
- Rules: every numeric cell becomes a row labelled with its row, column, unit and period, and a small curated `financials` table is built on top (Step 2).

**Decision:** Rules. `ingest/tables.py` writes every numeric cell to `table_cells` (110,647 cells from 12 reports, loaded in ~13 s by `scripts/load_tables.py`). Parentheses mean negative; a lone `-` is nil, not 0. Unit comes from the cell (`%`, `bps`), then per-share row labels (`dollars`), then the column header, then the table's corner cell or title (`millions`, `billions`, `thousands`). Period year and kind (`year`, `quarter`, `change`, `unknown`) come from the column header, falling back to the corner cell or title. A row with a label but no values (e.g. "Reported Results", "Adjusted Results") becomes the `row_group` of the rows below it, so reported and adjusted figures with the same row label stay apart. `table_cells` lives in the main `reportlens` database, next to the naive chunks, and isn't tied to `documents`: it's replaced in full on every load.

**Eval evidence:** none yet (Step 4). Spot checks against eval answers: Scotiabank 2023 net income attributable to equity holders is 7,410 under *Reported Results* (8,320 under *Adjusted Results*) and under the *Total* column of the segment table (4,019 under *Canadian Banking*); RBC 2024 net income 16,240; RBC 2024 CET1 ratio 13.2%; Scotiabank 2023 diluted EPS 5.78; BCE 2024 capex 3,897. Known gaps: 22% of cells have no unit (e.g. BCE's segment table states no scale) and 18% have no period year (mostly columns with an empty header).

## 2026-10-14 — financials: 12 headline metrics picked from table_cells by rules and a vote

**Context:** Step 2 of the SQL plan. `table_cells` has every number, but a question like "RBC's net income in 2024" matches hundreds of cells across tables (segment tables, quarterly tables, notes on subsidiaries). The SQL tool needs one clean row per figure.

**Options considered:**
- LLM picks the right cell per metric and report: costs quota and needs checking anyway.
- Rules: each metric is a few row-label regexes and a stored unit (`ingest/metrics.py`); every matching full-year cell is a candidate; the most common value wins.

**Decision:** Rules plus a vote (`ingest/financials.py`, rebuilt by `scripts/load_tables.py`). 12 metrics: net income, net income to shareholders, revenue, diluted EPS, CET1 ratio, ROE, total assets, capex, adjusted EBITDA, employees, mobile phone subscribers, dividends per share. Each row records:
- **scope**: `total`, or a segment read from the column header (e.g. *Canadian Banking*) or the table title. Columns that are some other breakdown (e.g. *U.S.*) are skipped.
- **basis**: `adjusted` if the row, row group or column says so, else `reported`.
- **report_year**: a report's figures for its own year *and* the prior year are both kept, one row per report, because reports restate (Scotiabank's 2023 diluted EPS is 5.78 in the 2023 report, 5.72 in the 2024 one; q014 and q016 expect the restated figures).
- **agreeing_cells / candidate_values**: how many cells printed the winning value, and how many distinct values were seen, as a confidence signal for the SQL tool and for review.
- Ties go to the value printed earliest in the report, usually the highlights table.

Fixes to `table_cells` found while checking (re-extracted, no re-parse): footnote markers are stripped from labels (`CET1 ratio (3)`, LaTeX superscripts); leading body rows of words or years are merged into the column headers, because Docling sometimes takes a title row as the header (this dropped cells with no period year from 18% to 14%); a scale named in the row label (`Number of employees (thousands)`) is used. Counts take a scale only from their own row, so a segment table's "($ millions)" no longer turns 2,788 employees into 2.8 billion.

**Eval evidence:** `scripts/check_financials.py` looks up every number in the expected answers of the 33 numeric, trend and comparison questions: **29/33** have a matching total-scope row, including q001's reported 7,410 (not the segment's 4,019) and both restated figures. The 4 misses are expected: q022/q039 expect an 83.9% change (derivable from stored net earnings, 375 vs 2,327), and q023/q040 expect BCE's 40,390 employees, which the tables print only as 40 thousand (40,390 is in the text). Of 671 rows, 187 saw more than one candidate value and 317 rest on a single cell, so low-agreement rows deserve a look before trusting them.

## 2026-10-14 — Text-to-SQL tool: three safety layers, one retry, answers cite table rows

**Context:** Step 3 of the SQL plan. An LLM writes SQL from the question, so the SQL must be treated as untrusted: it must never write, read other tables, or run for long.

**Options considered:**
- Trust a read-only role alone: simple, but a mistake in grants would be the only barrier, and the LLM gets no useful feedback on what it did wrong.
- Validate with regexes: easy to fool (comments, CTEs, nested queries).
- Parse with `sqlglot` and check the syntax tree, *and* run as a read-only role: two independent barriers, with clear rejection messages to feed back to the LLM.

**Decision:** Three layers.
1. **`sources/sql_safety.py`** parses the SQL with `sqlglot` (new dependency) and rejects anything but exactly one SELECT (or WITH/UNION of SELECTs) that reads only `financials` and `table_cells` (CTE names allowed, schema-qualified names rejected). It also rejects SELECT INTO, FOR UPDATE, and functions that reach outside the query (`pg_*`, `lo_*`, `dblink`, `set_config`, `current_setting`, `query_to_*`). A LIMIT of at most `SQL_ROW_LIMIT` (50) is enforced.
2. **A read-only role** (`reportlens_reader`, created by `core/db.py` on connect) has SELECT on those two tables only, `default_transaction_read_only = on` and a `statement_timeout`. Checked live: DELETE, CREATE TABLE and reading `chunks` all fail; `pg_sleep(6)` is cancelled at 5 s.
3. **`connect_reader()`** opens a read-only, non-autocommit session with the same timeout, and rolls back after every query.

The fast model writes the SQL (`SqlQuery` via `llm.structured`); if it is rejected or Postgres raises an error, the error message goes back for **one** retry. An empty SQL string means the tables can't answer (e.g. "why" questions), so no query runs. The main model answers from the rows, numbered `[n]` as column=value pairs, and each citation maps to the row's company, report and page. No rows means the fixed not-found reply, with no main-model call.

**Eval evidence:** none yet (Step 4). In manual tests the first prompt let the SQL model skip the `scope`/`basis` filters, and the answer model then reported Scotiabank's Canadian Banking segment (4,022) as the bank's net income: q001's original failure again. The prompt now requires those filters (defaulting to `total` / `reported`) and the label columns (`metric`, `scope`, `basis`, `unit`, or the table and row labels for `table_cells`) in every SELECT. After that change: q001 gives 7,410 (page 22), BCE capex 2022→2024 gives 5,133 → 3,897, the 2024 CET1 comparison gives RBC 13.2% vs Scotiabank 13.1%, and RBC's 2023 provision for credit losses (not a curated metric) comes from `table_cells`: 2,468 million.

## 2026-10-14 — SQL tool measured: better than text search on numeric questions, and the two complement each other

**Context:** Step 4 of the SQL plan. `run_evals.py --source sql --types numeric,trend,comparison` answers the 33 numeric, trend and comparison questions with the SQL tool. `as_answer` turns each returned row into a pseudo-chunk (company, report, page, `column=value` text), so the runner's recall, correctness and faithfulness scoring works unchanged.

**Decision:** Keep the SQL tool as a separate source for the Phase 4 router; don't merge it into text search.

**Eval evidence (n=33, same questions):**

| Source | Correctness | Faithfulness | Main-model tokens/question |
|--------|-------------|--------------|----------------------------|
| Text search (Phase 2 stack, 2026-10-10 run) | 0.67 | 1.00 | ~2,900 |
| SQL tool | **0.79** | 0.98 | **~560** |
| Better of the two per question (a perfect router) | 0.94 | | |

- **SQL wins 11 questions,** exactly the "number with the wrong label" failures: q001 (bank total, not a segment), q013/q014/q015 (reported, not adjusted; restated prior year), q033/q034 (everyday wording), q002/q043 (comparisons across years or banks).
- **Text wins 6:** "why" questions the SQL model rightly declines (q022, q039), BCE's headcount that the tables round to 40 thousand (q023, q040: 40,390 is only in the text), and two figures written at a precision the judge marked partly right (q004, q019).
- **Recall (0.47) isn't comparable with text search:** the SQL tool often reads a figure from a different table than the pages listed in the eval set, so a correct answer can have recall 0.
- q036 hit a provider JSON error and was re-run alone, as in Phase 2.

**Fixed after this run:** in q028 the SQL model picked the winner in SQL (`ORDER BY value DESC LIMIT 1`), so the answer model saw only RBC and refused. The prompt now asks for one row per company and year in comparisons; q028 passes manually. A full re-run with that rule ran out of the free tier's daily fast-model tokens after 9 of 33 questions (same 0.94 as before on those 9), so it was re-run the next day (see 2026-10-15). Because the fix came from looking at an eval result, the re-run should be read as tuned on this set.

**Cost note:** the SQL tool spends mostly fast-model tokens (a ~2k-token schema prompt per question, plus the two judge calls in evals), so one 33-question SQL eval uses roughly a third of the fast model's daily quota.

## 2026-10-15 — SQL tool re-run with the comparison rule: 0.80

**Context:** the re-run left pending on 2026-10-14, after the SQL prompt gained the rule "one row per company and year in comparisons; don't pick the winner in SQL".

**Decision:** keep the rule.

**Eval evidence (n=33, numeric/trend/comparison):** correctness 0.79 → **0.80**, faithfulness 0.98 → **1.00**, ~500 main-model tokens per question. q028 (the CET1 comparison that motivated the rule) went 0 → 1, and q019 0.5 → 1; q014 and q034 went 1 → 0.5 on wording (one stated a restated prior-year figure, one omitted context the expected answer includes). So the rule fixes what it targeted, and the rest is run-to-run noise of about one question. The rule came from reading an eval result, so this number is tuned on this set. q039 hit the per-minute rate limit (the graph extraction was running at the same time) and was re-run alone.

## 2026-10-16 — Knowledge graph: segments from financials, subsidiaries and executives extracted by the fast LLM

**Context:** Phase 3 graph step. Relationship questions ("Which American bank does RBC own?", "Who was Scotiabank's CFO in 2024?") are poorly served by both text chunks and SQL. The graph needs Company, Segment, Subsidiary and Executive nodes, each relationship citable to a page like the other sources.

**Options considered:**
- **Rules over `table_cells`:** free, but `table_cells` only holds numeric cells (the principal-office column is lost), and RBC prints a holding company and the companies held through it in one cell ("RBC US Group Holdings LLC RBC USA Holdco Corporation RBC Capital Markets, LLC City National Bank"), which no rule can split reliably.
- **LLM extraction over every chunk:** finds the most, but ~6,000 chunks would cost several days of the free tier's quota, and most chunks hold no relationships.
- **LLM extraction over a few selected passages per report:** the subsidiaries table (picked by its heading) and the short blocks that pair a name with an officer title (signatures under management's report, leadership pages).

**Decision:** the third, plus no LLM where the data already exists.
- **Segments** come from `financials` (the segments each report has figures for, with the first page printing one), so the graph shows BCE moving from Wireless/Wireline to Bell CTS in 2023 and RBC splitting Personal & Commercial Banking.
- **Subsidiaries:** the fast model reads each report's subsidiaries table in batches of 15 rows (header repeated) and returns name, location and ownership share; the page is the table's page.
- **Executives:** the fast model reads ≤ 300-character passages that contain an officer title (≤ 60 per report, each marked `[p. N]`) and returns name, title, role (`ceo`/`cfo`/`other`) and page. A person no passage names is dropped, so a made-up name can't reach the graph; board members signing the statements ("Director") are dropped too.
- **Names are matched by keys** (`ingest/graph_names.py`): case, punctuation, `&`/"and" and a leading "The" are ignored for companies, and middle initials for people ("David I. McKay" = "David McKay"). Footnote markers and printed shares ("(99.79%)") are stripped from names, and company names from the end of titles.
- **Every relationship carries `report_year` and `source_page`** and appears once per report, so "in 2023" is a filter and changes over time (a new CFO) are visible. The graph is rebuilt from scratch on each load; LLM output is cached by model, prompt and input (`data/cache/graph_facts.json`), so rebuilding costs nothing.
- **Neo4j 5 Community** runs in Docker next to Postgres. Community edition has no role-based access, so the Cypher tool (next step) will rely on read transactions, which the server refuses to write in, plus a query check, as the SQL tool does.

**Provider issues found and fixed in `core/llm.py`:** Groq's JSON mode rejected some inputs every time (`json_validate_failed`) although the plain reply was valid JSON, so `structured()` now retries without JSON mode after a rejection (Pydantic still validates); this was probably also q036's failure. And Groq's default 2,048-token output cap cut off long lists from the reasoning model, so `chat()`/`structured()` take an optional `max_tokens` (extraction uses 6,000: prompt plus cap must stay under the 8,000 tokens-per-minute limit).

**Known limitation:** the table parse of Scotiabank's 2023 subsidiaries table is garbled (names split across rows next to the wrong locations), so a few 2023 names are wrong, e.g. "Peru Holdings S.A."; one-word fragments such as "Scotia" are dropped by rule.

**Eval evidence:** none yet. The eval set has two relationship questions (q027, q042: City National Bank), and the graph has City National Bank under RBC in all four reports (pages 229, 233, 234, 247). More relationship questions are needed to measure the Cypher tool (the owner adds them).

## 2026-10-16 — Graph query tool: text check plus a read transaction, reusing the SQL tool's answer step

**Context:** Phase 3 Cypher tool. The fast LLM writes Cypher, which is untrusted like its SQL. Neo4j 5 Community has no role-based access, so there is no read-only user to run it as (the SQL tool's second layer).

**Options considered:**
- **Parse the Cypher:** the safest check, but there is no maintained Cypher parser for Python (the openCypher grammar would need ANTLR and a generated parser to maintain).
- **Rely on the read transaction alone:** the server refuses every write in it, but it still allows `CALL` procedures (e.g. `dbms.*`), `LOAD CSV` (reads local files and URLs) and `USE` (another database).
- **Fixed query templates** (one per question kind): fully safe, but every new kind of question would need code.
- **A check on the query's text, then a read transaction.**

**Decision:** the last one.
- `sources/cypher_safety.py` first blanks out string literals and comments (keeping positions, so a name such as 'Set Inc.' or a `//` inside a string can't fool it), then requires one statement starting with a read clause, rejects write and admin keywords, `CALL`, `LOAD`, `USE` and `apoc.*`/`dbms.*`/`db.*` functions, allows only the 4 labels and 3 relationship types (map keys such as `{id: 'rbc'}` are skipped), and adds or caps a `LIMIT` of 50. Unknown labels are not a security problem, but the message helps the LLM correct itself.
- The query runs with `RoutingControl.READ` and a 5 s transaction timeout, and only the first 50 records are fetched (a `LIMIT` after `UNION` limits only the last part). Tests check against the real server that a write is refused even when the text check is bypassed, and that a slow query times out.
- `graph_tool.py` follows `sql_tool.py` step for step (write, check, run, one retry with the error, answer from numbered rows) and reuses its answer step, row citations and eval adapter through a shared `RowsAnswer` model, so the two tools are easy to compare.
- Prompt rules learned from four trial questions (none from the eval set's answers): match names with `CONTAINS`, never exactly (the model wrote `{name: 'Scotiabank Chile'}`, but the name is "Scotiabank Chile S.A."); return locations instead of filtering on a country (RBC prints "U.S.", Scotiabank omits the country); with no year named, return every year, latest first, so a cut at 50 rows drops the oldest.
- The Cypher writer gets two retries instead of one: Groq can reject the JSON (one retry drops JSON mode) and then return an empty reply.

**Eval evidence (n=2, relationship):** correctness 0.50, faithfulness 0.50, ~850 tokens per question. q042 ("Which American bank does RBC own?") correct with all four reports' pages; q027 ("RBC's U.S. commercial bank") wrong, answering RBC USA Holdco Corporation: the graph has names and offices but not what business a subsidiary is in. Text search gets both (1.0 in every recorded run). Two questions can't rank the sources; questions on executives, ownership shares and segment changes are needed (the owner adds them). Known weakness seen in trials: for "which American bank", the model sometimes filters on 'bank' in the name without returning locations, so the "American" part comes from its own knowledge rather than the rows.

## 2026-10-16 — Chart summaries: rule-picked candidates, a vision model, estimates marked

**Context:** Phase 3 multimodal parsing. Docling's layout model marks pictures (~1,170 in the 12 light-mode parses) but drops the text inside them, so chart contents were invisible to search.

**Options considered:**
- **Send every picture to a vision model:** ~1,170 calls at ~2.8k tokens each, most of them photos and logos.
- **Docling's picture classifier:** needs a re-parse of every report (~7 min each in light mode, much longer with the extra model on this CPU).
- **Store only the PDF text inside each picture, no LLM:** free and exact, but axis ticks and labels without the chart's shape don't say what it shows, and bar values that aren't printed are lost.
- **A rule picks candidates, a vision model describes them.**

**Decision:** the last one.
- **Candidates:** pictures with at least 4 numbers in the PDF text layer inside their box (384). These reports draw charts as vector graphics with real text; photos and logos have none. All of the first 51 described were real charts.
- **Vision model:** `qwen/qwen3.8-27b`, the only image-capable model on this Groq account (`LLM_MODEL_VISION`), called through `core/llm.py` (new `vision` tier and `image_message`). It gets the crop (≤ 1,024 px), the printed text and the nearest heading above the picture, which often holds the title and unit outside the picture ("OPERATING REVENUES ($ millions)"); without the heading the first trial said "a financial metric … dollars".
- **Estimates are marked:** each value has `printed`; a value read off a bar's height or a line's position is written into the chunk as "(estimated from the chart)". A probe read 98,000 for a bar at about 110,000, so these must never pass as printed figures. In the first 51 summaries, 12 of 201 values were estimates, all from one line chart without printed values.
- **Storage:** one chunk per chart in the main database, `kind = 'figure'`, ids `{company}-{year}-fig-NNN`. Retrieval reads them only with `RETRIEVAL_FIGURES=true`, so every earlier eval row stays comparable. Summaries are cached by model, prompt and image.

**Cost:** ~2.8k vision tokens per chart (the image costs ~1.8k even when small, so smaller crops save little), about 1 call per minute under the per-minute limit, and the daily limit stopped the first run after ~63 calls. Describing all 384 takes about 6 daily runs.

**Known limitation:** BCE's 2022 report has 1 candidate: its charts are images without a text layer, so the rule misses them.

**Eval evidence:** none yet; a retrieval-only comparison (best stack, with and without figure chunks) waits until all reports are described.

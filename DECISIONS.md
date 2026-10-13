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

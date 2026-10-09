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

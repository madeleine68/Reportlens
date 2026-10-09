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

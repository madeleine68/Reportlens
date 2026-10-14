# Eval results

Append-only. One row per experiment.

| Date | Experiment | Recall@5 | MRR | Correctness | Faithfulness | p50 latency | Cost/query | Notes |
|------|------------|----------|-----|-------------|--------------|-------------|------------|-------|
| 2026-10-09 | baseline | 0.25 | 0.15 | 0.42 | 0.92 | 21.1s | $0.0000 (2,920 tok) | n=31; naive 450-tok chunks, bge-small, vector top-5; BCE/RBC/Scotiabank 2021-24 |
| 2026-10-09 | metadata-filters | 0.52 | 0.27 | 0.53 | 1.00 | 20.1s | $0.0000 (2,905 tok) | n=31; baseline + LLM company/year filters (fast model); no section filter |
| 2026-10-09 | keyword+filters | 0.84 | 0.61 | n/a | n/a | 0.5s | $0.0000 (0 tok) | n=31; retrieval only; Postgres FTS (OR query, ts_rank_cd) instead of vector, + company/year filters |
| 2026-10-09 | vector (n=42) | 0.31 | 0.17 | n/a | n/a | 0.0s | $0.0000 (0 tok) | n=42; retrieval only; baseline retriever on the 42-question set incl. paraphrases |
| 2026-10-09 | vector+filters (n=42) | 0.50 | 0.28 | n/a | n/a | 0.5s | $0.0000 (0 tok) | n=42; retrieval only; vector + company/year filters |
| 2026-10-09 | keyword+filters (n=42) | 0.69 | 0.48 | n/a | n/a | 2.4s | $0.0000 (0 tok) | n=42; retrieval only; Postgres FTS + company/year filters |
| 2026-10-09 | hybrid+filters (n=42) | 0.62 | 0.45 | n/a | n/a | 0.6s | $0.0000 (0 tok) | n=42; retrieval only; RRF (k=60) over top-20 vector + top-20 keyword, + company/year filters |
| 2026-10-09 | vector, naive chunks (n=42) | 0.31 | 0.17 | n/a | n/a | 0.0s | $0.0000 (0 tok) | n=42; retrieval only; no filters; chunking=naive |
| 2026-10-09 | keyword, naive chunks (n=42) | 0.35 | 0.18 | n/a | n/a | 0.1s | $0.0000 (0 tok) | n=42; retrieval only; no filters; chunking=naive |
| 2026-10-09 | hybrid+rerank, naive chunks (n=42) | 0.45 | 0.30 | n/a | n/a | 4.1s | $0.0000 (0 tok) | n=42; retrieval only; no filters; chunking=naive |
| 2026-10-09 | vector, contextual chunks (n=42) | 0.71 | 0.52 | n/a | n/a | 0.0s | $0.0000 (0 tok) | n=42; retrieval only; no filters; chunking=contextual |
| 2026-10-09 | keyword, contextual chunks (n=42) | 0.32 | 0.21 | n/a | n/a | 0.2s | $0.0000 (0 tok) | n=42; retrieval only; no filters; chunking=contextual |
| 2026-10-09 | hybrid+rerank, contextual chunks (n=42) | 0.63 | 0.33 | n/a | n/a | 4.5s | $0.0000 (0 tok) | n=42; retrieval only; no filters; chunking=contextual |
| 2026-10-09 | hybrid+rerank (reads label), contextual chunks (n=42) | 0.71 | 0.49 | n/a | n/a | 4.9s | $0.0000 (0 tok) | n=42; retrieval only; no filters; chunking=contextual; reranker reads label + text |
| 2026-10-09 | vector+rerank (reads label), contextual chunks (n=42) | 0.68 | 0.50 | n/a | n/a | 4.6s | $0.0000 (0 tok) | n=42; retrieval only; no filters; chunking=contextual; reranker reads label + text |
| 2026-10-09 | hybrid+filters+rerank, naive chunks (n=42) | 0.85 | 0.60 | n/a | n/a | 4.1s | $0.0000 (0 tok) | n=42; retrieval only; RRF top-20+20 → cross-encoder top-5; company/year filters; chunking=naive |
| 2026-10-09 | vector+filters, contextual chunks (n=42) | 0.76 | 0.61 | n/a | n/a | 0.0s | $0.0000 (0 tok) | n=42; retrieval only; company/year filters; chunking=contextual |
| 2026-10-09 | keyword+filters, contextual chunks (n=42) | 0.76 | 0.57 | n/a | n/a | 0.0s | $0.0000 (0 tok) | n=42; retrieval only; company/year filters; chunking=contextual |
| 2026-10-09 | hybrid+filters+rerank, contextual chunks (n=42) | 0.76 | 0.56 | n/a | n/a | 4.0s | $0.0000 (0 tok) | n=42; retrieval only; RRF top-20+20 → cross-encoder top-5 (reads label); company/year filters; chunking=contextual |
| 2026-10-09 | vector+filters+rerank, contextual chunks (n=42) | 0.78 | 0.57 | n/a | n/a | 4.2s | $0.0000 (0 tok) | n=42; retrieval only; vector top-40 → cross-encoder top-5 (reads label); company/year filters; chunking=contextual |
| 2026-10-10 | hybrid+filters+rerank, naive chunks (n=42) | 0.85 | 0.60 | 0.74 | 1.00 | 7.7s | $0.0000 (2,908 tok) | n=42; answers; RRF top-20+20 → cross-encoder top-5; company/year filters; chunking=naive; q037 re-run alone after a provider JSON error (fixed in llm.structured) |
| 2026-10-13 | hybrid+filters+rerank, table chunks (n=42) | 0.81 | 0.55 | n/a | n/a | 4.1s | $0.0000 (0 tok) | n=42; retrieval only; RRF top-20+20 → cross-encoder top-5; company/year filters; chunking=tables (TableFormer) |
| 2026-10-13 | hybrid+filters+rerank, table chunks (n=42) | 0.81 | 0.55 | 0.70 | 0.92 | 14.9s | $0.0000 (2,696 tok) | n=42; RRF top-20+20 → cross-encoder top-5; company/year filters; chunking=tables (TableFormer) |
| 2026-10-14 | sql tool: financials + table_cells (n=33) | 0.47 | 0.46 | 0.79 | 0.98 | 8.1s | $0.0000 (562 tok) | n=33; source=sql; types=numeric,trend,comparison; fast model writes SQL (1 retry); main model answers from rows; recall = expected page among first 5 rows; q036 re-run alone after a provider JSON error |

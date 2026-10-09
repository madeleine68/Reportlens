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

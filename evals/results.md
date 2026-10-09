# Eval results

Append-only. One row per experiment.

| Date | Experiment | Recall@5 | MRR | Correctness | Faithfulness | p50 latency | Cost/query | Notes |
|------|------------|----------|-----|-------------|--------------|-------------|------------|-------|
| 2026-10-09 | baseline | 0.25 | 0.15 | 0.42 | 0.92 | 21.1s | $0.0000 (2,920 tok) | n=31; naive 450-tok chunks, bge-small, vector top-5; BCE/RBC/Scotiabank 2021-24 |
| 2026-10-09 | metadata-filters | 0.52 | 0.27 | 0.53 | 1.00 | 20.1s | $0.0000 (2,905 tok) | n=31; baseline + LLM company/year filters (fast model); no section filter |

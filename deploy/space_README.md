---
title: ReportLens
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: RAG over Canadian annual reports, with page citations
---

# ReportLens

Questions about RBC, Scotiabank and BCE annual reports (fiscal 2021–2024), answered with page
citations by a router across text search (with corrective RAG), SQL over financial tables and a
Neo4j knowledge graph.

The demo is password-protected and rate-limited to protect a free-tier LLM quota.
Code, design decisions and eval results: https://github.com/madeleine68/Reportlens

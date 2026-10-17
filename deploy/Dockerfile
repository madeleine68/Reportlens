# The API + UI on Cloud Run (Phase 5). Built by Cloud Build from the folder
# scripts/deploy_cloudrun.py assembles: requirements.txt, src/.
FROM python:3.12-slim

# Run as an unprivileged user with a home and the app's data folder (caches).
RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only PyTorch: Cloud Run has no GPU here, and the CUDA build is ~2 GB larger.
RUN pip install --no-cache-dir uv
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

COPY --chown=user src ./src
RUN mkdir -p data/cache && chown -R user /app
USER user
ENV HOME=/home/user PYTHONPATH=/app/src PYTHONUNBUFFERED=1

# Download the embedding and reranking models into the image, so a cold start doesn't fetch
# them. Settings need LLM values to load; placeholders are fine because nothing calls the LLM.
RUN LLM_BASE_URL=unused LLM_API_KEY=unused LLM_MODEL_MAIN=unused LLM_MODEL_FAST=unused \
    python -c "from reportlens.api.service import warm_up_models; warm_up_models()"

# The best text-search stack (Phase 2); models are already local.
ENV HF_HUB_OFFLINE=1 RETRIEVAL_USE_FILTERS=true RETRIEVAL_MODE=hybrid RETRIEVAL_RERANK=true

EXPOSE 7860
CMD ["uvicorn", "reportlens.api.main:app", "--host", "0.0.0.0", "--port", "7860"]

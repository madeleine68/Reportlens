#!/usr/bin/env bash
# Run Phase 4 evals one after another on the free tier, waiting out the daily token limit.
# Each run resumes where it stopped (run_evals.py --resume), so a run can span several days.
# Usage (detached, survives closing the terminal; the machine must stay awake):
#   nohup scripts/eval_loop.sh >> data/cache/evals_run.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

# The best text-search stack (Phase 2), which the router and corrective RAG search with.
export RETRIEVAL_USE_FILTERS=true RETRIEVAL_MODE=hybrid RETRIEVAL_RERANK=true
export HF_HUB_OFFLINE=1 LOG_LEVEL=WARNING

WAIT_DAILY=10800  # 3 hours: the daily limit is a rolling 24-hour window
WAIT_ERRORS=600   # 10 minutes before re-asking questions that errored (e.g. per-minute limits)
MAX_ERROR_RETRIES=3

run() {  # name, then run_evals.py arguments
  local name="$1"; shift
  local retries=0
  while true; do
    echo "=== $(date) $name"
    uv run python evals/run_evals.py --name "$name" --resume "$@"
    local code=$?
    case $code in
      0) echo "=== $(date) $name finished"; return 0 ;;
      3) sleep "$WAIT_DAILY" ;;
      *) retries=$((retries + 1))
         if [ "$retries" -gt "$MAX_ERROR_RETRIES" ]; then
           echo "=== $(date) $name gave up after $MAX_ERROR_RETRIES retries (exit $code)"
           return 1
         fi
         sleep "$WAIT_ERRORS" ;;
    esac
  done
}

# Done 2026-10-19: "routed corrective" --source routed, "corrective" --source corrective.
# Settings placed before `run` apply only to that run.
# Done 2026-10-22: "routed" (ROUTED_CORRECTIVE=false), "routed corrective self-rag" (SELF_RAG=true).
RETRIEVAL_FIGURE_SLOTS=1 run "routed corrective + chart slot" --source routed --notes "RETRIEVAL_FIGURE_SLOTS=1"
echo "=== $(date) all evals done"

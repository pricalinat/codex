# Paper Retrieval Iteration Notes

## Root Cause Found
The previous system had no dedicated paper retrieval/relevance component. Relevance quality was limited by missing capability, not just bad parameters.

## Iteration Changes
- Added hybrid retriever with query-goal scoring signals:
  - title overlap
  - abstract overlap
  - keyword overlap
  - goal alignment overlap
- Added token normalization (light stemming for plural forms) to reduce lexical mismatch.
- Added benchmark-based metrics (`hit_rate@k`, `mrr@k`).
- Added weight tuning loop with `max_rounds=100` and target hit-rate threshold.
- Integrated retriever as `paper_search` agent in orchestrator service defaults.

## Usability Gate
Retriever is considered usable when benchmark `hit_rate@1 >= 1.0` on current regression set and all tests pass.

## Next Iterations
1. Replace in-memory demo papers with your real paper index.
2. Add hard negatives from your failed retrieval logs.
3. Add cross-encoder reranking after first-stage retrieval.
4. Track online acceptance/rejection feedback to continuously update weights.

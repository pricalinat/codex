# Ultrawork Iteration 1 (Guided)

## Goal
Improve retrieval robustness for acronym/abbreviation queries without regressing current ranking behavior.

## Scope
- Add abbreviation-aware query expansion in `PaperRetriever`.
- Keep current scoring model and two-stage reranking behavior.
- Add focused tests for new behavior and tie determinism.

## Files
- `src/multi_agent/paper_retrieval.py`
- `tests/test_paper_retrieval.py`

## Success Criteria
- Query with abbreviation (e.g. `TFT`) can match relevant paper with expanded terms.
- Ranking for equal scores is deterministic.
- Existing retrieval tests still pass.
- Full suite passes: `python3 -m unittest -v`.

## Run Commands (copy in order)
```bash
# 1) Plan only
opencode run -m opencode/big-pickle \
"Plan only. Do not edit files. Provide exact TDD steps for: Add abbreviation-aware query expansion + deterministic tie-break tests in paper_retrieval."

# 2) Implement with TDD
opencode run -m opencode/big-pickle \
"Implement with TDD: add tests first, then minimal code changes for abbreviation-aware query expansion in PaperRetriever and deterministic tie-break behavior. Run retrieval tests."

# 3) Run local verification
python3 -m unittest tests/test_paper_retrieval.py tests/test_agents.py tests/test_e2e.py -v
python3 -m unittest -v

# 4) Review pass
opencode run -m opencode/big-pickle \
"Review current workspace changes for bugs/regressions and missing tests in retrieval flow. Return findings only."
```

## Fallback
- If opencode outputs malformed or overly broad patches, restore only affected file(s), tighten prompt to "minimal diff", and rerun from step 1.

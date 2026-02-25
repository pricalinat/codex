# Multi-Agent Core (Center-Orchestrated)

This workspace contains a Python prototype for a center-orchestrated multi-agent system.

## What is implemented
- In-memory task/event store
- Orchestrator with dependency scheduling, retry logic, and `max_rounds` control (default 100)
- Agent registry with built-in `echo` and `summary` agents
- Service layer for task submission and querying
- Optional FastAPI app factory (`create_app`) if FastAPI is installed
- Unit + e2e tests via `unittest`

## Project Layout
- `src/multi_agent/models.py` - task/event/step models
- `src/multi_agent/store.py` - state store
- `src/multi_agent/orchestrator.py` - orchestrator core
- `src/multi_agent/agents.py` - agent interfaces and built-ins
- `src/multi_agent/api.py` - service and optional HTTP app
- `src/multi_agent/main.py` - runtime entrypoint
- `tests/` - test suite
- `docs/plans/` - design, implementation and collection plans

## Run Tests
```bash
python3 -m unittest -v
```

## Run API (optional)
Install dependencies first:
```bash
python3 -m pip install fastapi uvicorn
```

Start server:
```bash
python3 -m src.multi_agent.main
```

Health endpoint:
```bash
curl http://127.0.0.1:8000/health
```

## Notes
- Current implementation is intentionally minimal and deterministic for iterative extension.
- The orchestration loop supports iterative execution up to `max_rounds` to prevent infinite stalls.

## Test Analysis Assistant MVP (New)
Core module lives in `src/test_analysis_assistant/` and supports two input formats:
- junit xml (`<testsuite>` / `<testsuites>`)
- pytest text output (failure sections from terminal logs)

It now also includes a retrieval foundation for AI-native analysis:
- multi-source ingestion (`code`, `repository`, `requirements`, `system analysis`, `knowledge`)
- mixed-modality normalization (`text`, `table`, `image` with OCR stub fallback)
- `MultiSourceIngestor` helper for repository crawling and markdown requirements ingestion (table/image extraction stubs)
- `ArtifactBundle` ingestion API for OCR/extraction pipeline outputs (single payload with text/tables/images + provenance metadata)
- payload-driven extraction confidence overrides per unit (`text`/`table`/`image`) for degraded-but-usable OCR/extraction stubs
- image OCR sidecar fallback for file-based ingestion (`<image>.ocr.txt` / `<image>.txt`) to upgrade stub-only paths without external OCR deps
- intent-aware query planning for test-gap/risk/root-cause workflows
- explainable ranking with score breakdown (`lexical`, `source`, `intent`, `modality`, `extraction`)
- diversity-aware final ranking to improve cross-source evidence coverage
- deterministic chunking, ranking, and confidence scoring with extraction-quality weighting
- retrieval confidence factors include `extraction_reliability` for multimodal evidence quality calibration
- confidence calibration now includes `cross_source_conflict` to penalize contradictory multisource evidence
- prompt builder that emits citation-ready context blocks

### Quick Start
```bash
# analyze report file and print JSON
python3 -m src.test_analysis_assistant.cli analyze --input /path/to/report.txt

# pretty JSON output
python3 -m src.test_analysis_assistant.cli analyze --input /path/to/junit.xml --format pretty
```

### Output Shape (example)
```json
{
  "input_format": "pytest_text",
  "total_failures": 2,
  "clusters": [
    {"cluster_id": "C01", "error_type": "AssertionError", "count": 1},
    {"cluster_id": "C02", "error_type": "ModuleNotFoundError", "count": 1}
  ],
  "root_cause_hypotheses": [
    "C01: AssertionError appears in 1 case(s), likely behavior mismatch..."
  ],
  "fix_suggestions": [
    {"priority": "P0", "title": "Address ModuleNotFoundError cluster"}
  ]
}
```

## Paper Retrieval Iteration (New)
- Core module: `src/multi_agent/paper_retrieval.py`
- Agent entrypoint: `paper_search` in `src/multi_agent/agents.py`
- E2E path: `OrchestratorService.register_builtin_agents()` includes `paper_search`

Run retrieval-focused tests:
```bash
python3 -m unittest tests/test_paper_retrieval.py tests/test_agents.py tests/test_e2e.py -v
```

## External Paper Corpus
You can load a real paper corpus from JSONL (`paper_id`, `title`, `abstract`, `keywords`):

```python
from src.multi_agent.api import OrchestratorService

service = OrchestratorService(max_rounds=100, paper_corpus_path="/path/to/papers.jsonl")
service.register_builtin_agents()
```

`paper_search` now uses a two-stage retriever by default:
1) first-stage hybrid retrieval
2) second-stage goal-aware reranking

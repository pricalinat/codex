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

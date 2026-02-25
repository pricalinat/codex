# Multi-Agent Core Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a runnable local prototype of a center-orchestrated multi-agent core with task planning, execution, retries, and status APIs.

**Architecture:** Use a thin API layer, a pure-Python orchestrator core, and pluggable agents/tools. Start with in-memory persistence and deterministic testable behavior, then expose orchestrator operations through FastAPI-compatible interfaces.

**Tech Stack:** Python 3.9+, FastAPI (API layer), unittest (tests), dataclasses/pydantic-style schemas.

---

### Task 1: Domain Models and In-Memory State Store

**Files:**
- Create: `src/multi_agent/models.py`
- Create: `src/multi_agent/store.py`
- Test: `tests/test_store.py`

**Step 1: Write the failing test**
- Create tests for task creation, event append, and status updates.

**Step 2: Run test to verify it fails**
- Run: `python3 -m unittest tests/test_store.py -v`
- Expected: FAIL because modules do not exist.

**Step 3: Write minimal implementation**
- Implement `TaskRecord`, `StepRecord`, `EventRecord` and `InMemoryStateStore` methods.

**Step 4: Run test to verify it passes**
- Run: `python3 -m unittest tests/test_store.py -v`
- Expected: PASS.

**Step 5: Commit**
- `git add tests/test_store.py src/multi_agent/models.py src/multi_agent/store.py`
- `git commit -m "feat: add domain models and in-memory store"`

### Task 2: Orchestrator State Machine and Retry Logic

**Files:**
- Create: `src/multi_agent/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Step 1: Write the failing test**
- Add tests for successful multi-step execution and retry-until-success behavior.

**Step 2: Run test to verify it fails**
- Run: `python3 -m unittest tests/test_orchestrator.py -v`
- Expected: FAIL because orchestrator is not implemented.

**Step 3: Write minimal implementation**
- Implement plan execution loop, step dependency checks, retry counters, terminal status.

**Step 4: Run test to verify it passes**
- Run: `python3 -m unittest tests/test_orchestrator.py -v`
- Expected: PASS.

**Step 5: Commit**
- `git add tests/test_orchestrator.py src/multi_agent/orchestrator.py`
- `git commit -m "feat: add orchestrator execution and retries"`

### Task 3: Agent Interface and Built-In Agents

**Files:**
- Create: `src/multi_agent/agents.py`
- Test: `tests/test_agents.py`

**Step 1: Write the failing test**
- Add tests for agent registration, dispatch, and standardized result envelope.

**Step 2: Run test to verify it fails**
- Run: `python3 -m unittest tests/test_agents.py -v`
- Expected: FAIL.

**Step 3: Write minimal implementation**
- Implement base protocol, registry, and two sample agents (`echo`, `summary`).

**Step 4: Run test to verify it passes**
- Run: `python3 -m unittest tests/test_agents.py -v`
- Expected: PASS.

**Step 5: Commit**
- `git add tests/test_agents.py src/multi_agent/agents.py`
- `git commit -m "feat: add agent interface and registry"`

### Task 4: API Layer and End-to-End Smoke Test

**Files:**
- Create: `src/multi_agent/api.py`
- Create: `src/multi_agent/main.py`
- Test: `tests/test_e2e.py`

**Step 1: Write the failing test**
- Add test that submits task, runs orchestration, and inspects final state/output.

**Step 2: Run test to verify it fails**
- Run: `python3 -m unittest tests/test_e2e.py -v`
- Expected: FAIL.

**Step 3: Write minimal implementation**
- Add API service methods and optional FastAPI app factory if dependency available.

**Step 4: Run test to verify it passes**
- Run: `python3 -m unittest tests/test_e2e.py -v`
- Expected: PASS.

**Step 5: Commit**
- `git add tests/test_e2e.py src/multi_agent/api.py src/multi_agent/main.py`
- `git commit -m "feat: expose orchestrator API and app entrypoint"`

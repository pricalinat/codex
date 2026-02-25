# OpenCode Ultrawork Playbook

## Runtime Check
- Local binary: `opencode 1.2.10`
- Validated local models in this environment:
  - `opencode/big-pickle` (works)
  - `opencode/gpt-5-nano` (works)
- Known broken path: custom `--agent` aliases mapped to unavailable `anthropic/*` models.

## Standard Launch (with proxy)
```bash
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 opencode
```
If your proxy is on `17890`, replace the port.

## Ultrawork Session Pattern
1. Plan (plan-only, no file edits):
```bash
opencode run -m opencode/big-pickle "Plan only. Do not edit files. Return steps + risks + test strategy for: <task>"
```
2. Implement with TDD:
```bash
opencode run -m opencode/big-pickle "Implement with TDD for: <task>. First add/adjust tests, then minimal code changes, then run tests."
```
3. Multi-agent orchestration:
```bash
opencode run -m opencode/big-pickle "/orchestrate <task>"
```
4. Quality gates:
```bash
opencode run -m opencode/big-pickle "Review changed files for correctness, regressions, and missing tests: <scope>"
opencode run -m opencode/big-pickle "Security review for changed files and input parsing surfaces: <scope>"
```
5. Verification:
```bash
python3 -m unittest -v
```

## Safety Guardrails
- Never use `/plan` without explicit "Do not edit files" prefix.
- Always pass `-m opencode/big-pickle` or another validated model.
- If a run produces malformed edits, restore file(s) before continuing and re-run from planning step.
- Prefer one focused task packet per run to reduce noisy large diffs.

## Task Packet Template
```text
[Ultrawork Task]
Goal:
Scope:
Files:
Success Criteria:
Commands:
Fallback:
```

#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/Users/rrp/Documents/codex"
cd "$REPO_DIR"
mkdir -p logs scripts

LOG="logs/iterate_test_assistant_100_$(date +%Y%m%d_%H%M%S).log"
PROMPT_FILE="/tmp/test_assistant_round_prompt.txt"
START_ROUND="${START_ROUND:-1}"
END_ROUND="${END_ROUND:-100}"

echo "[start] $(date)" | tee -a "$LOG"

run_with_fallback() {
  local prompt="$1"
  local out rc

  # 1) Claude (YOLO per user preference)
  set +e
  out=$(claude -p --permission-mode dontAsk --dangerously-skip-permissions --output-format text "$prompt" 2>&1)
  rc=$?
  set -e
  echo "$out" | tee -a "$LOG"
  if [[ $rc -eq 0 ]]; then
    echo "[agent] claude" | tee -a "$LOG"
    return 0
  fi

  # 2) Codex fallback
  set +e
  out=$(codex exec --full-auto "$prompt" 2>&1)
  rc=$?
  set -e
  echo "$out" | tee -a "$LOG"
  if [[ $rc -eq 0 ]]; then
    echo "[agent] codex" | tee -a "$LOG"
    return 0
  fi

  # 3) OpenCode fallback
  set +e
  out=$(opencode run "$prompt" 2>&1)
  rc=$?
  set -e
  echo "$out" | tee -a "$LOG"
  if [[ $rc -eq 0 ]]; then
    echo "[agent] opencode" | tee -a "$LOG"
    return 0
  fi

  return 1
}

for i in $(seq "$START_ROUND" "$END_ROUND"); do
  echo "[round $i] ===== $(date) =====" | tee -a "$LOG"

  cat > "$PROMPT_FILE" <<EOF
You are iterating this Test Analysis Assistant repository.
Round: $i / 100

Mission:
Build a production-grade AI-native test analysis assistant that can ingest:
- raw code snippets, full code repositories, requirement docs, system analysis docs
- mixed content with tables/images (via OCR/extraction pipeline stubs if needed)
- background knowledge corpora
Then generate robust test analysis outputs: failure clustering, root-cause hypotheses, test-gap analysis, risk-based prioritization, and actionable plans.

Priority for this round:
1) one meaningful increment with runnable code + tests.
2) improve architecture for multimodal/multisource ingestion and retrieval.
3) improve AI leverage design (RAG interfaces, chunking, ranking, prompt strategy, confidence scoring).
4) if stuck, inspect and adapt ideas from open-source approaches like codewiki-style repo understanding, then implement concretely.
5) if direct integration is hard, implement degraded but usable fallback path.

Constraints:
- Keep current features working.
- Prefer Python stdlib; minimal deps.
- Update README briefly only when needed.
- Run tests and report results.
- Conclude with concise summary: changed files, why it matters, commands run, test status, next step.
EOF

  if ! run_with_fallback "$(cat "$PROMPT_FILE")"; then
    echo "[round $i] all agents failed" | tee -a "$LOG"
  fi

  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi

  set +e
  PYTHONPATH=. .venv/bin/python -m unittest -v 2>&1 | tee -a "$LOG"
  TEST_RC=${PIPESTATUS[0]}
  set -e
  echo "[round $i] test_rc=$TEST_RC" | tee -a "$LOG"

  if [[ -n "$(git status --porcelain)" ]]; then
    git add -A
    git commit -m "iter: test analysis assistant round $i"
    set +e
    git push origin main 2>&1 | tee -a "$LOG"
    PUSH_RC=${PIPESTATUS[0]}
    set -e
    echo "[round $i] push_rc=$PUSH_RC" | tee -a "$LOG"
  else
    echo "[round $i] no file changes; skip commit/push" | tee -a "$LOG"
  fi
done

echo "[done] $(date)" | tee -a "$LOG"

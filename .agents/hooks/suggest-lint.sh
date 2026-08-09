#!/usr/bin/env bash
# stop: if Rust sources differ from HEAD, remind the agent to run /lint (fail open).
set -euo pipefail

# Default: no follow-up.
empty='{}'

if ! command -v git >/dev/null 2>&1; then
  printf '%s\n' "$empty"
  exit 0
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf '%s\n' "$empty"
  exit 0
fi

changed="$(
  {
    git diff --name-only HEAD 2>/dev/null || true
    git diff --name-only --cached 2>/dev/null || true
    git ls-files --others --exclude-standard 2>/dev/null || true
  } | grep -E '\.rs$' || true
)"

if [[ -z "$changed" ]]; then
  printf '%s\n' "$empty"
  exit 0
fi

if ! command -v jq >/dev/null 2>&1; then
  printf '%s\n' "$empty"
  exit 0
fi

jq -n \
  --arg followup_message "Rust sources changed in this session. Run /lint (nightly fmt + clippy) before finishing if you have not already." \
  '{followup_message:$followup_message}'
exit 0

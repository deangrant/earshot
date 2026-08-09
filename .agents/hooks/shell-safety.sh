#!/usr/bin/env bash
# beforeShellExecution: block destructive git and model-bin commits (fail open on parse errors).
set -euo pipefail

allow() {
  printf '%s\n' '{"permission":"allow"}'
  exit 0
}

ask() {
  local user_message="$1"
  local agent_message="$2"
  jq -n \
    --arg user_message "$user_message" \
    --arg agent_message "$agent_message" \
    '{permission:"ask", user_message:$user_message, agent_message:$agent_message}'
  exit 0
}

deny() {
  local user_message="$1"
  local agent_message="$2"
  jq -n \
    --arg user_message "$user_message" \
    --arg agent_message "$agent_message" \
    '{permission:"deny", user_message:$user_message, agent_message:$agent_message}'
  exit 0
}

if ! command -v jq >/dev/null 2>&1; then
  allow
fi

input="$(cat)"
command="$(printf '%s' "$input" | jq -r '.command // empty')"

if [[ -z "$command" ]]; then
  allow
fi

# Destructive git
if [[ "$command" =~ git[[:space:]]+push[[:space:]]+.*(--force|-f)([[:space:]]|$) ]]; then
  deny \
    "Blocked force-push. Re-run only if you explicitly intend to rewrite the remote." \
    "Hook denied a git force-push."
fi

if [[ "$command" =~ git[[:space:]]+reset[[:space:]]+.*--hard ]]; then
  deny \
    "Blocked git reset --hard. Use a safer recovery path or confirm with the user first." \
    "Hook denied git reset --hard."
fi

if [[ "$command" =~ git[[:space:]]+clean[[:space:]]+.*-[a-zA-Z]*f ]]; then
  deny \
    "Blocked destructive git clean. Avoid wiping the worktree from the agent." \
    "Hook denied git clean -f."
fi

# Staging or committing model weights / large binaries
if [[ "$command" =~ git[[:space:]]+(add|commit|restore[[:space:]]+--staged) ]] &&
  [[ "$command" =~ \.(bin|ggml|gguf)([[:space:]]|$) || "$command" =~ (^|[[:space:]])ggml- ]]; then
  ask \
    "This git command may stage or commit model weight files (*.bin / ggml). Confirm before continuing." \
    "Hook flagged a git command that may include model binaries."
fi

allow

#!/usr/bin/env bash
# beforeReadFile: warn when reading model weight / binary artifacts (fail open).
set -euo pipefail

allow() {
  printf '%s\n' '{"permission":"allow"}'
  exit 0
}

if ! command -v jq >/dev/null 2>&1; then
  allow
fi

input="$(cat)"
file_path="$(printf '%s' "$input" | jq -r '.file_path // .path // empty')"

if [[ -z "$file_path" ]]; then
  allow
fi

base="$(basename "$file_path")"
if [[ "$base" == *.bin || "$base" == *.ggml || "$base" == *.gguf || "$base" == ggml-* ]]; then
  jq -n \
    --arg agent_message "Reading model/binary artifact '${base}'. Treat GGML weights as a trust boundary; do not commit them." \
    '{permission:"allow", agent_message:$agent_message}'
  exit 0
fi

allow

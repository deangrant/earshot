#!/usr/bin/env bash
# sessionStart: inject short earshot working constraints (fail open).
set -euo pipefail

context='Earshot: use cargo +nightly fmt; default CI omits CUDA (--all-features); GGML .bin models are a native trust boundary; agent tooling lives under .agents/.'

if command -v jq >/dev/null 2>&1; then
  jq -n --arg additional_context "$context" '{additional_context:$additional_context}'
else
  printf '%s\n' '{}'
fi

exit 0
